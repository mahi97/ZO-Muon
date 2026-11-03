"""
[inverse-metric] LOREN on the MeZO harness.

Port of loren/model/glue_transformer_loren.py (Seung, Lee & Ko, AAAI-26): configure_params L69-84, perturb_parameters
L86-100, lnes_zo_update L102-133, training_step L135-159. Why a port: the released harness uses a randomly initialised
classification head, first-N subsets, no test split, no MultiRC loader and no Qwen2 support, so its numbers cannot be
compared on the ZO-LLM protocol. The optimizer itself is copied line by line, INCLUDING two quirks of the released code:
  * weight decay is effectively 0.01 whatever --weight_decay says: loren.py:39-56 passes weight_decay positionally
    into LoZo's step_interval slot, so LoZo's default 0.01 stays. Here: --loren_weight_decay, default 0.01.
    It applies to all params and to the a vectors, K times per step.
  * for 1-D params ak_grad.view(shape).sum(0) is a scalar, so a only shifts uniformly.
Other released behaviour kept: momentum beta 0.9 stepped K times per training step on params with ndim >= 2,
one-sided queries with RLOO baseline (no f(theta)), a initialised from randn with the global RNG.
Hyperparameters from the README: damping 0.1, lr_cov 1e-3. K is unspecified for LOREN in the README: main.py's default is 10.
Extra memory: 1x parameters (momentum on ndim >= 2) plus the a vectors and transient temporaries.
"""
import math

import numpy as np
import torch

from MeZOTrainer import MeZOTrainer


class LORENTrainer(MeZOTrainer):

    def _configure(self, model):
        a_ = self.args
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.damping, self.lr_cov, self.beta1 = a_.loren_damping, a_.loren_lr_cov, 0.9
        self.d_sqrt = math.sqrt(self.damping)
        self.num_queries = a_.loren_num_queries
        assert self.num_queries >= 2, "At least 2 samples are required for RLOO estimation"
        self.loren_wd = a_.loren_weight_decay
        self.evec_a, self.momentum, self.param_info = {}, {}, {}
        for name, param in self.params:
            evec_size = param.size(1) if param.ndim >= 2 else param.numel()
            self.evec_a[name] = torch.randn(evec_size, device=param.device, dtype=param.dtype)
            if param.ndim >= 2:
                self.momentum[name] = torch.zeros_like(param)
            self.param_info[name] = {"shape": param.shape, "mfactor": param.numel() // evec_size}

    def _sqrt_cov_u(self, name, param):
        info = self.param_info[name]
        a_flat = self.evec_a[name].view(1, -1).expand(info["mfactor"], -1).reshape(-1)
        u_vec = torch.randn_like(param).view(-1)
        sq_norm_a = torch.dot(a_flat, a_flat)
        dot_au = torch.dot(a_flat, u_vec)
        dsna = math.sqrt(self.damping + sq_norm_a.item())
        alpha = (self.d_sqrt + dsna) * dot_au / (sq_norm_a * dsna)
        return u_vec.sub(alpha * a_flat).view(info["shape"]), a_flat, u_vec, sq_norm_a, dot_au, dsna

    @torch.no_grad()
    def _perturb(self, seed, scaling_factor):
        torch.manual_seed(seed)
        for name, param in self.params:
            s = self._sqrt_cov_u(name, param)[0]
            param.add_(s, alpha=scaling_factor * self.args.zo_eps)

    def zo_step(self, model, inputs):
        if not hasattr(self, "evec_a"):
            self._configure(model)
        self.seeds = [np.random.randint(1000000000) for _ in range(self.num_queries)]
        self.fvals = []
        losses = []
        for k in range(self.num_queries):
            self._perturb(self.seeds[k], 1)
            loss = self.zo_forward(model, inputs)
            self.fvals.append(loss.item())
            losses.append(loss)
            self._perturb(self.seeds[k], -1)
        assert self.args.gradient_accumulation_steps == 1
        return torch.stack(losses).mean()

    @torch.no_grad()
    def zo_update(self, model):
        eps = self.args.zo_eps
        lr = self._get_learning_rate()
        K = self.num_queries
        fmean = float(np.mean(self.fvals))
        dev = self.params[0][1].device
        fweight = torch.tensor(self.fvals, device=dev) - fmean
        q_inv = 1.0 / (K - 1)
        eps_inv = 1.0 / (eps * (K - 1))
        for k in range(K):
            torch.manual_seed(self.seeds[k])
            for name, param in self.params:
                s, a_flat, u_vec, sq_norm_a, dot_au, dsna = self._sqrt_cov_u(name, param)
                gx = s.mul(fweight[k] * eps_inv)
                if param.ndim >= 2:
                    self.momentum[name].mul_(self.beta1).add_(gx)
                    gx = self.momentum[name]
                param.mul_(1 - lr * self.loren_wd)
                param.add_(gx, alpha=-lr)
                c1 = dot_au * u_vec - a_flat
                c2 = (self.d_sqrt + dsna) * (dot_au ** 2 - sq_norm_a) * a_flat / (sq_norm_a * dsna)
                ak_grad = fweight[k] * (eps ** 2) * (c1 - c2) / dsna
                ak_grad = ak_grad.view(self.param_info[name]["shape"]).sum(0) * q_inv
                a = self.evec_a[name]
                a.mul_(1 - self.lr_cov * self.loren_wd)
                a.add_(ak_grad, alpha=-self.lr_cov)
        self.lr_scheduler.step()

    def extra_results(self):
        a = self.args
        return {"forwards_per_step": a.loren_num_queries, "loren_num_queries": a.loren_num_queries,
                "loren_damping": a.loren_damping, "loren_lr_cov": a.loren_lr_cov, "loren_weight_decay": a.loren_weight_decay}
