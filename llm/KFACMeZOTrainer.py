"""
[inverse-metric] K-FAC-shaped MeZO ("sample with the inverse metric").

KFACMeZOTrainer(MeZOTrainer) keeps MeZO's training loop, seed stream and two-point SPSA estimate and only
changes the perturbation / update direction of the nn.Linear weights inside the transformer blocks.

K-FAC factors (per block-linear weight W, out x in):
    A = E[a a^T]  over non-pad tokens of the layer input a (in x in)
    G = sum_t g_t g_t^T / N_examples, g = d loss_example / d (layer output) (out x out)   ("KFAC-expand")
  accumulated block-diagonally (block b, default 512) in fp32 from `kfac_calib_steps` backprop minibatches of
  `kfac_calib_bs` examples of the SAME training set. "true" Fisher: the label (option) is sampled from the model's
  own predictive distribution over options (softmax of the length-normalized option log-likelihoods, i.e. the
  distribution the training CE uses); "empirical": gold label. Only output gradients are computed
  (torch.autograd.grad w.r.t. the input-embedding output), no parameter gradients are materialized.
  Each block is eigendecomposed; damping delta = gamma * mean eigenvalue of that factor.

Placement A (kfac_placement="A", power p):
    z ~ N(0, I) drawn exactly as MeZO (torch.manual_seed(step_seed), one torch.normal per parameter in
    model.named_parameters() order, param dtype) -> for K-FAC'd weights E = c * G_p z A_p (blockwise, eigenbasis
    powers G_p = Q (lam+delta)^p Q^T), other params (embeddings, positions, 1-D) keep E = z.
    c is ONE global constant with c^2 * sum_l tr(G_p^2) tr(A_p^2) = sum_l m_l n_l, i.e. E||E||_F^2 over the
    K-FAC'd weights equals the isotropic value (so zo_eps keeps its meaning).
    Cov(vec E) = c^2 (A^{2p} kron G^{2p});  p=-1/2 -> K-FAC^-1, p=-1/4 -> K-FAC^-1/2, p=+1/2 -> K-FAC^+1.
    theta += eps E ; theta -= 2 eps E ; theta += eps E ; theta -= lr (g E + wd theta).   (p = 0 == MeZO, bit-exact)
Placement B (kfac_placement="B"): isotropic z for the perturbation (== MeZO), the update applies
    P = c^2 (G^{2p} (.) A^{2p}) (same constant c, so E[step] equals placement A's) to g z of the K-FAC'd weights.
"""
import json
import math
import os
import random
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from MeZOTrainer import MeZOTrainer
from utils import option_logits

logger = logging.getLogger(__name__)


def _blocks(d, b):
    bs = min(b, d)
    nb = (d + bs - 1) // bs
    return bs, nb, nb * bs - d  # block size, #blocks, padding


def _masked_flat(x, mask):
    """(.., d) -> (N, d) keeping non-pad tokens; mask (rows, T) in the same row-major order as x."""
    x = x.reshape(-1, x.size(-1))
    if mask is None:
        return x
    m = mask.reshape(-1)
    assert m.numel() == x.size(0), f"mask {tuple(mask.shape)} vs activations {tuple(x.shape)}"
    return x[m]


def _spec_stats(lam, delta):
    """Spectrum summary of a (block-diagonal) factor; lam = all real eigenvalues (fp64 numpy, clamped >= 0)."""
    lam = np.sort(np.clip(lam, 0, None))[::-1]
    d = lam.size
    tot = lam.sum()
    p = lam / tot if tot > 0 else np.full(d, 1.0 / d)
    nz = p[p > 0]
    k1 = max(1, int(math.ceil(0.01 * d)))
    k10 = max(1, int(math.ceil(0.10 * d)))
    lmin_pos = lam[lam > 0].min() if (lam > 0).any() else 0.0
    return {
        "dim": int(d),
        "lam_max": float(lam[0]), "lam_min": float(lam[-1]), "lam_mean": float(lam.mean()),
        "cond": float(lam[0] / lam[-1]) if lam[-1] > 0 else float("inf"),
        "cond_pos": float(lam[0] / lmin_pos) if lmin_pos > 0 else float("inf"),
        "cond_damped": float((lam[0] + delta) / (lam[-1] + delta)),
        "erank_entropy": float(np.exp(-(nz * np.log(nz)).sum())),  # Roy & Vetterli effective rank
        "erank_participation": float(tot ** 2 / (lam ** 2).sum()) if tot > 0 else 0.0,
        "frac_trace_top1pct": float(lam[:k1].sum() / tot) if tot > 0 else 0.0,
        "frac_trace_top10pct": float(lam[:k10].sum() / tot) if tot > 0 else 0.0,
        "delta": float(delta),
    }


class KFACMeZOTrainer(MeZOTrainer):

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        args = self.args
        assert args.kfac_placement in ("A", "B"), args.kfac_placement
        assert args.kfac_fisher in ("true", "empirical"), args.kfac_fisher
        assert getattr(args, "sparsity", 1.0) >= 1.0, "sparsity not supported with kfac_mezo"
        self.kfac = None  # name -> dict(Gb, Ab, out, in, ...)
        self.kfac_c = None
        self.kfac_backprop_examples = 0
        self.kfac_backprop_passes = 0
        self.kfac_calibs = []
        self.kfac_last_calib_step = None
        self._calib_seconds_pending = 0.0
        self.restore_checks = []
        self.pg_abs = []
        self.pg_decomp = []
        self._E = None
        assert args.kfac_scope in ("all", "weights", "rest"), args.kfac_scope
        assert args.kfac_scope == "all" or args.kfac_cache_E, "kfac_scope needs the cached E"
        # scope=rest perturbs no K-FAC'd weight, so it needs no factors
        self.kfac_identity = (args.kfac_power == 0.0 or args.kfac_scope == "rest")
        self._kfac_names = None

    # ------------------------------------------------------------------ factor estimation
    def _target_linears(self, model):
        out = {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and ".layers." in name and mod.weight.requires_grad and mod.weight.dim() == 2:
                out[name] = mod
        return out

    @torch.no_grad()
    def _acc_blocks(self, S, X, b):
        """S (nb, bs, bs) += blockdiag(X^T X); X (N, d) fp32."""
        N, d = X.shape
        bs, nb, pad = _blocks(d, b)
        if pad:
            X = F.pad(X, (0, pad))
        Xb = X.view(N, nb, bs).transpose(0, 1)  # (nb, N, bs)
        S.add_(torch.bmm(Xb.transpose(1, 2), Xb))

    def _calib_loss(self, model, inputs, gen):
        """Loss whose output-gradients define the Fisher. Returns (loss, n_examples)."""
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        option_len = inputs["option_len"]
        num_options = inputs.get("num_options")
        # only the option positions go through lm_head (always: long-context tasks would not fit in backprop otherwise)
        shift_logits = option_logits(model, input_ids, attn, option_len).float()
        shift_labels = input_ids[:, -shift_logits.size(1):].clone()
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for i, l in enumerate(option_len.tolist() if torch.is_tensor(option_len) else option_len):
            mask[i, -int(l):] = True
        mask &= shift_labels != model.config.pad_token_id  # same masking as forward_wrap_with_option_len
        if num_options is not None:
            logp = F.log_softmax(shift_logits, dim=-1)
            sel = torch.gather(logp, -1, (shift_labels * mask).unsqueeze(-1)).squeeze(-1)
            scores = (sel * mask).sum(-1) / mask.sum(-1)  # (rows,)
            nopt = num_options.tolist() if torch.is_tensor(num_options) else list(num_options)
            labels = inputs["labels"]
            losses, start = [], 0
            while start < len(nopt):
                n = int(nopt[start])
                s = scores[start:start + n]
                if self.args.kfac_fisher == "true":
                    with torch.no_grad():
                        y = torch.multinomial(F.softmax(s.detach(), -1), 1, generator=gen).squeeze()
                else:
                    y = labels[start]
                losses.append(F.cross_entropy(s.unsqueeze(0), y.view(1)))
                start += n
            return torch.stack(losses).mean(), len(losses)
        else:
            # LM loss on the option tokens (train_as_classification False, e.g. Copa)
            tgt = shift_labels.clone()
            if self.args.kfac_fisher == "true":
                with torch.no_grad():
                    probs = F.softmax(shift_logits[mask], -1)
                    tgt[mask] = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
            tgt[~mask] = -100
            # mean over the rows' examples: per-example loss = mean over its option tokens (as the ZO objective's
            # token-mean CE over the batch is close to this); one example per row
            tok = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), tgt.view(-1),
                                  ignore_index=-100, reduction="none").view(tgt.shape)
            per_ex = (tok * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            return per_ex.mean(), int(input_ids.size(0))

    def _calib_groups(self, generator):
        """kfac_calib_steps * kfac_calib_bs examples in sampler order, grouped into backward passes of at most kfac_calib_max_tokens
        padded tokens (and kfac_calib_microbs examples). The statistics are sums, so grouping changes memory, not the result
        beyond rounding. Backprop keeps ~3-4 MB per token at 1.3-1.5B, so long tasks cannot take a batch of 16 in one pass."""
        args = self.args
        total = args.kfac_calib_steps * args.kfac_calib_bs
        ds = self.train_dataset
        seen, cur, rows, maxlen = 0, [], 0, 0
        while seen < total:
            for i in RandomSampler(ds, generator=generator):
                if seen >= total:
                    break
                item = ds[int(i)]
                opts = item if isinstance(item, list) else [item]
                r = len(opts)
                L = int(math.ceil(max(len(o["input_ids"]) for o in opts) / 8) * 8)
                if cur and ((rows + r) * max(maxlen, L) > args.kfac_calib_max_tokens or len(cur) >= args.kfac_calib_microbs):
                    yield cur
                    cur, rows, maxlen = [], 0, 0
                cur.append(item); rows += r; maxlen = max(maxlen, L); seen += 1
        if cur:
            yield cur

    def kfac_calibrate(self, model):
        args = self.args
        t0 = time.time()
        b = args.kfac_block
        rng_state = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(), torch.cuda.get_rng_state_all())
        n_cal = len(self.kfac_calibs)
        seed = int(args.seed) * 1000003 + 7919 * n_cal + 17
        g_cpu = torch.Generator(device="cpu"); g_cpu.manual_seed(seed)
        dev = next(model.parameters()).device
        g_dev = torch.Generator(device=dev); g_dev.manual_seed(seed)

        targets = self._target_linears(model)
        stats = {}
        for name, mod in targets.items():
            o, i = mod.weight.shape
            bi, nbi, _ = _blocks(i, b); bo, nbo, _ = _blocks(o, b)
            stats[name] = {"A": torch.zeros(nbi, bi, bi, device=mod.weight.device, dtype=torch.float32), "nA": 0,
                           "G": torch.zeros(nbo, bo, bo, device=mod.weight.device, dtype=torch.float32), "nG": 0}
        ctx = {"mask": None, "scale": 1.0, "emb": None}
        handles = []

        def fwd_hook(name):
            def hook(mod, inp, out):
                x = inp[0].detach()
                m = ctx["mask"]
                # OPT flattens (rows, T, d) -> (rows*T, d) before fc1/fc2, so always index with the flattened mask
                X = _masked_flat(x, m)
                self._acc_blocks(stats[name]["A"], X.float(), b)
                stats[name]["nA"] += X.size(0)
                if out.requires_grad:
                    def ghook(grad):
                        Gx = _masked_flat(grad.detach(), m)
                        self._acc_blocks(stats[name]["G"], Gx.float() * ctx["scale"], b)
                    out.register_hook(ghook)
            return hook

        for name, mod in targets.items():
            handles.append(mod.register_forward_hook(fwd_hook(name)))
        emb = model.get_input_embeddings()
        handles.append(emb.register_forward_hook(lambda m, i, o: ctx.__setitem__("emb", o)))

        n_ex = 0
        model.eval()  # same mode as the ZO forward (no dropout)
        try:
            with torch.enable_grad():
                for batch_items in self._calib_groups(g_cpu):
                    batch = self._prepare_inputs(self.data_collator(batch_items))
                    am = batch.get("attention_mask")
                    ctx["mask"] = am.bool() if am is not None else None
                    loss, nex = self._calib_loss(model, batch, g_dev)
                    ctx["scale"] = float(nex)  # grad of mean loss * n_examples = per-example-loss gradient
                    torch.autograd.grad(loss, [ctx["emb"]])
                    ctx["emb"] = None
                    n_ex += nex
                    self.kfac_backprop_passes += 1
        finally:
            for h in handles:
                h.remove()
            model.eval()
        torch.cuda.empty_cache()  # the autograd graphs are gone; give the eigensolver room
        torch.cuda.synchronize()
        t_stats = time.time() - t0

        # ---- eigendecomposition, damping, powers, global normalization
        p = float(args.kfac_power)
        q = p if args.kfac_placement == "A" else 2.0 * p
        gamma = float(args.kfac_damping)
        factors, spectra = {}, {}
        num, den = 0.0, 0.0
        for name, st in stats.items():
            o, i = targets[name].weight.shape
            ent = {"out": o, "in": i}
            for side, d, cnt in (("A", i, st["nA"]), ("G", o, n_ex)):
                S = st[side] / max(cnt, 1)
                bs, nb, pad = _blocks(d, b)
                Sd = S.double() if args.kfac_eigh_fp64 else S
                Sd = 0.5 * (Sd + Sd.transpose(1, 2))
                parts = []  # (block offset, eigenvalues (k, r), eigenvectors (k, r, r))
                n_full = nb if pad == 0 else nb - 1
                if n_full > 0:
                    lam, Q = torch.linalg.eigh(Sd[:n_full])
                    parts.append((0, lam, Q))
                if pad:
                    r = bs - pad
                    lam_r, Q_r = torch.linalg.eigh(Sd[n_full, :r, :r])
                    parts.append((n_full, lam_r.unsqueeze(0), Q_r.unsqueeze(0)))
                all_lam = torch.cat([l.reshape(-1) for _, l, _ in parts]).clamp(min=0)
                mean_lam = float(all_lam.mean())
                delta = gamma * mean_lam if mean_lam > 0 else gamma
                spectra[f"{name}.{side}"] = _spec_stats(all_lam.cpu().numpy().astype(np.float64), delta)
                P = torch.zeros(nb, bs, bs, dtype=Sd.dtype, device=Sd.device)
                tr2 = 0.0
                for off, lam, Q in parts:
                    lp = lam.clamp(min=0) + delta
                    Pq = (Q * lp.pow(q).unsqueeze(-2)) @ Q.transpose(-1, -2)
                    k, r = Q.shape[0], Q.shape[-1]
                    P[off:off + k, :r, :r] = Pq
                    if r < bs:  # identity on the padded coordinates (they only ever multiply zeros)
                        P[off:off + k, r:, r:] = torch.eye(bs - r, dtype=P.dtype, device=P.device)
                    tr2 += float(lp.pow(2.0 * p).sum())  # tr(F_p^2) with p the SAMPLING power (normalization)
                ent[side] = P.float()
                ent["tr2_" + side] = tr2
                del Sd, S
            num += o * i
            den += ent["tr2_A"] * ent["tr2_G"]
            factors[name + ".weight"] = ent
            st["A"] = None; st["G"] = None
        del stats
        c2 = num / den  # c^2: E||c G_p Z A_p||^2 summed over K-FAC'd weights == sum m n
        cfac = math.sqrt(c2) if args.kfac_placement == "A" else c2
        energy = {}
        for name, ent in factors.items():
            ent["G"].mul_(cfac)  # fold c (A) or c^2 (B) into the G blocks
            energy[name] = c2 * ent["tr2_A"] * ent["tr2_G"] / (ent["out"] * ent["in"])  # per-weight E||E||^2/(mn)
        self.kfac = factors
        self.kfac_c = math.sqrt(c2)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        secs = time.time() - t0

        random.setstate(rng_state[0]); np.random.set_state(rng_state[1])
        torch.random.set_rng_state(rng_state[2]); torch.cuda.set_rng_state_all(rng_state[3])

        self.kfac_backprop_examples += n_ex
        rec = {"step": int(self.state.global_step), "examples": n_ex, "seconds": secs, "seconds_stats": t_stats,
               "c": self.kfac_c, "n_weights": len(factors),
               "energy_ratio_min": float(min(energy.values())), "energy_ratio_max": float(max(energy.values()))}
        self.kfac_calibs.append(rec)
        logger.info(f"[kfac] calibration {n_cal}: {rec}")
        # spectra of every calibration (later ones get a _calN suffix)
        fn = args.kfac_spectra_file or os.path.join(args.output_dir, "kfac_spectra.json")
        if n_cal > 0:
            root, ext = os.path.splitext(fn); fn = f"{root}_cal{n_cal}{ext}"
        os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
        with open(fn, "w") as f:
            json.dump({"calibration": rec, "model": args.model_name, "task": args.task_name,
                       "fisher": args.kfac_fisher, "damping_rel": gamma, "block": b, "power": p,
                       "placement": args.kfac_placement, "summary": self._summarize_spectra(spectra),
                       "energy_ratio_per_weight": energy, "factors": spectra}, f, indent=1)
        rec["spectra_file"] = fn
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"kfac/calib_seconds": secs, "kfac/backprop_examples": self.kfac_backprop_examples,
                           "kfac/c": self.kfac_c, "train/global_step": self.state.global_step},
                          step=int(self.state.global_step))
        except Exception:
            pass
        self._calib_seconds_pending += secs
        return rec

    @staticmethod
    def _summarize_spectra(spectra):
        groups = {}
        for k, v in spectra.items():
            parts = k.split(".")
            key = parts[-2] + "." + parts[-1]  # e.g. q_proj.A
            groups.setdefault(key, []).append(v)
        out = {}
        for key, vs in sorted(groups.items()):
            out[key] = {m: {"median": float(np.median([v[m] for v in vs])), "min": float(np.min([v[m] for v in vs])),
                            "max": float(np.max([v[m] for v in vs]))}
                        for m in ["cond_pos", "cond_damped", "erank_entropy", "erank_participation",
                                  "frac_trace_top1pct", "frac_trace_top10pct", "dim"]}
        return out

    # ------------------------------------------------------------------ transforms
    @torch.no_grad()
    def _transform(self, name, Z):
        """blockdiag(Gb) @ Z @ blockdiag(Ab) (constants already folded into Gb). Z (out, in) -> same dtype."""
        f = self.kfac[name]
        Gb, Ab = f["G"], f["A"]
        o, i = Z.shape
        bo, nbo, po = _blocks(o, self.args.kfac_block)
        bi, nbi, pi = _blocks(i, self.args.kfac_block)
        Zf = Z.float()
        if po or pi:
            Zf = F.pad(Zf, (0, pi, 0, po))
        T = torch.matmul(Gb, Zf.view(nbo, bo, nbi * bi))  # (nbo, bo, in_p)
        T = T.view(nbo * bo, nbi, bi).transpose(0, 1)  # (nbi, out_p, bi)
        E = torch.matmul(T, Ab)  # (nbi, out_p, bi)
        E = E.transpose(0, 1).reshape(nbo * bo, nbi * bi)
        if po or pi:
            E = E[:o, :i]
        return E.to(Z.dtype).contiguous()

    # ------------------------------------------------------------------ ZO step / update
    def _maybe_calibrate(self, model):
        gs = self.state.global_step
        need = self.kfac is None and not self.kfac_identity
        if (not need and not self.kfac_identity and self.args.kfac_refresh > 0 and gs > 0
                and gs % self.args.kfac_refresh == 0 and self.kfac_last_calib_step != gs):
            need = True
        if need:
            self.kfac = None
            torch.cuda.empty_cache()
            self.kfac_calibrate(model)
            self.kfac_last_calib_step = gs

    def _draw(self):
        """z for every param in MeZO order from the step seed; transformed (placement A) for K-FAC'd weights."""
        torch.manual_seed(self.zo_random_seed)
        out = []
        scope = self.args.kfac_scope
        for name, param in self.named_parameters_to_optim:
            # always drawn, so the RNG stream does not depend on the scope; None = this parameter is frozen for this run
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if scope != "all" and (name in self._kfac_names) != (scope == "weights"):
                z = None
            elif self.args.kfac_placement == "A" and not self.kfac_identity and name in self.kfac:
                z = self._transform(name, z)
            out.append(z)
        return out

    @torch.no_grad()
    def _pg_decompose(self, model, inputs):
        """Diagnostic: two-point directional derivative along E restricted to subsets of the parameters, same batch and draw.
        kfac = the K-FAC'd weights with this arm's E; rest = embeddings, positions, 1-D params (isotropic in every arm);
        kfac_iso = the K-FAC'd weights with the isotropic z of the same seed. Needs the cached E. Restores by exact copy."""
        assert self._E is not None and self.kfac is not None
        eps = self.args.zo_eps
        torch.manual_seed(self.zo_random_seed)
        Z = [torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype) for _, p in self.named_parameters_to_optim]
        items = list(zip(self.named_parameters_to_optim, self._E, Z))

        def two_point(select):
            chosen = [(p, d) for ((n, p), E, z) in items for d in [select(n, E, z)] if d is not None]
            keep = [p.data for p, _ in chosen]
            for (p, d), w in zip(chosen, keep):
                p.data = w + eps * d
            l1 = self.zo_forward(model, inputs).item()
            for (p, d), w in zip(chosen, keep):
                p.data = w - eps * d
            l2 = self.zo_forward(model, inputs).item()
            for (p, _), w in zip(chosen, keep):
                p.data = w
            return (l1 - l2) / (2 * eps)

        self.pg_decomp.append({
            "kfac": two_point(lambda n, E, z: E if n in self.kfac else None),
            "rest": two_point(lambda n, E, z: E if n not in self.kfac else None),
            "kfac_iso": two_point(lambda n, E, z: z if n in self.kfac else None),
        })

    @torch.no_grad()
    def _record_E_stats(self):
        """One-sample check of the normalization on the first step after a calibration (placement A: ||E||^2 / sum mn ~ 1)."""
        e2, mn, mx, per = 0.0, 0, 0.0, {}
        for (name, _), E in zip(self.named_parameters_to_optim, self._E):
            if name in self.kfac:
                v = float((E.double() ** 2).sum())
                e2 += v; mn += E.numel(); mx = max(mx, float(E.abs().max()))
                t = name.split(".")[-2]
                a = per.setdefault(t, [0.0, 0]); a[0] += v; a[1] += E.numel()
        self.kfac_calibs[-1]["E_norm2_over_mn_sample"] = e2 / mn
        self.kfac_calibs[-1]["E_max_abs_sample"] = mx
        self.kfac_calibs[-1]["E_norm2_over_mn_by_type"] = {k: a[0] / a[1] for k, a in per.items()}
        logger.info(f"[kfac] E sample: ||E||^2/mn={e2 / mn:.4f} max|E|={mx:.3f} by type={self.kfac_calibs[-1]['E_norm2_over_mn_by_type']}")

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        assert random_seed is None
        if self._E is not None:
            for (name, param), E in zip(self.named_parameters_to_optim, self._E):
                if E is None:
                    continue
                param.data = param.data + scaling_factor * E * self.args.zo_eps  # same expression as MeZO
        else:
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if self.args.kfac_placement == "A" and not self.kfac_identity and name in self.kfac:
                    z = self._transform(name, z)
                param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_step(self, model, inputs):
        self._maybe_calibrate(model)
        self.named_parameters_to_optim = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if self._kfac_names is None:
            self._kfac_names = {n + ".weight" for n in self._target_linears(model)}
        self.zo_random_seed = np.random.randint(1000000000)
        snap = None
        if self.args.kfac_check_restore > 0 and len(self.restore_checks) < self.args.kfac_check_restore:
            snap = [p.data.detach().to("cpu", copy=True) for _, p in self.named_parameters_to_optim]
        self._E = self._draw() if self.args.kfac_cache_E else None
        if self._E is not None and self.kfac is not None and self.kfac_last_calib_step == self.state.global_step:
            self._record_E_stats()
        if self.args.kfac_pg_decomp > 0 and len(self.pg_decomp) < self.args.kfac_pg_decomp:
            self._pg_decompose(model, inputs)

        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.pg_abs.append(abs(self.projected_grad))
        assert self.args.gradient_accumulation_steps == 1
        self.zo_perturb_parameters(scaling_factor=1)

        if snap is not None:
            mx, mx_rel, mx_kf, ulp = 0.0, 0.0, 0.0, 0.0
            for (name, p), s in zip(self.named_parameters_to_optim, snap):
                d = (p.data.detach().cpu() - s).abs()
                m = float(d.max())
                mx = max(mx, m)
                if self.kfac is not None and name in self.kfac:
                    mx_kf = max(mx_kf, m)
                mx_rel = max(mx_rel, float((d / s.abs().clamp(min=1e-12)).max()))
                ulp = max(ulp, float(torch.finfo(s.dtype).eps * s.abs().max()))
            self.restore_checks.append({"step": int(self.state.global_step), "max_abs": mx, "max_abs_kfac_weights": mx_kf,
                                        "max_rel": mx_rel, "eps_dtype_times_max_abs_W": ulp})
            logger.info(f"[kfac] restore check: {self.restore_checks[-1]}")
        return loss1

    def zo_update(self, model):
        args = self.args
        lr = self._get_learning_rate()
        if self._E is not None:
            Es = self._E
        else:
            Es = self._draw()
        for (name, param), z in zip(self.named_parameters_to_optim, Es):
            if z is None:
                continue
            if args.kfac_placement == "B" and not self.kfac_identity and name in self.kfac:
                z = self._transform(name, z)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - lr * (self.projected_grad * z + args.weight_decay * param.data)
            else:
                param.data = param.data - lr * (self.projected_grad * z)
        self._E = None
        self.lr_scheduler.step()

    # ------------------------------------------------------------------ reporting
    def extra_results(self):
        a = self.args
        return {"kfac_scope": a.kfac_scope, "kfac_placement": a.kfac_placement, "kfac_power": a.kfac_power, "kfac_damping": a.kfac_damping,
                "kfac_fisher": a.kfac_fisher, "kfac_block": a.kfac_block, "kfac_refresh": a.kfac_refresh,
                "kfac_backprop_examples": self.kfac_backprop_examples,
                "kfac_backprop_passes": self.kfac_backprop_passes,
                "kfac_calibrations": self.kfac_calibs,
                "kfac_calib_seconds_total": float(sum(c["seconds"] for c in self.kfac_calibs)),
                "restore_checks": self.restore_checks,
                "pg_decomp_rms": {k: float(np.sqrt(np.mean([r[k] ** 2 for r in self.pg_decomp]))) for k in ("kfac", "rest", "kfac_iso")} if self.pg_decomp else None,
                "pg_decomp_n": len(self.pg_decomp),
                "pg_full_rms": float(np.sqrt(np.mean(np.square(self.pg_abs)))) if self.pg_abs else None,
                "projected_grad_abs_mean": float(np.mean(self.pg_abs)) if self.pg_abs else None,
                "projected_grad_abs_median": float(np.median(self.pg_abs)) if self.pg_abs else None}
