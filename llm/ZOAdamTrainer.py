"""
[inverse-metric] ZO-Adam on the MeZO harness.

Port of ZO-LLM/zo-bench trainer.py (trainer=zo_adam): optimizer at L315-316, update at L1045-1075 (MeZO two-point
estimate, then param.grad = projected_grad * z and one Adam step per parameter tensor).
Why a port: zo-bench pins transformers 4.28.1, which has no Qwen2 model class; see results/p3a_plan.md for the exact error.
Same defaults as zo-bench: torch.optim.Adam(lr=args.learning_rate) with betas (0.9, 0.999), eps 1e-8, no weight decay.
Extra memory: 2x parameters (Adam m, v).
"""
import torch
from MeZOTrainer import MeZOTrainer


class ZOAdamTrainer(MeZOTrainer):

    def zo_update(self, model):
        args = self.args
        if not hasattr(self, "zo_adam"):
            assert args.lr_scheduler_type == "constant", "zo-bench did not implement an lr schedule for ZO-Adam"
            self.zo_adam = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.grad = self.projected_grad * z
            self.zo_adam.step()  # updates only the params with a grad, i.e. this one
            param.grad = None
        self.lr_scheduler.step()  # no-op on the update: bound to the HF optimizer, as in zo-bench

    def extra_results(self):
        return {"forwards_per_step": 2}
