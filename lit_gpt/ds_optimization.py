import torch
import torch.nn as nn
import lightning as L

from lit_gpt.utils import chunked_cross_entropy


class AlphaModel(nn.Module):
    """Prototype for the DS Optim alpha model."""

    def __init__(
        self,
        alpha: float = None,
        cfg=None,  # this is a CLISetting dataclass, see train.py
        fabric: L.Fabric = None,  # this is our distributed env management object
    ):
        super().__init__()

        assert alpha is not None, "AlphaModel requires an initial alpha value."
        self.initial_alpha = alpha

        # self.alpha = nn.Parameter(torch.tensor(alpha, device=fabric.device), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(alpha, device="cpu"), requires_grad=True)
        self.cfg = cfg

        # explicitly noting state we manage
        self.grad_bucket = None
        self.model_optimizer_state_bucket = None
        self.model_at_t_state_bucket = None

    def init_grad_bucket(self, model: nn.Module):
        self.grad_bucket = {
            name: torch.zeros_like(param).cpu()  # this should be equiv, shape wise to param.grad
            # name: torch.zeros_like(param.grad)
            # name: param.grad.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad  # can't use .grad here, it's None at initialization
        }

    def store_model_grads(self, model: nn.Module):
        # copy the grads from the model into the grad_bucket
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.grad_bucket[name].copy_(param.grad)

    def accumulate_model_grads(self, model: nn.Module):
        # accumulate the grads from the model into the grad_bucket
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.grad_bucket[name].add_(param.grad.to(self.grad_bucket[name].device))

    def set_model_grads(self, model: nn.Module):
        # copy the grads from the bucket into the model
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad.copy_(self.grad_bucket[name])

    def attach_model_at_t(self, model_at_t: nn.Module):
        self.model_at_t_state_bucket = model_at_t

    def store_model_at_t(self, model_at_t: nn.Module):
        # self.model_at_t_state_bucket = copy.deepcopy(model_at_t.state_dict())
        self.model_at_t_state_bucket.load_state_dict(model_at_t.state_dict())

    def restore_model_at_t(self, model: nn.Module):
        model.load_state_dict(self.model_at_t_state_bucket.state_dict())

    def attach_model_at_t_star(self, model_at_t_star: nn.Module):
        self.model_at_t_star_state_bucket = model_at_t_star

    def store_model_at_t_star(self, model_at_t_star: nn.Module):
        self.model_at_t_star_state_bucket.load_state_dict(model_at_t_star.state_dict())

    def restore_model_at_t_star(self, model: nn.Module):
        model.load_state_dict(self.model_at_t_star_state_bucket.state_dict())

    def attach_model_optimizer_bucket(self, model_optimizer_bucket: torch.optim.Optimizer):
        # attach the optimizer_bucket to the model_optimizer
        self.model_optimizer_state_bucket = model_optimizer_bucket

    def store_model_optimizer_state(
        self,
        model_optimizer: torch.optim.Optimizer,
    ):

        # copy the current optimizer state into the optimizer_state_bucket
        # make a copy of the model_optimizer
        # self.model_optimizer_state_bucket = copy.deepcopy(model_optimizer.state_dict())
        # self.model_optimizer_state_bucket = copy.deepcopy(
        #     {k: v.cpu() for k, v in model_optimizer.state_dict().items()}
        # )
        # We might not be able to keep this in mem, rather write to disk and read back in.
        self.model_optimizer_state_bucket.load_state_dict(model_optimizer.state_dict())

    def restore_model_optim_state(self, model_optimizer: torch.optim.Optimizer):
        # copy the optimizer state from the optimizer_state_bucket back into the optimizer
        # model_optimizer.load_state_dict(self.model_optimizer_state_bucket)
        model_optimizer.load_state_dict(self.model_optimizer_state_bucket.state_dict())

    def prep_for_theta_stage(
        self,
        model: torch.nn.Module,
        model_at_t: torch.nn.Module,
    ):
        self.store_model_at_t(model_at_t)
        self.init_grad_bucket(model)

    def prep_for_alpha_stage(
        self,
        model_optimizer: torch.optim.Optimizer,
        model_at_t_star: torch.nn.Module,
        model_at_t: torch.nn.Module = None,
    ):
        # FIXME
        self.store_model_optimizer_state(model_optimizer)
        self.store_model_at_t_star(model_at_t_star)

        if model_at_t is not None:
            assert self.cfg.theta_tn_setting == "tstar"
            # The model we will start each alpha step from will be the model at the end of the theta stage
            self.store_model_at_t(model_at_t)
            # Additionally, we need to shift the alphas to (alpha - initial_alpha) to start the alpha stage
            self.alpha.data = self.alpha.data - self.initial_alpha

    def finalize_theta_alpha_stage(
        self,
        model: torch.nn.Module,
        model_optimizer: torch.optim.Optimizer,
        fabric: L.Fabric,
    ):
        if self.cfg.finalize_at_t_star:
            self.restore_model_at_t_star(model)
        else:
            self.forward(
                model=model,
                model_optimizer=model_optimizer,
                fabric=fabric,
                finalize=True,
            )
        if self.cfg.theta_tn_setting == "tstar":
            # we undo the shift of the alphas to return them to their full values
            self.alpha.data = self.alpha.data + self.initial_alpha
        if self.cfg.restore_model_optim_after_alpha_stage:
            self.restore_model_optim_state(model_optimizer)

    def forward(
        self,
        model: nn.Module,  # this is a fabric managed model, which will have _unreduced_ prev gradients at train time
        model_optimizer: torch.optim.Optimizer,  # this is a fabric managed distributed optimizer
        fabric: L.Fabric,
        alpha_optimizer: torch.optim.Optimizer = None,  # this is a sharded optimizer, one per rank
        input_ids: torch.Tensor = None,
        targets: torch.Tensor = None,
        finalize: bool = False,
    ):
        # restore the model to the state at t
        self.restore_model_at_t(model)
        # fabric.print(f"passed restore_model_at_t")

        # reset the grads in the model to the grads in the grad_bucket
        self.set_model_grads(model)
        # fabric.print(f"passed set_model_grads")

        # project params as a function of current alphas with a weighted allreduce and step
        coef = float(self.alpha.data)
        fabric.all_reduce(
            {name: param.grad for name, param in model.named_parameters() if param.grad is not None},
            reduce_op=torch.distributed._make_nccl_premul_sum(coef),
        )

        # fabric.print(f"passed all_reduce")

        model_optimizer.step()
        model_optimizer.zero_grad()
        # fabric.print(f"passed step and zero_grad")

        if finalize:
            fabric.print(f"Finalizing alpha stage by applying grad buckets mixed w/ new alphas to model_t ...")
            return

        # FIXME
        self.restore_model_optim_state(model_optimizer)
        # fabric.print(f"passed restore_model_optim_state")

        # Now we compute and _ungaurded_ fwd loss bwd pass to get synced grads as fn of val batch
        with fabric.no_backward_sync(model, enabled=False):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets)
            fabric.backward(loss)  # no accumulation in alpha training for now

        # fabric.print(f"passed alpha backward")

        # Now we do a grad_val * grad_bucket product to get this rank's alpha grad and assign it
        # to the alpha.grad attribute
        # FIXME... was the overleaf missing a negative sign?
        self.alpha.grad = torch.neg(
            torch.dot(
                # torch.cat(
                #     [param.grad.flatten() for param in model.parameters() if param.requires_grad]
                # ),
                # torch.cat([grad.flatten().to(fabric.device) for grad in self.grad_bucket.values()]),
                torch.cat([param.grad.flatten().cpu() for param in model.parameters() if param.requires_grad]),
                torch.cat([grad.flatten() for grad in self.grad_bucket.values()]),
            )
        )

        # fabric.print(f"passed grad grad product")

        # Now we step the alpha_optimizer, updating the rank's alpha value
        alpha_optimizer.step()
        alpha_optimizer.zero_grad()

        # fabric.print(f"passed alpha step and zero_grad")
