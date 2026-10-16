import os
import torch
from .logger import logger
from .gpu_distribute import gh
from .path import _get_adapter_dir


def _get_projector(model):
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model, "peft_config"):
        return actual_model.base_model.model.model.embedding_projector
    else:
        return actual_model.model.embedding_projector


class BestTracker:
    """In-memory snapshot of best-val-loss weights (projector + optional LoRA).

    Call update() each time val_loss improves during training. At training end,
    pass this tracker to save_final so the on-disk weights reflect the best
    validation step rather than the last training step.
    """

    def __init__(self):
        self.projector_state: dict | None = None
        self.lora_state: dict | None = None
        self.val_loss: float = float("inf")

    def update(self, unwrapped_model, val_loss: float, save_lora: bool = False) -> None:
        """Snapshot weights if val_loss is a new minimum. Only rank 0 acts."""
        if not gh.is_main or val_loss >= self.val_loss:
            return
        self.val_loss = val_loss
        projector = _get_projector(unwrapped_model)
        self.projector_state = {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()}
        if save_lora:
            from peft.utils import get_peft_model_state_dict
            peft_sd = get_peft_model_state_dict(unwrapped_model)
            self.lora_state = {k: v.detach().cpu().clone() for k, v in peft_sd.items()}
        logger.info(f"⭐ New best val_loss={val_loss:.4f} — in-memory snapshot updated")

    def restore(self, unwrapped_model, save_lora: bool = False) -> bool:
        """Load best snapshot back into model. Returns True if a snapshot existed."""
        if not gh.is_main or self.projector_state is None:
            return False
        projector = _get_projector(unwrapped_model)
        projector.load_state_dict(self.projector_state)
        if save_lora and self.lora_state is not None:
            from peft.utils import set_peft_model_state_dict
            set_peft_model_state_dict(unwrapped_model, self.lora_state)
        logger.info(f"⭐ Restored best weights (val_loss={self.val_loss:.4f}) for final save")
        return True


def save_final(
    unwrapped_model,
    save_lora: bool = False,
    best_tracker: "BestTracker | None" = None,
):
    """Save final LoRA adapters and/or projector weights. Only rank 0 performs I/O.

    If best_tracker is provided and has a snapshot, the best weights are restored
    into the model before saving, so the on-disk final equals the best checkpoint.
    """
    _dir = _get_adapter_dir()
    os.makedirs(_dir, exist_ok=True)

    # Restore best weights before writing so final == best
    if best_tracker is not None and gh.is_main:
        restored = best_tracker.restore(unwrapped_model, save_lora=save_lora)
        if not restored:
            logger.info("ℹ️ No best snapshot found; saving current (final-step) weights")

    # Save LoRA adapters (PEFT models)
    if save_lora and gh.is_main:
        if unwrapped_model is None:
            raise ValueError("unwrapped_model is required when save_lora=True")
        unwrapped_model_cpu = unwrapped_model.to("cpu")
        unwrapped_model_cpu.save_pretrained(str(_dir))
        logger.info(f"💾 Saved LoRA adapters to {_dir}")

    # Save projector weights
    save_path = os.path.join(_dir, "projector.bin")
    if save_lora:
        save_path = os.path.join(_dir, "lora_projector.bin")

    if gh.is_main:
        projector = _get_projector(unwrapped_model)
        torch.save(projector.state_dict(), save_path)

    gh.barrier()

    if gh.is_main:
        logger.info("✅ Final save complete.")