import os
import torch as th
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, get_peft_model
from .model import *
from .constants import *
from utils import logger 
from peft import LoraConfig, TaskType
from utils.core.path import _get_adapter_dir
from utils import p_args


def load_projector_weights(
    model, 
    weight_path
):
    # validate weight file exists
    if not os.path.exists(weight_path):
        logger.warning(f"⚠️ Projector weights not found at {weight_path}, using random initialization")
        return False
    
    try:
        weights = th.load(weight_path, map_location='cpu')
        
        # Handle Format 2: Direct projector state dict (standard format)
        if isinstance(weights, dict):
            projector_weights = weights.get('projector_state_dict', weights)
        else:
            logger.error(f"❌ Unexpected weight format in {weight_path}")
            return False
        
        # normalize keys (strip "embedding_projector." prefix if present)
        def _strip_prefix(d, prefix="embedding_projector."):
            return { (k[len(prefix):] if k.startswith(prefix) else k) : v for k,v in d.items() }
        projector_weights = _strip_prefix(projector_weights)

        model.model.embedding_projector.load_state_dict(projector_weights, strict=False)

        # Use bfloat16 to match saved parameters
        model.model.embedding_projector.to(dtype=torch.bfloat16)
        
        # Verify the weights were loaded properly
        param_norm = sum(p.norm().item() for p in model.model.embedding_projector.parameters())
        if param_norm < 1e-6:
            logger.warning("⚠️ Projector parameters appear to be uninitialized after loading")
            return False
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load projector weights from {weight_path}: {e}")
        return False


def configure_kwargs(
    device, 
    load_8bit, 
    load_4bit, 
    use_flash_attn,
    device_map="auto",
    tp_plan=None,
):
    """Configure model loading kwargs"""

    kwargs = {}

    # tp_plan shards every weight while it is read off disk, so each rank only
    # ever materializes its own slice. It is mutually exclusive with device_map.
    if tp_plan is not None:
        pass
    elif device_map is not None:
        if device_map == "auto":
            if device == "cuda":
                kwargs["device_map"] = "auto"
            elif device != "cpu":
                kwargs["device_map"] = {"": device}
        else:
            kwargs["device_map"] = device_map
    elif device != "cuda":
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = "auto"

    # Quantization settings
    if load_8bit and load_4bit:
        raise ValueError("Choose only one of load_8bit or load_4bit.")

    # Check for CUDA requirement for quantization
    has_device_map = "device_map" in kwargs
    is_cpu_device = (not has_device_map) or kwargs.get("device_map") == "cpu" or device == "cpu"

    if (load_8bit or load_4bit) and is_cpu_device:
        raise ValueError("bitsandbytes 4/8-bit requires CUDA devices.")

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    return kwargs


def _check_projector_dims(dec_config, model) -> None:
    emb_cfg = AutoConfig.from_pretrained(p_args.model.emb_id)
    emb_hidden = getattr(emb_cfg, "hidden_size", None)
    dec_hidden = getattr(dec_config, "hidden_size", None)

    if emb_hidden is None or dec_hidden is None:
        logger.warning("⚠️ Unable to read hidden_size for emb_id or dec_id; skipping dim checks")
        return

    projector = getattr(getattr(model, "model", None), "embedding_projector", None)
    if projector is None:
        logger.warning("⚠️ embedding_projector not found on model; skipping dim checks")
        return

    if int(projector.dim_in) != int(emb_hidden):
        raise ValueError(
            "Projector input dim mismatch: "
            f"expected emb_id hidden_size {emb_hidden}, got {projector.dim_in}."
        )

    if int(projector.dim_out) != int(dec_hidden):
        raise ValueError(
            "Projector output dim mismatch: "
            f"expected dec_id hidden_size {dec_hidden}, got {projector.dim_out}."
        )


def load_einv_from_checkpoint(
    tokenizer,
    checkpoint_dir=None,
    device="cuda",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
):
    """Load a trained EInv model from an Accelerate save_state checkpoint dir.

    Use this when the adapter dir has only `checkpoints/model.safetensors`
    (full state from `accelerator.save_state`) and no separate `projector.bin`
    — i.e. a training that didn't run `save_final()`. `model.safetensors`
    already contains the trained projector + base weights.

    Returns the model on `device`, ready for inference (`eval()`).
    """
    from transformers import AddedToken

    # Pre-add EM special token so len(tokenizer) matches the saved vocab size
    if DEFAULT_EM_TOKEN not in set(tokenizer.additional_special_tokens):
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                AddedToken(DEFAULT_EM_TOKEN, special=True, normalized=False)
            ]
        })
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(_get_adapter_dir(), "checkpoints")

    config = EInvConfig.from_pretrained(p_args.model.dec_id)
    config.vocab_size = len(tokenizer)  # match resized vocab from training

    model = EInvLlamaForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    _check_projector_dims(config, model)
    model.initialize_einv_tokenizer(tokenizer)  # num_new_tokens=0, no resize
    model = model.to(device)
    model.eval()
    return model


def load_pretrained_model(
    load_8bit=False,
    load_4bit=False,
    device_map=None,
    device="cpu",
    use_flash_attn=False,
    model_type=None,   # "base", "einv", "einv_lora"
    resume_from=False,
    train_config=None,
    proj_dir=None,
    **extra_kwargs,
):
    """
    Load pretrained model.
    """
    mt = (model_type or "").lower()
    is_base = mt in ("base", "")
    is_einv = mt == "einv"
    is_einv_lora = mt == "einv_lora"

    # Configure kwargs
    kwargs = configure_kwargs(
        device=device,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        use_flash_attn=use_flash_attn,
        device_map=device_map,
        tp_plan=extra_kwargs.get("tp_plan"),
    )
    kwargs.update(extra_kwargs)

    # Load tokenizer, use model id directly
    tokenizer = AutoTokenizer.from_pretrained(p_args.model.dec_id, use_fast=True)
    context_len = getattr(tokenizer, "model_max_length", 2048)
    if proj_dir is None:
        proj_dir = _get_adapter_dir()

    # ===== 1) BASE =====
    if is_base:
        model = AutoModelForCausalLM.from_pretrained(p_args.model.dec_id, low_cpu_mem_usage=True, **kwargs)
        return tokenizer, model, context_len

    # ===== 2) EINV =====
    if is_einv:
        config = EInvConfig.from_pretrained(p_args.model.dec_id)
        model = EInvLlamaForCausalLM.from_pretrained(p_args.model.dec_id, config=config, low_cpu_mem_usage=True, **kwargs)
        _check_projector_dims(config, model)
        model.initialize_einv_tokenizer(tokenizer)

        if resume_from:
            logger.info("Loading pretrained embedding projector weights...")
            load_projector_weights(model, os.path.join(proj_dir, "projector.bin"))

        return model.tokenizer, model, context_len

    # ===== 3) EINV + LORA =====
    if is_einv_lora:
        config = EInvConfig.from_pretrained(p_args.model.dec_id)
        model = EInvLlamaForCausalLM.from_pretrained(p_args.model.dec_id, config=config, low_cpu_mem_usage=True, **kwargs)
        _check_projector_dims(config, model)
        model.initialize_einv_tokenizer(tokenizer)

        if resume_from:
            logger.info("Loading finetuned embedding projector and LoRA weights...")
            load_projector_weights(model, os.path.join(proj_dir, "lora_projector.bin"))
            model = PeftModel.from_pretrained(model, proj_dir)
        else:
            logger.info("Loading pretrained embedding projector weights for finetuning...")
            load_projector_weights(model, os.path.join(proj_dir, "projector.bin"))
            lora_config, layers = get_lora_config(train_config)
            model = get_peft_model(model, lora_config)

        return model.tokenizer, model, context_len

    raise ValueError(f"Unknown model_type '{model_type}'. Use 'base', 'einv', or 'einv_lora'.")

def get_lora_config(train_config):
    # Choose layers exactly as listed in YAML
    layers = sorted(set(int(i) for i in train_config["lora"]["layers"]))
    
    # Build explicit module name lists
    target_modules = []
    
    # Attention targets
    for layer in layers:
        for att_target in train_config["lora"]["target_atts"]:
            target_modules.append(f"model.layers.{layer}.self_attn.{att_target}")
    
    # MLP targets (optional)
    if train_config["lora"].get("target_mlps"):
        for layer in layers:
            for mlp_target in train_config["lora"]["target_mlps"]:
                target_modules.append(f"model.layers.{layer}.mlp.{mlp_target}")
    
    # Use single rank/alpha for all modules
    r = int(train_config["lora"]["att_r"])  # or choose appropriate default
    alpha = int(train_config["lora"].get("att_alpha", r*2))
    
    lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=float(train_config["lora"]["dropout"]),
        bias=train_config["lora"].get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        modules_to_save=train_config["lora"].get("modules_to_save", []),
    )
    
    return lora, layers


