# Here is quantizing add-on to the model 

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from lora import _parse_target_layers
from model import GPT 


@dataclass
class QuantConfig:
    enable: bool = False
    mode: str = "none"          # none | int8 | nf4 | fp4
    targets: str = "linear"     # linear | all-linear
    compute_dtype: str = "bfloat16"
    double_quant: bool = True
    quant_storage: str = "uint8"
    backend: str = "bitsandbytes"
    
    def is_compatible(self, o: 'QuantConfig', n_layer: int) -> bool:
        if not o: return False
        if not self.enable and not o.enable: return True
        return all([
            self.mode == o.mode,
            self.targets == o.targets,
            self.compute_dtype == o.compute_dtype,
            self.double_quant == o.double_quant,
        ])

def quant_config_to_dict(cfg: QuantConfig | None) -> dict:
    if not cfg: return dict()
    return asdict(cfg)

def dict_to_quant_config(d: dict | None) -> QuantConfig:
    if not d: return QuantConfig()
    cfg = QuantConfig()
    for attr, val in d.items():
        setattr(cfg, attr, val)
    return cfg


def _import_bitsandbytes():
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "Quantization backend 'bitsandbytes' is requested, "
            "but bitsandbytes is not installed. Try: pip install bitsandbytes"
        ) from e

    return bnb

def _torch_dtype(dtype: str):
    dtype = dtype.strip().lower()
    if dtype in ("float16", "fp16", "half"):
        return torch.float16
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported quant compute dtype: {dtype}")

def quantize_linear(src: nn.Linear, config: QuantConfig) -> nn.Module:
    bnb = _import_bitsandbytes()
    
    if config.mode not in ("nf4", "fp4"):
        raise ValueError(f"Unsupported 4-bit quantization mode: {config.mode}")

    compute_dtype = _torch_dtype(config.compute_dtype)

    if config.quant_storage.strip().lower() != "uint8":
        raise ValueError(f"Unsupported quant storage dtype: {config.quant_storage}")
    quant_storage = torch.uint8
    
    dst = bnb.nn.Linear4bit(
        src.in_features,
        src.out_features,
        bias=(src.bias is not None),
        compute_dtype=compute_dtype,
        compress_statistics=config.double_quant,
        quant_type=config.mode,
        quant_storage=quant_storage,
        device=src.weight.device,
    )
    
    state = {
        "weight": src.weight.detach().clone().to(dtype=compute_dtype),
    }

    if src.bias is not None:
        state["bias"] = src.bias.detach().clone().to(dtype=compute_dtype)

    dst.load_state_dict(state, strict=True)
    dst.requires_grad_(False)
    dst.train(src.training)

    return dst


def freeze_base_model(model: GPT) -> int:
    cnt = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False
            cnt += p.numel()
    return cnt


TARGETS = {
    "attn.c_attn": ("attn", "c_attn"),
    "attn.c_proj": ("attn", "c_proj"),
    "mlp.c_fc":    ("mlp", "c_fc"),
    "mlp.c_proj":  ("mlp", "c_proj"),
}

# apply to all model, if need to apply to certan layers use 'target_layers'
def apply_quantizing(model: GPT, config: QuantConfig, target_layers: str|None=None) -> bool:
    if not config.enable:
        return False

    if config.backend != "bitsandbytes":
        raise ValueError(f"Unsupported quantization backend: {config.backend}")
    bnb = _import_bitsandbytes()

    if target_layers is None:
        target_layers = "all"
    layers = _parse_target_layers(target_layers, model.config.n_layer)

    if config.targets.strip() == "all":
        targets = list(TARGETS.keys())
    else:
        targets = [x.strip() for x in targets.split(",") if x.strip()]
    
    applied_cnt = 0

    for i in layers:
        for target in targets:
            if target not in TARGETS:
                raise ValueError(f"Unsupported target to apply quantization: {target}")
            
            subpath, attr = TARGETS[target]
            parent = model.transformer.h[i].get_submodule(subpath)
            
            obj = getattr(parent, attr)
            
            if isinstance(obj, (bnb.nn.Linear4bit, bnb.nn.LinearFP4, bnb.nn.LinearNF4)):
                print(f"Quantization is already applied to {target} on layer {i}")
                print("skipping for now")
                continue
            
            if not isinstance(obj, nn.Linear):
                raise ValueError(
                    f"{target} on layer {i} is not nn.Linear, got {type(obj)}. "
                    "Hint: apply quantization before applying LoRA."
                )
            
            obj = quantize_linear(obj, config)
            setattr(parent, attr, obj)
            
            applied_cnt += 1
    model.quant_config = config
    return applied_cnt > 0
            