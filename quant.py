# Here is quantizing add-on to the model 

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import _parse_target_layers
from model import GPT 


@dataclass
class QuantConfig:
    enable: bool = False
    mode: str = "int8" # fake: int8 | int4; bitsandbytes: nf4 | fp4
    targets: str = "all-linear" # 'all' (= 'all-linear')  or some from attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
    target_layers: str = "all"
    backend: str = "fake" # 'fake', 'bitsandbytes' does not really work

    # bitsanbytes
    compute_dtype: str = "bfloat16"
    double_quant: bool = True
    quant_storage: str = "uint8"

    # fake    
    fake_weight_per_channel: bool = True
    fake_act_bits: int = 0  # 0 = disable activation fake quant
    
    def is_compatible(self, o: 'QuantConfig', n_layer: int) -> bool:
        if not o: return not self.enable
        if not self.enable and not o.enable: return True
        return all([
            self.enable == o.enable,
            self.mode == o.mode,
            self.targets == o.targets,
            _parse_target_layers(self.target_layers, n_layer) == _parse_target_layers(o.target_layers, n_layer),
            self.backend == o.backend,
            self.backend != 'bitsandbytes' or all([
                self.compute_dtype == o.compute_dtype,
                self.double_quant == o.double_quant,
                self.quant_storage == o.quant_storage,
            ]),
            self.backend != 'fake' or all([
                self.fake_weight_per_channel == o.fake_weight_per_channel,
                self.fake_act_bits == o.fake_act_bits
            ])
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

def _torch_dtype(dtype: str):
    dtype = dtype.strip().lower()
    if dtype in ("float16", "fp16", "half"):
        return torch.float16
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported quant compute dtype: {dtype}")

def _bits_from_mode(mode: str) -> int:
    mode = mode.strip().lower()
    if mode in ('int8', 'int4'):
        return int(mode[-1])
    raise ValueError(
        f"Fake quant supports only mode='int8' or mode='int4', got {mode!r}"
    )
    
def fake_quant_symmetric(
        x: torch.Tensor,
        bits: int,
        per_channel: bool = False,
        ch_axis: int = 0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
    if bits <= 0:
        return x

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax

    if per_channel:
        dims = tuple(i for i in range(x.ndim) if i != ch_axis)
        scale = x.detach().abs().amax(dim=dims, keepdim=True).clamp(min=eps) / qmax
    else:
        scale = x.detach().abs().amax().clamp(min=eps) / qmax

    q = x / scale
    q = (q.round() - q).detach() + q
    q = torch.clamp(q, qmin, qmax)
    return q * scale

class FakeQuantLinear(nn.Module):
    def __init__(self, src: nn.Linear, config: QuantConfig):
        super().__init__()

        if not isinstance(src, nn.Linear):
            raise TypeError(f"FakeQuantLinear expects nn.Linear, got {type(src)}")

        self.base = src
        self.in_features = src.in_features
        self.out_features = src.out_features

        self.weight_bits = _bits_from_mode(config.mode)
        self.act_bits = config.fake_act_bits
        self.weight_per_channel = config.fake_weight_per_channel

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_bits > 0:
            x = fake_quant_symmetric(
                x,
                bits=self.act_bits,
                per_channel=False,
            )

        w = fake_quant_symmetric(
            self.base.weight,
            bits=self.weight_bits,
            per_channel=self.weight_per_channel,
            ch_axis=0,
        )

        return F.linear(x, w, self.base.bias)


def _import_bitsandbytes():
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "Quantization backend 'bitsandbytes' is requested, "
            "but bitsandbytes is not installed. Try: pip install bitsandbytes"
        ) from e

    return bnb

def is_module_supported(module: nn.Module, config: QuantConfig) -> bool:
    if config.backend == 'bitsandbytes':
        return isinstance(module, nn.Linear)
    if config.backend == 'fake':
        return isinstance(module, nn.Linear)
    return False
    
def is_module_quantized(module: nn.Module, config: QuantConfig) -> bool:
    if config.backend == 'bitsandbytes':
        bnb = _import_bitsandbytes()
        return isinstance(module, (bnb.nn.Linear4bit, bnb.nn.LinearFP4, bnb.nn.LinearNF4))
    if config.backend == 'fake':
        return isinstance(module, FakeQuantLinear)
    return False

def quantize_linear(src: nn.Linear, config: QuantConfig) -> nn.Module:
    if config.backend == 'fake':
        return FakeQuantLinear(src, config)
    
    if config.backend != "bitsandbytes":
        raise ValueError(f"Unsupported quantization backend: {config.backend}")
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


def freeze_base_model(model: GPT, only_linear: bool = True) -> int:
    cnt = 0
    params = [p for l in model.transformer.h for p in l.parameters()] if only_linear else model.parameters()
    for p in params:
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
def apply_quantizing(model: GPT, config: QuantConfig) -> bool:
    if not config.enable:
        return False

    if config.backend not in ("fake", "bitsandbytes"):
        raise ValueError(f"Unsupported quantization backend: {config.backend}")

    layers = _parse_target_layers(config.target_layers, model.config.n_layer)

    if config.targets.strip() in ('all', 'all-linear'):
        targets = list(TARGETS.keys())
    else:
        targets = [x.strip() for x in config.targets.split(",") if x.strip()]
    
    applied_cnt = 0

    for i in layers:
        for target in targets:
            if target not in TARGETS:
                raise ValueError(f"Unsupported target to apply quantization: {target}")
            
            subpath, attr = TARGETS[target]
            parent = model.transformer.h[i].get_submodule(subpath)
            
            obj = getattr(parent, attr)
            
            if is_module_quantized(obj, config):
                print(f"Quantization is already applied to {target} on layer {i}")
                print("skipping for now")
                continue
            
            if not is_module_supported(obj, config):
                raise ValueError(
                    f"{target} on layer {i} is not supported, got {type(obj)}. "
                    "Hint: apply quantization before applying LoRA."
                )
            
            obj = quantize_linear(obj, config)
            setattr(parent, attr, obj)
            applied_cnt += 1
    model.quant_config = config
    return applied_cnt > 0
            