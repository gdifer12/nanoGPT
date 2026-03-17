# Here is LoRA add-on to the model 

from dataclasses import dataclass

import torch.nn as nn

from model import GPT 

@dataclass
class LoRAConfig:
    targets: str = "" # from {attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj} divided by commas
    target_layers: str = "all" # "all" or list of python-like slices through commas (example: "1,3:5,6:10:2,12" -> [1, 3, 4, 6, 8, 12])
    rank: int = 8
    alpha: float = 1.0 # scale is alpha/rank 
    bias: bool = False # if True do not freese bias on target (if exists and not freezed)

class LoRALinear(nn.Module):
    def __init__(self, src: nn.Linear, config: LoRAConfig):
        super().__init__()
        
        rank = config.rank
        assert rank > 0
        self.scale = config.alpha / rank
        
        unfreeze_bias = (config.bias and src.bias is not None and src.bias.requires_grad)
        
        self.w0 = src
        self.w0.requires_grad_(False)
        if unfreeze_bias:
            self.w0.bias.requires_grad_(True)

        self.A = nn.Linear(src.in_features, rank, bias=False)
        self.B = nn.Linear(rank, src.out_features, bias=False)

        nn.init.zeros_(self.B.weight)
    
    @property
    def bias(self):
        return self.w0.bias
    
    @property
    def weight(self):
        return self.w0.weight + self.scale * (self.B.weight @ self.A.weight)

    def forward(self, x):
        return self.w0(x) + self.scale * self.B(self.A(x))


def _parse_target_layers(target_layers: str, n_layer: int) -> list[int]:
    spec = target_layers.strip()
    all_layers = range(n_layer)
    if spec == "all":
        return list(all_layers)
    
    if spec == "":
        return []
    
    layers: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if token == "":
            print(f"Warning: empty token in layers indices ({target_layers})")
            continue
        if ":" not in token:
            layers.add(int(token))
            continue
        
        parts = token.split(":")
        if len(parts) > 3:
            raise ValueError(f"Cannot parse: {token}")
        if len(parts) == 2:
            parts.append("")
        parts = [int(x) if x.strip() != "" else None for x in parts]
        
        layers.update(all_layers[slice(*parts)])
        
    layers = sorted(list(layers))
    if len(layers) and layers[-1] >= n_layer:
        raise ValueError(f"Try to apply LoRA to layer that not exists: {layers[-1]}, n_layer: {n_layer}")
    return layers
    
TARGETS = {
    "attn.c_attn": ("attn", "c_attn"),
    "attn.c_proj": ("attn", "c_proj"),
    "mlp.c_fc":    ("mlp", "c_fc"),
    "mlp.c_proj":  ("mlp", "c_proj"),
}

def apply_LoRA(model: GPT, config: LoRAConfig):
    layers = _parse_target_layers(config.target_layers, model.config.n_layer)
    targets = [x.strip() for x in config.targets.split(",") if x.strip()]
    for i in layers:
        for target in targets:
            if target not in TARGETS:
                raise ValueError(f"Unsupported target to apply LoRA: {target}")
            
            subpath, attr = TARGETS[target]
            parent = model.transformer.h[i].get_submodule(subpath)
            
            obj = getattr(parent, attr)
            if isinstance(obj, LoRALinear):
                print(f"LoRA is already applied to {target} on layer {i}")
                continue
            if not isinstance(obj, nn.Linear):
                raise ValueError(f"{target} on layer {i} in not a suitable object")
            
            setattr(parent, attr, LoRALinear(obj, config))