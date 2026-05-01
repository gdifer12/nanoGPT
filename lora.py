# Here is LoRA add-on to the model 

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT 

@dataclass
class LoRAConfig:
    enable: bool = False
    targets: str = "" # "all", "all-linear" (all \ {wte, wpe}) or subset of {attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj, wte, wpe} divided by commas
    target_layers: str = "all" # "all" or list of python-like slices through commas (example: "1,3:5,6:10:2,12" -> [1, 3, 4, 6, 8, 12])
    rank: int = 8
    alpha: float = 1.0 # scale is alpha/rank 
    bias: bool = False # if True do not freese bias on target (if exists and not freezed)
    merge_weights: bool = True # merge weights when eval mode to fasten inference

    # True if optimizer_state can be reused if apply o after self
    def is_compatible(self, o: 'LoRAConfig', n_layer: int) -> bool:
        if not o: return not self.enable
        if not self.enable and not o.enable: return True
        return all([
            self.enable == o.enable,
            self.targets == o.targets,
            _parse_target_layers(self.target_layers, n_layer) == _parse_target_layers(o.target_layers, n_layer),
            self.rank == o.rank,
            self.bias == o.bias,
        ])
        
def lora_config_to_dict(cfg: LoRAConfig | None) -> dict:
    if not cfg: return dict()
    return asdict(cfg)

def dict_to_lora_config(d: dict | None) -> LoRAConfig:
    if not d: return LoRAConfig()
    cfg = LoRAConfig()
    for attr, val in d.items():
        setattr(cfg, attr, val)
    return cfg


def _linear_in_features(src) -> int:
    for name in ("in_features", "input_features"):
        if hasattr(src, name):
            return int(getattr(src, name))
    raise TypeError(f"{type(src)} has no in/input features attribute")
    
def _linear_out_features(src) -> int:
    for name in ("out_features", "output_features"):
        if hasattr(src, name):
            return int(getattr(src, name))
    raise TypeError(f"{type(src)} has no out/output features attribute")

def _linear_bias(src):
    return getattr(src, "bias", None)

# suppose to not use src enywhere else 
def _to_linear_like(src):
    if isinstance(src, LoRALinear):
        if src.can_merge:
            if not src.merged:
                with torch.no_grad():
                    src.base.weight.add_(src.lora_delta_weight())
                src.merged = True
            return src.base
        else:
            raise TypeError(f"Can't decide what to do with unmergeble different adapter")

    if not hasattr(src, "forward"):
        raise TypeError(f"{type(src)} is not callable as linear-like")
    _linear_in_features(src)
    _linear_out_features(src)
    return src

def _has_mergeable_dense_weight(src) -> bool:
    w = getattr(src, "weight", None)
    return (
        isinstance(w, torch.Tensor)
        and w.ndim == 2
        and w.is_floating_point()
    )


class LoRALinear(nn.Module):
    def __init__(self, src, config: LoRAConfig):
        super().__init__()
        
        rank = config.rank
        assert rank > 0
        self.scale = config.alpha / rank
        
        bias = _linear_bias(src)
        unfreeze_bias = (config.bias and bias is not None and bias.requires_grad)
        
        self.base = _to_linear_like(src)
        self.base.requires_grad_(False)
        if unfreeze_bias:
            bias.requires_grad_(True)

        self.in_features = _linear_in_features(src)
        self.out_features = _linear_out_features(src)

        self.A = nn.Linear(self.in_features, rank, bias=False)
        self.B = nn.Linear(rank, self.out_features, bias=False)

        nn.init.zeros_(self.B.weight)
        
        self.merged = False
        self.can_merge = _has_mergeable_dense_weight(src)
        self.merge_weights = config.merge_weights and self.can_merge
    
    @property
    def bias(self):
        return _linear_bias(self.base)
        
    @property
    def weight(self):
        if not self.can_merge:
            raise RuntimeError(f"{type(self.base)} has no accessible dense weight")
        if self.merged: return self.base.weight
        return self.base.weight + self.lora_delta_weight()

    def lora_delta_weight(self) -> torch.Tensor:
        return self.scale * (self.B.weight @ self.A.weight)

    def train(self, mode = True):
        super().train(mode)
        if not (self.merge_weights and self.can_merge):
            return self
        
        with torch.no_grad():
            if mode:
                if self.merge_weights and self.merged:
                    self.base.weight.sub_(self.lora_delta_weight())
                    self.merged = False
            else:
                if self.merge_weights and not self.merged:
                    self.base.weight.add_(self.lora_delta_weight())
                    self.merged = True
        return self

    def forward(self, x):
        if self.merged: return self.base(x)
        return self.base(x) + self.scale * self.B(self.A(x))

def _lora_linear_is_same_to_cfg(ln: LoRALinear, cfg: LoRAConfig):
    return all([
        ln.A.weight.shape[0] == cfg.rank,
        ln.bias is None or ln.bias.requires_grad >= cfg.bias,
    ])


class LoRAEmbedding(nn.Module):
    def __init__(self, src: nn.Embedding, config: LoRAConfig):
        super().__init__()
    
        rank = config.rank
        assert rank > 0
        self.scale = config.alpha / rank
        
        self.src = _to_Embedding(src)
        self.src.requires_grad_(False)
        
        self.A = nn.Embedding(
            num_embeddings=self.num_embeddings, 
            embedding_dim=rank,
            padding_idx=self.padding_idx,
            max_norm=None,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=False)
        self.B = nn.Linear(rank, self.embedding_dim, bias=False)

        nn.init.zeros_(self.B.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.A.weight[self.padding_idx].zero_()
                
        self.merged = False
        self.merge_weights = config.merge_weights
        
    @property    
    def num_embeddings(self): return self.src.num_embeddings
    @property
    def embedding_dim(self): return self.src.embedding_dim
    @property
    def padding_idx(self): return self.src.padding_idx
    @property
    def max_norm(self): return self.src.max_norm
    @property
    def norm_type(self): return self.src.norm_type
    @property
    def scale_grad_by_freq(self): return self.src.scale_grad_by_freq
    @property
    def sparse(self): return self.src.sparse

    @property
    def weight(self):
        if self.merged:
            return self.src.weight
        return self.src.weight + self.lora_delta_weight()

    def lora_delta_weight(self) -> torch.Tensor:
        delta = self.A.weight @ self.B.weight.T

        if self.padding_idx is not None:
            delta = delta.clone()
            delta[self.padding_idx].zero_()

        return delta * self.scale

    def train(self, mode: bool = True):
        super().train(mode)
        self.src.train(mode)
        with torch.no_grad():
            if mode:
                if self.merge_weights and self.merged:
                    self.src.weight.sub_(self.lora_delta_weight())
                    self.merged = False
            else:
                if self.merge_weights and not self.merged:
                    self.src.weight.add_(self.lora_delta_weight())
                    self.merged = True

    def forward(self, x):
        if self.merged:
            return self.src.forward(x)
        base = self.src.forward(x)
        
        return base + self.B(self.A(x)) * self.scale

def _to_Embedding(src: LoRAEmbedding | nn.Embedding):
    if not isinstance(src, (LoRAEmbedding, nn.Embedding)): raise TypeError(f"Can not convert {str(type(src))} to nn.Embedding")
    if not isinstance(src, LoRAEmbedding): return src
    if not src.merged:
        with torch.no_grad():
            src.src.weight.add_(src.lora_delta_weight())
    return src.src

def _lora_embedding_same_to_cfg(src: LoRAEmbedding, cfg: LoRAConfig):
    return src.A.embedding_dim == cfg.rank


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
    "wte": "wte",
    "wpe": "wpe",
}
    
TARGETS_ON_LAYERS = {
    "attn.c_attn": ("attn", "c_attn"),
    "attn.c_proj": ("attn", "c_proj"),
    "mlp.c_fc":    ("mlp", "c_fc"),
    "mlp.c_proj":  ("mlp", "c_proj"),
}

def apply_LoRA(model: GPT, config: LoRAConfig):
    if not config.enable:
        return False
    layers = _parse_target_layers(config.target_layers, model.config.n_layer)
    if config.targets.strip() == "all":
        targets = list(TARGETS.keys()) + list(TARGETS_ON_LAYERS.keys())
    elif config.targets.strip() == "all-linear":
        targets = list(TARGETS_ON_LAYERS.keys())
    else:
        targets = [x.strip() for x in config.targets.split(",") if x.strip()]
    applied_cnt = 0
    
    _targets = []
    for target in targets:
        if target in TARGETS:
            attr = TARGETS[target]
            parent = model.transformer
            obj = getattr(parent, attr)
            
            if not isinstance(obj, (LoRAEmbedding, nn.Embedding)):
                raise ValueError(f"{target} is not a suitable object")
            if isinstance(obj, LoRAEmbedding):
                print(f"LoRA is already applied to {target}")
                if _lora_embedding_same_to_cfg(obj, config):
                    obj.scale = config.alpha / config.rank
                else:
                    obj = LoRAEmbedding(obj, config)
            else:
                obj = LoRAEmbedding(obj, config)
            
            setattr(parent, attr, obj)
            applied_cnt += 1
        else:
            _targets.append(target)
    targets = _targets
    
    for i in layers:
        for target in targets:
            if target not in TARGETS_ON_LAYERS:
                raise ValueError(f"Unsupported target to apply LoRA: {target}")
            
            subpath, attr = TARGETS_ON_LAYERS[target]
            parent = model.transformer.h[i].get_submodule(subpath)
            
            obj = getattr(parent, attr)
            
            if isinstance(obj, LoRALinear):
                print(f"Warning: LoRA is already applied to {target} on layer {i}")
                if _lora_linear_is_same_to_cfg(obj, config):
                    obj.scale = config.alpha / config.rank
                else:
                    obj = LoRALinear(obj, config)
            else:
                obj = LoRALinear(obj, config)
            setattr(parent, attr, obj)
            applied_cnt += 1
    model.lora_config = config
    return applied_cnt > 0