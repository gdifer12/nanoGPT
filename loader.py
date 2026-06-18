import os

import torch

from model import GPTConfig, GPT
from quant import freeze_base_model, apply_quantizing, dict_to_quant_config
from lora import apply_LoRA, dict_to_lora_config


def load_model(
    init_from: str = 'scratch', 
    out_dir: str | None = None,
    device: str = 'cuda',
    model_args: dict | None = None,
    meta_vocab_size: int | None = None,
    dropout: int | None = None,
    quant_enable: bool = False,
    block_size: int | None = None,
    saving_mode: str = 'auto',
    iter_num: int = 0,
    best_val_loss: float = 1e9,
) -> GPT:
    no_special_model_args = (model_args is None)
    if no_special_model_args: model_args = dict()
    checkpoint = None

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        base_model = None
        saving_mode = 'full'
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout) if dropout is not None else None
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
            
        base_model = init_from
    elif init_from == 'resume':
        if model_path is None:
            model_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Initial model not found on path: {model_path}")
        print(f"Resuming from {model_path}")
        
        checkpoint = torch.load(model_path, map_location="cpu" if quant_enable else device)
        checkpoint_model_args = checkpoint['model_args']
        if no_special_model_args: 
            model_args = checkpoint_model_args
        else:
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        old_quant = dict_to_quant_config(checkpoint.get('quant_config', None))
        old_lora = dict_to_lora_config(checkpoint.get('lora_config', None))
        
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        def strip_prefix_inplace(state_dict, prefix="_orig_mod."):
            for k, v in list(state_dict.items()):
                if k.startswith(prefix):
                    state_dict[k[len(prefix):]] = state_dict.pop(k)    
        
        state_dict = checkpoint['model']
        strip_prefix_inplace(state_dict)
        
        checkpoint_kind = checkpoint.get('checkpoint_kind', 'full')
        if checkpoint_kind == 'full':
            if old_quant.enable:
                raise NotImplementedError("Full quantized checkpoints are not supported yet")

            if old_lora.enable:
                apply_LoRA(model, old_lora)
            
            model.load_state_dict(state_dict)
            
            base_model = model_path
        else:
            # TODO не работает если чекпоинт сохранён при init_from = "gpt2*"
            base_model_path = checkpoint['base_model']
            if not os.path.exists(base_model_path) or not os.path.isfile(base_model_path):
                print(f"Base model not found on path: {base_model_path}")
                exit(1)
            print(f"Loading waights from base: {base_model_path}")
            
            
            base_checkpoint = torch.load(base_model_path, map_location=device)
            base_state_dict = base_checkpoint['model']
            strip_prefix_inplace(base_state_dict)
            model.load_state_dict(base_state_dict, strict=True)
            
            if block_size is None:
                block_size = model.config.block_size
            elif block_size < model.config.block_size:
                model.crop_block_size(block_size)
                model_args['block_size'] = block_size

            if old_quant.enable:
                freeze_base_model(model)
                apply_quantizing(model, old_quant)

            if old_lora.enable:
                apply_LoRA(model, old_lora)

            model.load_state_dict(state_dict, strict=False)
            
            base_model = base_model_path
        
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        raise ValueError(f'Unable to init from: {init_from}, can init from: scratch, resume, gpt2*')
    
    # crop down the model block size if desired, using model surgery
    if block_size is not None and block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    
    return {
        'model': model,
        'model_args': model_args,
        'checkpoint': checkpoint,
        'base_model': base_model,
        'saving_mode': saving_mode,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }