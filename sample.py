"""
Sample from a trained model
"""
import os
import json
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from loader import load_model

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume' - artefact from train.py
model_path = ""
output_path = None
append_mode = True
jsonify_output = False
json_header = {}
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
if not model_path:
    model_path = None
if not isinstance(json_header, dict):
    json_header = json.loads(json_header)
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
load_data = load_model(
    init_from=init_from,
    out_dir=out_dir,
    model_path=model_path,
    device=device
)

model = load_data['model']
checkpoint = load_data['checkpoint']

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

def sampling(start: str, json_header: dict):
    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    res = [None] * num_samples
    sep = '---------------'
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                
                cur = decode(y[0][len(start_ids):].tolist())
                
                if not jsonify_output:
                    cur += f'\n{sep}'
                    
                res[k] = cur
                
                print(f'{cur}\n{sep}')
                
    if output_path is not None:
        mode = "a" if append_mode else "w"
        with open(output_path, mode) as f:
            if jsonify_output:
                for i in range(len(res)):
                    text = res[i]
                    res[i] = json_header
                    res[i]['sample_id'] = i
                    res[i]['generated_text'] = text
                    res[i]['max_new_tokens'] = max_new_tokens
                    res[i] = json.dumps(res[i])
            res = '\n'.join(res)
            f.write(res + '\n')

if start.startswith('JSONFILE:'):
    starts = []
    with open(start[9:], 'r', encoding='utf-8') as f:
        for s in f:
            starts.append(json.loads(s))
    if not append_mode:
        with open(output_path, "w") as f:
            append_mode = True
    for s in starts:
        t = dict(json_header)
        t.update(s)
        sampling(s['prompt'], t)
else:
    sampling(start, json_header)