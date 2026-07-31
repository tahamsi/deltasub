from __future__ import annotations
import torch
from .base import AdapterInput

def synthetic_input(*,batch_size=3,dimension=16,prefix_tokens=1,token_budget=4,seed=0,device="cpu",precision="fp32"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dtype=torch.bfloat16 if precision=="bf16" else torch.float32
        parents=torch.randn(batch_size,256,dimension,dtype=dtype,device=device)
        prefixes=torch.randn(batch_size,prefix_tokens,dimension,dtype=dtype,device=device)
        children=parents[:,:,None,:]+.01*torch.randn(batch_size,256,4,dimension,dtype=dtype,device=device)
        pos=torch.linspace(-1,1,256,device=device,dtype=dtype)[:,None].expand(256,dimension)[None].expand(batch_size,-1,-1).clone()
    return AdapterInput(tuple(f"sample-{i}" for i in range(batch_size)),tuple("view-0" for _ in range(batch_size)),
        parents,prefixes,pos,torch.zeros_like(prefixes),torch.ones(batch_size,256,dtype=torch.bool,device=device),
        token_budget,device,precision,seed,child_tokens=children,mode="fixture")
