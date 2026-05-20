import torch
from transformers import AutoModelForCausalLM
from pythia import load_model

model_name = "EleutherAI/pythia-70m-deduped"
m1 = AutoModelForCausalLM.from_pretrained(model_name)
_, m2 = load_model()

p1 = next(m1.parameters())
p2 = next(m2.parameters())

print(f"Param diff: {(p1 - p2).abs().max().item()}")
