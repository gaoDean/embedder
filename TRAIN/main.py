import torch
from model_patcher import load_model
import config

def main():
    tokenizer, model = load_model()
    print(f"Model {config.MODEL_NAME} modified for hybrid inference!")

    device = config.DEVICE
    model.to(device)

    # Test coherence
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: {prompt}")

    # 1. Generate without context_vector
    with torch.no_grad():
        out_none = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"Generated (no context): {tokenizer.decode(out_none[0], skip_special_tokens=True)}")

    # 2. Generate with random context_vector
    vec = torch.randn(1, config.CONTEXT_DIM).to(device=device, dtype=model.dtype)
    with torch.no_grad():
        out_vec = model.generate(**inputs, max_new_tokens=20, do_sample=False, context_vector=vec)
    print(f"Generated (with context): {tokenizer.decode(out_vec[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
