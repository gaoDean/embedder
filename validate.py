import torch
from model_patcher import load_model
import config as cfg
from jina_inference import Jina
import checkpoints

text = "This is a testing sentence."
TESTS = 10

@torch.no_grad()
def main():
    cp = checkpoints.get_latest_checkpoint()

    tokenizer, model = load_model()
    model.to("mps")
    checkpoint = torch.load(cp, map_location="mps")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model.float()

    jina = Jina()
    vec = jina.model(text).to("mps")

    for i in range(TESTS):
        tokenizer_out = tokenizer(
            "This",
            add_special_tokens=True,
            return_tensors="pt"
        ).to("mps")

        out = model.generate(
            **tokenizer_out,
            max_new_tokens=50,
            eos_token_id=model.config.eos_token_id,
            pad_token=model.config.pad_token_id,
            do_sample=True,          # Enables random sampling
            temperature=0.7,         # Adds randomness (lower is more deterministic)
            top_p=0.8,               # Nucleus sampling
            repetition_penalty=1.2,   # Penalizes repeating the same words
            context_vector=vec
        )

        print(tokenizer.batch_decode(out, skip_special_tokens=False))


if __name__ == "__main__":
    main()
