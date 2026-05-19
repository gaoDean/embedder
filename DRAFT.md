# Inverse Embedder Training Pipeline Draft

## Overview
The goal of this pipeline is to train an "inverse embedder" model capable of taking a text embedding vector and generating text that encapsulates the original semantic meaning of that embedding. 

## Base Architecture
- **Base Model:** Pythia-70m (`EleutherAI/pythia-70m-deduped`), an autoregressive causal language model.
- **Modification:** A cross-attention layer will be inserted into each transformer block of Pythia-70m, positioned sequentially after the self-attention layer. These cross-attention weights will be initialized neutrally (e.g., zero-initialized) so that they do not disrupt the pre-trained features initially. 
- **Input Embedding:** The cross-attention mechanism will attend to the target embedding vector as a single length-1 sequence.

## Training Process
1. **Pre-computation:** 
   - Pass the text dataset through the Jina embedding model to extract the target vector embeddings.
   - Cache these target embeddings to avoid redundant computation and speed up training.

2. **Generation:** 
   - For a given training step, inject a cached target embedding into the modified Pythia-70m model's cross-attention layers.
   - The modified Pythia model generates a sequence of 512 tokens.

3. **Evaluation via Embedding Model:** 
   - Feed the 512 generated tokens into the *frozen* Jina embedding model.
   - Extract the generated embedding representation of this text.
   - Compute a global loss based on the distance (e.g., L2 norm or cosine distance) between the generated embedding and the ideal target embedding.

4. **Token-Level Cost Calculation (Gradient-based Relevance):** 
   - Backpropagate the global embedding loss through the frozen Jina model.
   - Capture the gradients at the continuous input embedding layer (as demonstrated via hooks in `inv_test.py`).
   - Compute the gradient magnitude (contribution) for each of the 512 tokens. 
   - **Cost Assignment:** Tokens with high gradient magnitudes are highly relevant to the semantic goal, so they are assigned a *low cost* (or high reward). Conversely, tokens with low gradient magnitudes are irrelevant and are assigned a *high cost*.

5. **Thesaurus & Semantic Smoothing:**
   - To make the training signal robust, introduce a thesaurus mechanism (e.g., a pre-computed similarity matrix of the tokenizer's vocabulary embeddings).
   - Distribute the calculated token-level costs across the vocabulary based on semantic similarity. 
   - If a specific generated word is penalized (high cost), its immediate synonyms and semantically similar words in the thesaurus will also receive a proportional penalty. 
   - Similarly, words similar to highly relevant/rewarded words will share the low cost. This ensures the model learns the semantic neighborhood rather than just suppressing exact token IDs, preventing it from simply swapping a penalized word for an exact synonym.

6. **Pythia Backpropagation & Update (Vectorized / Reward-Weighted):**
   - Generating 512 tokens autoregressively creates a massive sequential computation graph. Directly backpropagating through this loop step-by-step 512 times is highly inefficient and memory-intensive in PyTorch.
   - **Better Method (Reward-Weighted Training / Policy Gradient style):** Once the 512 tokens are generated and their smoothed costs are calculated via the Jina model, we detach the token sequence and treat it as the "target".
   - We perform a single, parallel forward pass through Pythia using these 512 tokens (exactly like standard LLM training via Teacher Forcing). This produces the logits for all 512 positions simultaneously.
   - We calculate the cross-entropy loss for each token at each position.
   - **Applying the Cost:** We multiply the cross-entropy loss of each token by its computed "cost" (high cost = large penalty/update, low cost/high reward = small or negative penalty).
   - **Language Modeling Loss:** We add the standard autoregressive language modeling cross-entropy loss to maintain fluency, governed by an importance hyperparameter.
   - We sum/average these combined losses into a single scalar value and call `.backward()` **exactly once**. PyTorch's autograd engine will automatically and efficiently distribute the gradients across all 512 sequence steps in parallel, updating the cross-attention and model weights.

## Open Questions & Clarifications
*All previous open questions have been addressed.*