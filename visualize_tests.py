import os
import numpy as np
import matplotlib.pyplot as plt
from transformer_numpy import TokenEmbedding, softmax, create_causal_mask

# ==== Setup folder hasil ====
base_dir = os.path.dirname(os.path.abspath(__file__))   # folder script ini
output_dir = os.path.join(base_dir, "results")          # subfolder results
os.makedirs(output_dir, exist_ok=True)

# ==== Token Embedding ====
print("=== Embedding Test ===")
vocab_size, d_model, seq_len, batch_size = 1000, 64, 5, 2
embedding = TokenEmbedding(vocab_size, d_model)
token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
out = embedding.forward(token_ids)
print("Token IDs:\n", token_ids)
print("Embedding output shape:", out.shape)
print("Sample embedding vector (first token, first 5 dims):\n", out[0, 0, :5])
print()

# ==== Softmax ====
print("=== Softmax Test ===")
x = np.array([1.0, 2.0, 3.0])
out_softmax = softmax(x)
print("Input:", x)
print("Softmax output:", out_softmax)
print("Sum of probabilities:", np.sum(out_softmax))

# Plot softmax distribution
plt.figure(figsize=(5,4))
plt.bar(range(len(out_softmax)), out_softmax, color="skyblue")
plt.title("Softmax Distribution")
plt.xlabel("Index")
plt.ylabel("Probability")
softmax_path = os.path.join(output_dir, "softmax_distribution.png")
plt.savefig(softmax_path)
print(f"✅ Softmax plot saved to: {softmax_path}")
plt.close()

# ==== Causal Mask ====
print("=== Causal Mask Test ===")
seq_len = 5
mask = create_causal_mask(seq_len)
print("Causal mask:\n", mask)

# Plot causal mask heatmap
plt.figure(figsize=(5,4))
plt.matshow(mask, cmap="viridis")
plt.title("Causal Mask Heatmap", pad=20)
plt.colorbar()
mask_path = os.path.join(output_dir, "causal_mask.png")
plt.savefig(mask_path)
print(f"✅ Causal mask plot saved to: {mask_path}")
plt.close()
