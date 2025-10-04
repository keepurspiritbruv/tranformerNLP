import numpy as np

# Softmax function
def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# TOKEN EMBEDDING
# Embedding token untuk mengonversi token menjadi vektor
class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01  # Inisialisasi matriks embedding

    def forward(self, token_ids):
        # token_ids: array of shape (batch_size, seq_len)
        # embeddings: array of shape (batch_size, seq_len, d_model)
        return self.embedding[token_ids]


# Positional Encoding (Sinusoidal)
class PositionalEncoding:
    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        self.max_len = max_len

        # Membuat matriks positional encoding
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        # x: array of shape (batch_size, seq_len, d_model)
        # x + pe (positional encoding): array of shape (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]
    
# Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    # Qk^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Apply mask (casual mask untuk decoder)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax
    attention_weights = softmax(scores, axis=-1)

    # Attention * V
    output = np.matmul(attention_weights, V)
    return output, attention_weights

# Multi-Head Attention
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0 # d_model harus habis dibagi num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Inisialisasi bobot untuk Q, K, V, dan output
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01

    def split_heads(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)
    
    def combine_heads(self, x):
        # x: (batch_size, num_heads, seq_len, d_k)
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * d_k)
        return x  # (batch_size, seq_len, d_model)
    
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        batch_size = x.shape[0]

        # Linear projections
        Q = np.matmul(x, self.W_Q)  # (batch_size, seq_len, d_model)
        K = np.matmul(x, self.W_K)  # (batch_size, seq_len, d_model)
        V = np.matmul(x, self.W_V)  # (batch_size, seq_len, d_model)

        # Split heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len, d_k)

        # Scaled Dot-Product Attention
        batch_size, num_heads, seq_len, d_k = Q.shape
        Q = Q.reshape(batch_size * num_heads, seq_len, d_k)
        K = K.reshape(batch_size * num_heads, seq_len, d_k)
        V = V.reshape(batch_size * num_heads, seq_len, d_k)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Reshape back
        attn_output = attn_output.reshape(batch_size, num_heads, seq_len, d_k)

        # Combine heads
        output = self.combine_heads(attn_output)

        # Final linear layer
        output = np.matmul(output, self.W_O)

        return output 
    
# Feed-Forward Network
class FeedForwardNetwork:
    # Position Wise FFN: FFN(X) = max(0, XW1 + b1)W2 + b2
    def __init__(self, d_model, d_ff):
        # Inisialisasi bobot dan bias
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((d_model))

    def forward(self, x):
        hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)  # ReLU
        output = np.matmul(hidden, self.W2) + self.b2            # ✅ pakai hidden
        return output
    
# Layer Normalization
class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones((d_model))
        self.beta = np.zeros((d_model))

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        output = self.gamma * normalized + self.beta
        return output

# Casual Masking
def create_causal_mask(seq_len):
    # Membuat casual mask untuk mencegah attention ke token berikutnya
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask  # (seq_len, seq_len)

# Transformer Decoder Block
class TransformerDecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln1 = LayerNormalization(d_model)
        self.ln2 = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        # Mask: (seq_len, seq_len)
        
        # Pre-norm + MHA + Residual
        attn_ouput = self.mha.forward(self.ln1.forward(x), mask)
        out1 = x + attn_ouput  # Residual connection

        # Pre-norm + FFN + Residual
        ffn_output = self.ffn.forward(self.ln2.forward(out1))
        out2 = out1 + ffn_output  # Residual connection

        return out2
    
# Full Decoder Only Transformer
class DecoderOnlyTransformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layers
                # Embedding layers
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.blocks = [
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.ln_f = LayerNormalization(d_model)
        
        # Output projection ke vocab size
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.01
    
    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        
        # Token embedding + Positional encoding
        x = self.token_embedding.forward(token_ids)
        x = self.positional_encoding.forward(x)
        
        # Create causal mask
        mask = create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Final layer norm
        x = self.ln_f.forward(x)
        
        # Project to vocabulary size
        logits = np.matmul(x, self.output_projection)
        
        # Get probability distribution for next token (last position)
        last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        probs = softmax(last_token_logits, axis=-1)
        
        return logits, probs
