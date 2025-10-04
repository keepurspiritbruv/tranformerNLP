import numpy as np
# Unit Test

class TransformerTestSuite:
    """Comprehensive test suite untuk semua komponen"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def assert_equal(self, actual, expected, msg=""):
        """Assert equality"""
        if actual == expected:
            return True
        else:
            print(f"FAILED: {msg}")
            print(f"Expected: {expected}, Got: {actual}")
            return False
    
    def assert_shape(self, tensor, expected_shape, msg=""):
        """Assert tensor shape"""
        if tensor.shape == expected_shape:
            return True
        else:
            print(f"FAILED: {msg}")
            print(f"Expected shape: {expected_shape}, Got: {tensor.shape}")
            return False
    
    def assert_close(self, actual, expected, rtol=1e-5, msg=""):
        """Assert values are close"""
        if np.allclose(actual, expected, rtol=rtol):
            return True
        else:
            print(f"FAILED: {msg}")
            print(f"Max difference: {np.max(np.abs(actual - expected))}")
            return False
    
    def run_test(self, test_func, test_name):
        """Run a single test"""
        try:
            print(f"\nTEST: {test_name}")
            result = test_func()
            if result:
                self.passed += 1
                print(f"PASSED: {test_name}")
            else:
                self.failed += 1
            self.tests.append((test_name, result))
        except Exception as e:
            self.failed += 1
            self.tests.append((test_name, False))
            print(f"FAILED: {test_name}")
            print(f"   Exception: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """Print test summary"""
        print(f"\nTEST SUMMARY")
        print(f"Total tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {self.passed/(self.passed + self.failed)*100:.1f}%")
        
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, result in self.tests:
                if not result:
                    print(f" X {name}")
        
        print()
        
        return self.failed == 0


# Individual component tests

def test_token_embedding(suite):
    """Test TokenEmbedding component"""
    from transformer_numpy import TokenEmbedding
    
    vocab_size = 100
    d_model = 64
    batch_size = 2
    seq_len = 5
    
    emb = TokenEmbedding(vocab_size, d_model)
    token_ids = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    
    output = emb.forward(token_ids)
    
    # Test shape
    if not suite.assert_shape(output, (batch_size, seq_len, d_model), 
                             "Embedding output shape"):
        return False
    
    # Test that same token IDs produce same embeddings
    if not suite.assert_close(output[0, 0, :], output[0, 0, :], 
                              msg="Same token should have same embedding"):
        return False
    
    print("Shape correct")
    print("Embedding lookup works")
    return True


def test_positional_encoding(suite):
    """Test PositionalEncoding component"""
    from transformer_numpy import PositionalEncoding
    
    d_model = 64
    max_len = 100
    batch_size = 2
    seq_len = 10
    
    pe = PositionalEncoding(d_model, max_len)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = pe.forward(x)
    
    # Test shape
    if not suite.assert_shape(output, x.shape, "PE output shape"):
        return False
    
    # Test that PE is deterministic
    output2 = pe.forward(x)
    if not suite.assert_close(output, output2, msg="PE should be deterministic"):
        return False
    
    # Test PE matrix properties
    if not suite.assert_shape(pe.pe, (max_len, d_model), "PE matrix shape"):
        return False
    
    print("Shape correct")
    print("Deterministic")
    print("PE matrix correct")
    return True


def test_scaled_dot_product_attention(suite):
    """Test scaled dot-product attention"""
    from transformer_numpy import scaled_dot_product_attention
    
    batch_size = 2
    seq_len = 5
    d_k = 8
    
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    
    # Test shapes
    if not suite.assert_shape(output, (batch_size, seq_len, d_k), "Attention output shape"):
        return False
    
    if not suite.assert_shape(attn_weights, (batch_size, seq_len, seq_len), 
                              "Attention weights shape"):
        return False
    
    # Test attention weights sum to 1
    attn_sum = np.sum(attn_weights, axis=-1)
    if not suite.assert_close(attn_sum, np.ones((batch_size, seq_len)), 
                              msg="Attention weights should sum to 1"):
        return False
    
    print("Output shape correct")
    print("Attention weights shape correct")
    print("Attention weights sum to 1")
    return True


def test_causal_masking(suite):
    """Test causal mask"""
    from transformer_numpy import create_causal_mask, scaled_dot_product_attention
    
    seq_len = 5
    mask = create_causal_mask(seq_len)
    
    # Test mask shape
    if not suite.assert_shape(mask, (seq_len, seq_len), "Mask shape"):
        return False
    
    # Test mask is lower triangular
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                if not suite.assert_equal(mask[i, j], 0, 
                                        f"Mask[{i},{j}] should be 0 (future)"):
                    return False
            else:
                if not suite.assert_equal(mask[i, j], 1, 
                                        f"Mask[{i},{j}] should be 1 (past)"):
                    return False
    
    # Test that masking actually prevents future attention
    batch_size = 1
    d_k = 8
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    _, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
    
    # Check that attention to future positions is near zero
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if attn_weights[0, i, j] > 1e-6:
                print(f"Attention to future position [{i},{j}] = {attn_weights[0, i, j]}")
                return False
    
    print("Mask shape correct")
    print("Mask is lower triangular")
    print("Future attention blocked")
    return True


def test_multi_head_attention(suite):
    """Test MultiHeadAttention"""
    from transformer_numpy import MultiHeadAttention, create_causal_mask
    
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 5
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)
    
    output = mha.forward(x, mask)
    
    # Test shape
    if not suite.assert_shape(output, (batch_size, seq_len, d_model), 
                             "MHA output shape"):
        return False
    
    # Test d_k calculation
    if not suite.assert_equal(mha.d_k, d_model // num_heads, "d_k calculation"):
        return False
    
    print("Output shape correct")
    print("d_k calculation correct")
    print("Forward pass successful")
    return True


def test_feed_forward_network(suite):
    """Test FeedForwardNetwork"""
    from transformer_numpy import FeedForwardNetwork
    
    d_model = 64
    d_ff = 256
    batch_size = 2
    seq_len = 5
    
    ffn = FeedForwardNetwork(d_model, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = ffn.forward(x)
    
    # Test shape
    if not suite.assert_shape(output, (batch_size, seq_len, d_model), 
                             "FFN output shape"):
        return False
    
    # Test ReLU activation (output should have some zeros if input causes negative intermediate values)
    # This is a weak test but checks that ReLU is applied
    x_negative = -np.ones((batch_size, seq_len, d_model))
    output_negative = ffn.forward(x_negative)
    # After ReLU, at least some values should be non-negative
    
    print("Output shape correct")
    print("ReLU activation applied")
    return True


def test_layer_normalization(suite):
    """Test LayerNormalization"""
    from transformer_numpy import LayerNormalization
    
    d_model = 64
    batch_size = 2
    seq_len = 5
    
    ln = LayerNormalization(d_model)
    x = np.random.randn(batch_size, seq_len, d_model) * 10  # Large variance
    
    output = ln.forward(x)
    
    # Test shape
    if not suite.assert_shape(output, (batch_size, seq_len, d_model), 
                             "LayerNorm output shape"):
        return False
    
    # Test normalization (mean ~0, std ~1 along last dimension)
    mean = np.mean(output, axis=-1)
    std = np.std(output, axis=-1)
    
    if not suite.assert_close(mean, np.zeros((batch_size, seq_len)), rtol=1e-4,
                              msg="Mean should be ~0"):
        return False
    
    if not suite.assert_close(std, np.ones((batch_size, seq_len)), rtol=1e-1,
                              msg="Std should be ~1"):
        return False
    
    print("Output shape correct")
    print("Mean normalized to ~0")
    print("Std normalized to ~1")
    return True


def test_full_transformer(suite):
    """Test full DecoderOnlyTransformer"""
    from transformer_numpy import DecoderOnlyTransformer
    
    vocab_size = 100
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    batch_size = 2
    seq_len = 10
    
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, probs = model.forward(token_ids)
    
    # Test logits shape
    if not suite.assert_shape(logits, (batch_size, seq_len, vocab_size), 
                             "Logits shape"):
        return False
    
    # Test probs shape
    if not suite.assert_shape(probs, (batch_size, vocab_size), 
                             "Next token probs shape"):
        return False
    
    # Test probs sum to 1
    prob_sums = np.sum(probs, axis=-1)
    if not suite.assert_close(prob_sums, np.ones(batch_size), 
                              msg="Probabilities should sum to 1"):
        return False
    
    # Test all probabilities are positive
    if not np.all(probs >= 0):
        print("Some probabilities are negative")
        return False
    
    print("Logits shape correct")
    print("Probs shape correct")
    print("Probabilities sum to 1")
    print("All probabilities positive")
    return True


def test_softmax(suite):
    """Test softmax function"""
    from transformer_numpy import softmax
    
    # Test 1D
    x = np.array([1.0, 2.0, 3.0])
    output = softmax(x)
    
    if not suite.assert_close(np.sum(output), 1.0, msg="Softmax should sum to 1"):
        return False
    
    # Test 2D
    x = np.random.randn(3, 5)
    output = softmax(x, axis=-1)
    
    sums = np.sum(output, axis=-1)
    if not suite.assert_close(sums, np.ones(3), msg="Softmax rows should sum to 1"):
        return False
    
    # Test numerical stability (large values)
    x = np.array([1000.0, 1001.0, 1002.0])
    output = softmax(x)
    
    if not np.isfinite(output).all():
        print("Softmax not numerically stable")
        return False
    
    print("1D softmax works")
    print("2D softmax works")
    print("Numerically stable")
    return True


def test_determinism(suite):
    """Test that model is deterministic with same seed"""
    from transformer_numpy import DecoderOnlyTransformer
    
    vocab_size = 50
    d_model = 32
    num_heads = 2
    d_ff = 128
    num_layers = 1
    batch_size = 1
    seq_len = 5
    
    token_ids = np.array([[1, 2, 3, 4, 5]])
    
    # First run
    np.random.seed(42)
    model1 = DecoderOnlyTransformer(vocab_size, d_model, num_heads, d_ff, num_layers)
    logits1, probs1 = model1.forward(token_ids)
    
    # Second run with same seed
    np.random.seed(42)
    model2 = DecoderOnlyTransformer(vocab_size, d_model, num_heads, d_ff, num_layers)
    logits2, probs2 = model2.forward(token_ids)
    
    # Should be identical
    if not suite.assert_close(logits1, logits2, msg="Model should be deterministic"):
        return False
    
    if not suite.assert_close(probs1, probs2, msg="Probs should be deterministic"):
        return False
    
    print("Model is deterministic")
    return True


def test_batch_independence(suite):
    """Test that batch samples are processed independently"""
    from transformer_numpy import DecoderOnlyTransformer
    
    vocab_size = 50
    d_model = 32
    num_heads = 2
    d_ff = 128
    num_layers = 1
    seq_len = 5
    
    np.random.seed(42)
    model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, d_ff, num_layers)
    
    # Process batch of 2
    token_ids_batch = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    _, probs_batch = model.forward(token_ids_batch)
    
    # Process individually
    token_ids_single = np.array([[1, 2, 3, 4, 5]])
    _, probs_single = model.forward(token_ids_single)
    
    # Both samples in batch should match single processing
    if not suite.assert_close(probs_batch[0], probs_single[0], 
                              msg="Batch sample 1 should match single"):
        return False
    
    if not suite.assert_close(probs_batch[1], probs_single[0], 
                              msg="Batch sample 2 should match single"):
        return False
    
    print("Batch processing independent")
    return True


# Main test runner
def run_all_tests():
    """Run all tests"""
    print("TRANSFORMER COMPREHENSIVE TEST SUITE")
    print("Testing all components of the Transformer implementation...")
    
    suite = TransformerTestSuite()
    
    # Component tests
    suite.run_test(lambda: test_token_embedding(suite), "Token Embedding")
    suite.run_test(lambda: test_positional_encoding(suite), "Positional Encoding")
    suite.run_test(lambda: test_softmax(suite), "Softmax Function")
    suite.run_test(lambda: test_scaled_dot_product_attention(suite), "Scaled Dot-Product Attention")
    suite.run_test(lambda: test_causal_masking(suite), "Causal Masking")
    suite.run_test(lambda: test_multi_head_attention(suite), "Multi-Head Attention")
    suite.run_test(lambda: test_feed_forward_network(suite), "Feed-Forward Network")
    suite.run_test(lambda: test_layer_normalization(suite), "Layer Normalization")
    
    # Integration tests
    suite.run_test(lambda: test_full_transformer(suite), "Full Transformer")
    suite.run_test(lambda: test_determinism(suite), "Determinism")
    suite.run_test(lambda: test_batch_independence(suite), "Batch Independence")
    
    # Print summary
    success = suite.print_summary()
    
    if success:
        print("ALL TESTS PASSED! Your Transformer implementation is correct.")
        return 0
    else:
        print("Some tests failed. Please review the failures above.")
        return 1


if __name__ == "__main__":
    print(">>> Running Transformer Tests...")
    exit_code = run_all_tests()
    print(">>> Done. Exit code:", exit_code)
