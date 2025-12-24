"""
Tests for the transformer model.
"""

import pytest
import torch

from src.training.model import (
    TransformerConfig,
    TransformerLM,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
)


@pytest.fixture
def config():
    """Create a small config for testing."""
    return TransformerConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
    )


@pytest.fixture
def model(config):
    """Create a model for testing."""
    return TransformerLM(config)


class TestTransformerConfig:
    """Tests for TransformerConfig."""
    
    def test_small_config(self):
        """Test small model configuration."""
        config = TransformerConfig.small()
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
    
    def test_medium_config(self):
        """Test medium model configuration."""
        config = TransformerConfig.medium()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
    
    def test_large_config(self):
        """Test large model configuration."""
        config = TransformerConfig.large()
        assert config.hidden_size == 1280
        assert config.num_hidden_layers == 36


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""
    
    def test_output_shape(self, config):
        """Test attention output shape."""
        attn = MultiHeadAttention(config)
        x = torch.randn(2, 10, config.hidden_size)
        
        output, _ = attn(x)
        
        assert output.shape == x.shape
    
    def test_with_cache(self, config):
        """Test attention with KV cache."""
        attn = MultiHeadAttention(config)
        x = torch.randn(2, 10, config.hidden_size)
        
        output, kv_cache = attn(x, use_cache=True)
        
        assert kv_cache is not None
        assert len(kv_cache) == 2  # key and value


class TestFeedForward:
    """Tests for FeedForward."""
    
    def test_output_shape(self, config):
        """Test FFN output shape."""
        ffn = FeedForward(config)
        x = torch.randn(2, 10, config.hidden_size)
        
        output = ffn(x)
        
        assert output.shape == x.shape


class TestTransformerBlock:
    """Tests for TransformerBlock."""
    
    def test_output_shape(self, config):
        """Test block output shape."""
        block = TransformerBlock(config)
        x = torch.randn(2, 10, config.hidden_size)
        
        output, _ = block(x)
        
        assert output.shape == x.shape


class TestTransformerLM:
    """Tests for TransformerLM."""
    
    def test_forward(self, model, config):
        """Test model forward pass."""
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        
        outputs = model(input_ids)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 10, config.vocab_size)
    
    def test_forward_with_labels(self, model, config):
        """Test model forward pass with labels."""
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        assert "loss" in outputs
        assert outputs["loss"].item() > 0
    
    def test_forward_with_attention_mask(self, model, config):
        """Test model forward pass with attention mask."""
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        attention_mask = torch.ones(2, 10)
        attention_mask[:, 5:] = 0  # Mask second half
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs["logits"].shape == (2, 10, config.vocab_size)
    
    def test_kv_cache(self, model, config):
        """Test KV cache during inference."""
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        outputs = model(input_ids, use_cache=True)
        
        assert outputs["past_key_values"] is not None
        assert len(outputs["past_key_values"]) == config.num_hidden_layers
    
    def test_num_parameters(self, model):
        """Test parameter counting."""
        num_params = model.num_parameters()
        assert num_params > 0
        
        num_params_no_embed = model.num_parameters(exclude_embeddings=True)
        assert num_params_no_embed < num_params
    
    @pytest.mark.slow
    def test_generate(self, model, config):
        """Test text generation."""
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        generated = model.generate(input_ids, max_length=10)
        
        assert generated.shape[1] >= 5  # At least as long as input
        assert generated.shape[1] <= 15  # At most max_length + input
    
    def test_generate_greedy(self, model, config):
        """Test greedy generation."""
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        generated = model.generate(input_ids, max_length=5, do_sample=False)
        
        assert generated.shape[1] >= 5


class TestModelTrainingCompatibility:
    """Tests for training compatibility."""
    
    def test_gradient_flow(self, model, config):
        """Test that gradients flow through the model."""
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_weight_tying(self, model):
        """Test that input and output embeddings are tied."""
        assert model.token_embeddings.weight is model.lm_head.weight
