"""
Transformer model architecture for causal language modeling.

Implements a GPT-style decoder-only transformer.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import settings


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    tie_word_embeddings: bool = True
    
    @classmethod
    def from_settings(cls) -> "TransformerConfig":
        """Create config from settings."""
        return cls(
            vocab_size=settings.tokenization.vocab_size,
            max_position_embeddings=settings.training.max_seq_length,
        )
    
    @classmethod
    def small(cls) -> "TransformerConfig":
        """Small model config (~125M params)."""
        return cls(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
    
    @classmethod
    def medium(cls) -> "TransformerConfig":
        """Medium model config (~350M params)."""
        return cls(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )
    
    @classmethod
    def large(cls) -> "TransformerConfig":
        """Large model config (~760M params)."""
        return cls(
            hidden_size=1280,
            num_hidden_layers=36,
            num_attention_heads=20,
            intermediate_size=5120,
        )


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache for inference
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        present_key_value = (key, value) if use_cache else None
        
        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, present_key_value


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """Single transformer block with attention and FFN."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class TransformerLM(nn.Module):
    """
    Transformer Language Model for causal language modeling.
    
    GPT-style decoder-only architecture with:
    - Pre-norm layer normalization
    - GELU activation
    - Rotary position embeddings (optional)
    - KV caching for efficient inference
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype),
            diagonal=1,
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> dict:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get position IDs
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            position_ids = torch.arange(past_len, past_len + seq_len, device=device)
        else:
            position_ids = torch.arange(seq_len, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device, hidden_states.dtype)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # Expand attention mask
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * float("-inf")
            causal_mask = causal_mask + attention_mask
        
        # Forward through layers
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values else None
            
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final norm and output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling probability
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        device = input_ids.device
        past_key_values = None
        
        for _ in range(max_length):
            # Forward pass
            outputs = self(
                input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            if do_sample:
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if EOS
            if next_token.item() == self.config.eos_token_id:
                break
        
        return input_ids
    
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.token_embeddings.weight.numel()
            total -= self.position_embeddings.weight.numel()
        return total
