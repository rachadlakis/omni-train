from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    n_layers: int = 2
    vocab_size: int = 8
    max_seq_len: int = 16
    dim: int = 16
    n_heads: int = 4
    dropout_p: float = 0.1


class Attention(nn.Module):
    """Multi-head self-attention module with residual dropout."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0

        # ------------------------------------------------------------------ #
        # STEP 1: Compute Head Dimensions
        # Split the model dimension evenly across heads so each head operates
        # on an independent subspace of the full embedding.
        # ------------------------------------------------------------------ #
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)

        # ------------------------------------------------------------------ #
        # STEP 2: Initialize Projection Weights
        # Define separate linear projections for queries, keys, values, and
        # the final output recombination. No bias terms are used.
        # ------------------------------------------------------------------ #
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()

        # ------------------------------------------------------------------ #
        # STEP 1: Project Inputs to Query, Key, and Value Spaces
        # Apply the learned linear maps to produce Q, K, V tensors from the
        # input, then reshape them into per-head views for parallel attention.
        # ------------------------------------------------------------------ #
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        # ------------------------------------------------------------------ #
        # STEP 2: Transpose for Head-First Layout
        # Rearrange dimensions so each head can attend over the full sequence
        # independently, yielding shape (bsz, n_heads, seq_len, head_dim).
        # ------------------------------------------------------------------ #
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # ------------------------------------------------------------------ #
        # STEP 3: Compute Scaled Dot-Product Attention
        # Use PyTorch's fused kernel for efficiency. Dropout is applied during
        # training only; no explicit mask is provided (full attention).
        # ------------------------------------------------------------------ #
        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,
            self.dropout_p if self.training else 0,
        )

        # ------------------------------------------------------------------ #
        # STEP 4: Recombine Heads and Project Output
        # Transpose back to sequence-first layout, flatten the head dimension,
        # apply the output projection, then apply residual dropout.
        # ------------------------------------------------------------------ #
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))

    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network with GELU activation."""

    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()

        # ------------------------------------------------------------------ #
        # STEP 1: Define Feed-Forward Layers
        # Expand to hidden_dim with w1, apply GELU non-linearity, then project
        # back to dim with w2. Residual dropout is applied on the output.
        # ------------------------------------------------------------------ #
        self.w1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()


class TransformerBlock(nn.Module):
    """Single transformer layer: pre-norm attention followed by pre-norm FFN."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        # ------------------------------------------------------------------ #
        # STEP 1: Initialize Sub-Layers with Pre-Norm
        # Each sub-layer (attention and FFN) is preceded by a LayerNorm and
        # followed by a residual connection, following the pre-LN convention.
        # ------------------------------------------------------------------ #
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p)

    def forward(self, x):
        # ------------------------------------------------------------------ #
        # STEP 2: Apply Attention and FFN with Residual Connections
        # Normalize before each sub-layer and add the input back afterward
        # to form the two residual branches that make up the transformer block.
        # ------------------------------------------------------------------ #
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def reset_parameters(self):
        self.attention_norm.reset_parameters()
        self.attention.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters()


# A toy transformer model, partly inspired by the nanoGPT model:
# https://github.com/karpathy/nanoGPT.
class Transformer(nn.Module):
    """Full transformer model with token and positional embeddings."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.model_args = args
        self.max_seq_len = args.max_seq_len

        # ------------------------------------------------------------------ #
        # STEP 1: Initialize Embedding Tables
        # Token embeddings map vocabulary indices to vectors; positional
        # embeddings encode each sequence position up to max_seq_len.
        # ------------------------------------------------------------------ #
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = nn.Dropout(args.dropout_p)

        # ------------------------------------------------------------------ #
        # STEP 2: Stack Transformer Blocks
        # Build a sequence of N identical TransformerBlock layers to form
        # the main body of the network.
        # ------------------------------------------------------------------ #
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        # ------------------------------------------------------------------ #
        # STEP 3: Initialize Output Head
        # A final LayerNorm stabilizes representations before the linear
        # projection maps hidden states back to vocabulary logits.
        # ------------------------------------------------------------------ #
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens):
        _bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len

        # ------------------------------------------------------------------ #
        # STEP 1: Compute Input Representations
        # Sum token and positional embeddings to produce the initial hidden
        # state, then apply dropout before passing through the layer stack.
        # ------------------------------------------------------------------ #
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)
        h = h + p
        h = self.dropout(h)

        # ------------------------------------------------------------------ #
        # STEP 2: Pass Through Transformer Layers
        # Feed the embedded input sequentially through each TransformerBlock,
        # accumulating contextual representations across layers.
        # ------------------------------------------------------------------ #
        for layer in self.layers:
            h = layer(h)

        # ------------------------------------------------------------------ #
        # STEP 3: Project to Vocabulary Logits
        # Apply the final norm then the output linear layer to produce per-token
        # logits over the vocabulary. Cast to float for numerical stability.
        # ------------------------------------------------------------------ #
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def reset_parameters(self):
        self.tok_embeddings.reset_parameters()
        self.pos_embeddings.reset_parameters()
        self.norm.reset_parameters()
        self.output.reset_parameters()