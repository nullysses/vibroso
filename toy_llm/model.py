from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = dict[str, torch.Tensor]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def build_rope_cache(block_size: int, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(block_size, dtype=torch.float32)
    inv_freq = 1.0 / (10_000 ** (torch.arange(0, head_size, 2).float() / head_size))
    freqs = torch.outer(positions, inv_freq)
    angles = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return angles.cos().unsqueeze(0), angles.sin().unsqueeze(0)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    t = x.shape[-2]
    cos = cos[:, position_offset : position_offset + t, :].to(dtype=x.dtype)
    sin = sin[:, position_offset : position_offset + t, :].to(dtype=x.dtype)
    if x.ndim == 4:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_size = n_embd // n_head
        if head_size % 2 != 0:
            raise ValueError("RoPE requires each attention head size to be even")
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = head_size
        self.block_size = block_size
        self.dropout = dropout
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
        rope_cos, rope_sin = build_rope_cache(block_size, head_size)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def _attention(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        b, t, c = x.shape
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.block_size}")

        cache_len = 0
        if kv_cache is not None:
            cache_len = int(kv_cache["k"].shape[2])
            keep = max(self.block_size - t, 0)
            if cache_len > keep:
                kv_cache = {
                    "k": kv_cache["k"][:, :, -keep:, :] if keep else kv_cache["k"][:, :, :0, :],
                    "v": kv_cache["v"][:, :, -keep:, :] if keep else kv_cache["v"][:, :, :0, :],
                }
                cache_len = int(kv_cache["k"].shape[2])

        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_size).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin, position_offset=cache_len)
        k = apply_rope(k, self.rope_cos, self.rope_sin, position_offset=cache_len)

        if kv_cache is not None:
            k = torch.cat((kv_cache["k"], k), dim=2)
            v = torch.cat((kv_cache["v"], v), dim=2)
            if k.shape[2] > self.block_size:
                k = k[:, :, -self.block_size :, :]
                v = v[:, :, -self.block_size :, :]
        new_cache = {"k": k, "v": v} if use_cache else None

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            # When there is no cache, this is the full prompt/training path and SDPA
            # must apply a causal mask. When a KV cache exists, q contains only the new
            # token positions and k/v contain only past + current tokens, so there are no
            # future tokens to mask.
            is_causal=kv_cache is None,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y)), new_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._attention(x)
        return out

    def forward_with_cache(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        out, new_cache = self._attention(x, kv_cache=kv_cache, use_cache=True)
        assert new_cache is not None
        return out, new_cache


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float, multiple_of: int = 8):
        super().__init__()
        hidden_dim = int((8 / 3) * n_embd)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        mlp_type: str = "gelu",
    ):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        if mlp_type == "gelu":
            self.ffwd = FeedForward(n_embd, dropout)
        elif mlp_type == "swiglu":
            self.ffwd = SwiGLUFeedForward(n_embd, dropout)
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type!r}")
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        attn_out, new_cache = self.sa.forward_with_cache(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_cache


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        mlp_type: str = "gelu",
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (n_embd // n_head) % 2 != 0:
            raise ValueError("RoPE requires n_embd / n_head to be even")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                    mlp_type=mlp_type,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.block_size}")
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))
        return logits, loss

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        kv_caches: list[KVCache | None] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        _, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.block_size}")
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        if len(kv_caches) != len(self.blocks):
            raise ValueError("kv_caches length must match number of transformer blocks")

        x = self.token_embedding_table(idx)
        new_caches: list[KVCache] = []
        for block, kv_cache in zip(self.blocks, kv_caches):
            x, new_cache = block.forward_with_cache(x, kv_cache)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        return self.lm_head(x), new_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0 when provided")
        kv_caches: list[KVCache | None] | None = None
        for _ in range(max_new_tokens):
            if use_kv_cache:
                idx_cond = idx[:, -self.block_size :] if kv_caches is None else idx[:, -1:]
                logits, kv_caches = self.forward_with_cache(idx_cond, kv_caches)
            else:
                idx_cond = idx[:, -self.block_size :]
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
