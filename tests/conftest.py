"""A tiny synthetic Nemotron-shaped model: real module paths
(``backbone.layers.{i}.mixer``), mixer classes named like the trust_remote_code
modeling file (topology classifies by class name), and tuple-returning blocks —
so the full hook/cache/splice surface is exercised on CPU with no download."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

D_MODEL = 16
VOCAB = 50
N_EXPERTS = 4


class NemotronHMamba2Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        return self.proj(x)


class NemotronHAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        # HF attention modules typically return (hidden, attn_weights, ...)
        return self.proj(x), None


class NemotronHMOE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList(nn.Linear(D_MODEL, D_MODEL) for _ in range(N_EXPERTS))
        self.shared_experts = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        out = self.shared_experts(x) + sum(e(x) for e in self.experts) / N_EXPERTS
        return out, torch.zeros(x.shape[0])  # (hidden, router_logits)


class _Block(nn.Module):
    def __init__(self, mixer: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(D_MODEL)
        self.mixer = mixer

    def forward(self, x):
        out = self.mixer(self.norm(x))
        hidden = out[0] if isinstance(out, tuple) else out
        return (x + hidden,)  # HF blocks commonly return tuples


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList(
            [_Block(NemotronHMamba2Mixer()), _Block(NemotronHAttention()), _Block(NemotronHMOE())]
        )

    def forward(self, input_ids):
        h = self.embeddings(input_ids)
        for layer in self.layers:
            (h,) = layer(h)
        return h


class TinyNemotron(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.lm_head = nn.Linear(D_MODEL, VOCAB)
        self.config = SimpleNamespace(hidden_size=D_MODEL, use_cache=False)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.lm_head(self.backbone(input_ids)))


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return TinyNemotron()


@pytest.fixture
def hooked(tiny_model):
    from nemotron_lens import HookedNemotron

    return HookedNemotron(tiny_model)


@pytest.fixture
def tokens():
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (2, 7))
