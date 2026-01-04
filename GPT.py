'''
Simple implementation of GPT architecture
'''

import torch
import torch.nn as nn
from torchinfo import summary
from einops import rearrange


class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, max_content_size = 1024, dropout = 0):
        super().__init__()

        self.num_heads = num_heads
        self.key = nn.Linear(d_in, d_out * num_heads, bias = False)
        self.query = nn.Linear(d_in, d_out * num_heads, bias = False)
        self.value = nn.Linear(d_in, d_out * num_heads, bias = False)
        self.out_proj = nn.Linear(d_out * num_heads, d_out)

        self.register_buffer('att_mask', torch.triu(torch.ones(max_content_size, max_content_size), diagonal= 1).bool())
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        _, num_tokens, _ = x.shape

        kw = self.key(x)
        qw = self.query(x)
        vw = self.value(x)

        #re-order (batch, d_in, num_heads * d_out) -> (batch, d_in, num_heads, d_out)
        hkw = rearrange(kw, 'b t (h d) -> b h t d', h=self.num_heads)
        hqw = rearrange(qw, 'b t (h d) -> b h t d', h=self.num_heads)
        hvw = rearrange(vw, 'b t (h d) -> b h t d', h=self.num_heads)

        attn_scores = hqw @ rearrange(hkw, 'b h t d -> b h d t')
        attn_scores = attn_scores.masked_fill(self.att_mask[:num_tokens, :num_tokens], -torch.inf)

        attn_norm = torch.softmax(attn_scores / (kw.shape[-1] ** 0.5), dim = -1)
        y = self.dropout(attn_norm @ hvw)
        y = self.out_proj(rearrange(y, 'b h t d -> b t (h d)'))
        return y

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.nn(x)
    

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attention = MultiheadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["n_heads"], cfg["context_size"], cfg["drop_rate"])
        self.feedforward = FeedForward(cfg["emb_dim"])
        self.layer_norm_1 = nn.LayerNorm(cfg["emb_dim"])
        self.layer_norm_2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        x_0 = x

        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.drop(x)
        x = x + x_0


        x_0 = x

        x = self.layer_norm_2(x)
        x = self.feedforward(x)
        x = self.drop(x)
        x = x + x_0

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_size"], cfg["emb_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])

        self.decoder_blocks = nn.Sequential(
            *[Decoder(cfg) for _ in range(cfg["layers"])]
        )

        self.layer_norm = nn.LayerNorm(cfg["emb_dim"])
        self.final_out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, token_len =in_idx.shape
        in_int_idx = in_idx.to(dtype=torch.int32)

        tok_embeds = self.token_embedding(in_int_idx)
        pos_embeds = self.position_embedding(torch.arange(token_len, device = in_idx.device))

        tokens = tok_embeds + pos_embeds
        x = self.drop_embedding(tokens)

        x = self.decoder_blocks(x)
        x = self.layer_norm(x)

        return self.final_out(x)

cfg = {
    'vocab_size' : 50257, #BPE vocab size
    'context_size' : 2048,
    'emb_dim' : 768,
    'n_heads' : 12,
    'layers' : 12,
    'drop_rate' : 0.1
}


model = GPTModel(cfg)

summary(model, input_size = (1, 100))
