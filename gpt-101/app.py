# tiny_gpt_final.py

# imports
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import random

torch.manual_seed(1337); random.seed(1337)

# ---------- device ----------
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# ---------- GPU hyperparameters ----------
block_size   = 128
embed_dim    = 128
n_head       = 4
n_layer      = 4
batch_size   = 64
learning_rate= 3e-4
max_iters    = 2000
attn_dropout = 0.1
resid_dropout= 0.1
mlp_dropout  = 0.1



# ---------- Sample CPU hyperparameters ----------
block_size   = 16
embed_dim    = 32
n_head       = 2
n_layer      = 2
batch_size   = 8
learning_rate= 3e-4
max_iters    = 100
attn_dropout = 0.1
resid_dropout= 0.1
mlp_dropout  = 0.1

# ---------- dataset ----------
text = Path("input.txt").read_text(encoding="utf-8")
chars = sorted(list(set(text)))
vocab_size = len(chars)

charToIndexMapping = {ch: i for i, ch in enumerate(chars)}
indexToCharMapping = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [charToIndexMapping[c] for c in s]

def decode(indices):
    return "".join([indexToCharMapping[i] for i in indices])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_dataset_batch(split):
    src = train_data if split == "train" else val_data
    hi = len(src) - block_size - 1
    ix = torch.randint(hi, (batch_size,))
    x = torch.stack([src[i:i+block_size]       for i in ix]).to(torch.long)
    y = torch.stack([src[i+1:i+block_size+1]   for i in ix]).to(torch.long)
    return x.to(device), y.to(device)

# ---------- transformer ----------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.K = nn.Linear(embed_dim, head_size, bias=False)
        self.Q = nn.Linear(embed_dim, head_size, bias=False)
        self.V = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
    def forward(self, x):
        B, T, C = x.shape
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        scale = math.sqrt(k.size(-1))
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, n_head):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)
    def forward(self, x):
        h = torch.cat([head(self.ln(x)) for head in self.heads], dim=-1)
        h = self.proj(h)
        h = self.attn_drop(h)
        return x + self.resid_drop(h)

class MLP(nn.Module):
    def __init__(self, expansion=4):
        super().__init__()
        hidden = embed_dim * expansion
        self.ln = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(mlp_dropout)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.resid_drop = nn.Dropout(resid_dropout)
    def forward(self, x):
        h = self.fc2(self.drop(self.act(self.fc1(self.ln(x)))))
        return x + self.resid_drop(h)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = embed_dim // n_head
        self.attention = MultiHeadedAttention(head_size=head_size, n_head=n_head)
        self.mlp = MLP(expansion=4)
    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x

# ---------- model ----------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, topk=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            last = logits[:, -1, :] / temperature
            if topk is not None and topk > 0:
                k = min(topk, last.size(-1))
                topv, topi = torch.topk(last, k=k, dim=-1)
                probs = topv.softmax(dim=-1)
                pick_local = torch.multinomial(probs, num_samples=1)
                next_id = topi.gather(-1, pick_local)
            else:
                probs = last.softmax(dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx

# ---------- training ----------
print("model is being created...")
model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)


print("training loop is starting...")
for step in range(1, max_iters + 1):
    xb, yb = get_dataset_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step} | train loss {loss.item():.4f}")

# ---------- generation ----------
print("generation is starting...")
prompt = "ROMEO: "
start_ids = torch.tensor([[charToIndexMapping[c] for c in prompt]], dtype=torch.long, device=device)
out_ids = model.generate(start_ids, max_new_tokens=300, temperature=0.9, topk=50)[0].tolist()
print(decode(out_ids))
