from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing


REPO_ID = "HuggingFaceFW/fineweb"
CONFIG  = "sample-10BT" 


data_stream = load_dataset(REPO_ID, CONFIG, split="train", streaming=True)


# device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# GPU hyperparameters
block_size   = 128
embed_dim    = 128
n_head       = 4
n_layer      = 4
batch_size   = 64
learning_rate= 3e-4
max_iters    = 1000
attn_dropout = 0.1
resid_dropout= 0.1
mlp_dropout  = 0.1

TRAIN_DATA_SIZE = 1_000_000
TEST_DATA_SIZE = 50000


train_stream = data_stream.take(TRAIN_DATA_SIZE);
val_stream = data_stream.skip(TRAIN_DATA_SIZE).take(TEST_DATA_SIZE)




# save tokens
tok_path = Path("bpe_tokenizer.json")

if not tok_path.exists():
    # handle unknown tokens
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel()
    tok.decode = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=16000,
        min_frequency=2,
        # handle padding unknown, beggining of sentence , end of senetence , end of sentence tokens
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    tok.train_from_iterator((row['text'] for row in train_stream), trainer=trainer)
    tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <bos> $B <eos>",
        special_tokens=[("<bos>", tok.token_to_id("<bos>")), ("<eos>", tok.token_to_id("<eos>"))]
    )
    tok.save(str(tok_path))

tok = Tokenizer.from_file(str(tok_path))
vocab_size = tok.get_vocab_size()

def encode_bpe(s: str):
    return tok.encode(s).ids

def decode_bpe(ids):
    return tok.decode(ids)



# stream blocks
def stream_blocks(data_stream):

  buffer=[]

  for doc in data_stream:

    ids = tok.encode(doc['text']).ids

    if not ids:
      continue
    buffer.extend(ids)

    while len(buffer) > block_size  + 1:

      x = torch.tensor(buffer[:block_size], dtype=torch.long)
      y = torch.tensor(buffer[1:block_size+1], dtype=torch.long)

      yield x, y

      buffer = buffer[block_size:]


def batcher(block_gen):
    xs, ys = [], []
    for x, y in block_gen:
        xs.append(x)
        ys.append(y)
        if len(xs) == batch_size:
            yield torch.stack(xs).to(device), torch.stack(ys).to(device)
            xs, ys = [], []


train_block_gen = stream_blocks(train_stream)
val_block_gen   = stream_blocks(val_stream)

train_batch_gen = batcher(train_block_gen)
val_batch_gen   = batcher(val_block_gen)



def get_dataset_batch(split):
    if split == "train":
        x, y = next(train_batch_gen)  # raises StopIteration if stream ends
    else:
        x, y = next(val_batch_gen)
    return x, y


# transformer
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.K = nn.Linear(embed_dim, head_size, bias=False)
        self.Q = nn.Linear(embed_dim, head_size, bias=False)
        self.V = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
    def forward(self, x):
        B, T, C = x.shape
        k = self.K(x); q = self.Q(x); v = self.V(x)
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

# model
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

# training
print("model is being created...")
model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

print("training loop is starting...")
for step in range(1, max_iters + 1):
    xb, yb = get_dataset_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 10 == 0:
        with torch.no_grad():
            vx, vy = get_dataset_batch("val")
            _, vloss = model(vx, vy)
        print(f"step {step} | train {loss.item():.4f} | val {vloss.item():.4f}")

# generation (BPE encode/decode)
print("generation is starting...")
prompt = "What data do we have ? "
start_ids = torch.tensor([encode_bpe(prompt)], dtype=torch.long, device=device)
out_ids = model.generate(start_ids, max_new_tokens=300, temperature=0.9, topk=50)[0].tolist()
print(decode_bpe(out_ids))


