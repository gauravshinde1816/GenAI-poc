import math, random
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

# tokenizer: train/load BPE on input.txt
text_path = Path("input.txt")
text = text_path.read_text(encoding="utf-8")

# save tokens
tok_path = Path("bpe_tokenizer.json")

if not tok_path.exists():
    # handle unknown tokens
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=16000,
        min_frequency=2,
        # handle padding unknown, beggining of sentence , end of senetence , end of sentence tokens
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    tok.train(files=[str(text_path)], trainer=trainer)
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

# get BPE encodings
ids = torch.tensor(encode_bpe(text), dtype=torch.long)