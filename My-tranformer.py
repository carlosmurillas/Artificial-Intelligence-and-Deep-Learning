import torch
import torch.nn as nn
from torch.nn import functional as F

from data import build_char_pipeline, text

# --- device handling: use the GPU if Colab gave us one ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():   # Apple Silicon, if running locally
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)

# Reproducibility: the same seed every run (matches Labs 8 and 9).
torch.manual_seed(1337)

## ----------------------------  Head ------------------------------------------------
block_size = 32    # context length: how many characters the model sees at once
batch_size = 32    # how many sequences we train on in parallel
n_embd     = 64    # embedding width (the "d" in the lecture)
n_head     = 4     # number of attention heads per block
n_layer    = 4     # number of transformer blocks (the depth of the stack)
dropout    = 0.0   # regularisation; 0.0 keeps this short run simple
learning_rate = 1e-3
max_iters     = 2000   # training steps
eval_interval = 250    # print train/val loss every N steps

class Head(nn.Module):
    """One head of causal self-attention -- the worked example as a module."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # a fixed lower-triangular matrix used for the causal mask
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                            # (B, T, head_size)
        q = self.query(x)                                          # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5        # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)                                          # (B, T, head_size)
        return wei @ v                                             # (B, T, head_size)

## ----------------------------  MultiHeadAttention ------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Several heads of self-attention in parallel, concatenated and projected."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # concat over the head dim
        return self.dropout(self.proj(out))                   # project back to n_embd

## ----------------------------  FeedForward ------------------------------------------------

class FeedForward(nn.Module):
    """Per-token MLP: n_embd -> 4*n_embd -> n_embd (same size in and out,
    so it fits the residual connection x + ffwd(x))."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

## ----------------------------  Attention Block ------------------------------------------------

class Block(nn.Module):
    """A transformer block: attention + feed-forward, each with a
    residual connection and a pre-LayerNorm."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # residual around attention   (pre-norm)
        x = x + self.ffwd(self.ln2(x))    # residual around feed-forward (pre-norm)
        return x

## ----------------------------  GPT ------------------------------------------------

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                              # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                                  # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                               # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]            # crop context to the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature    # focus on the last step
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

## ----------------------------  data pipeline + model ------------------------------------------------
P = build_char_pipeline(text, block_size=block_size, batch_size=batch_size, device=device)
vocab_size = P["vocab_size"]
encode, decode = P["encode"], P["decode"]
gpt = GPTLanguageModel(vocab_size).to(device)
print(f"{sum(p.numel() for p in gpt.parameters()) / 1e6:.2f}M parameters")

## ----------------------------  training loop ------------------------------------------------
def train_model(model, get_batch, estimate_loss, max_iters=3000, eval_interval=500,
                lr=1e-3, device="cpu"):
    """Generic AdamW training loop, reused by every model in Labs 8-11."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

gpt = train_model(gpt, P["get_batch"], P["estimate_loss"], max_iters=max_iters,
                  eval_interval=eval_interval, lr=learning_rate, device=device)

## ----------------------------  Sampling for trained Model ------------------------------------------------
gpt.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(gpt.generate(context, max_new_tokens=500)[0].tolist()))