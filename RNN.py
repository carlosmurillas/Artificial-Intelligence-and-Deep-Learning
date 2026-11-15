import math, time
import torch
import torch.nn as nn
from torch.nn import functional as F

from data import build_char_pipeline, text

torch.manual_seed(1337)          

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('PyTorch', torch.__version__, '| device:', device)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU — Runtime > Change runtime type > GPU (the lab still runs on CPU, just slower).')


## ----------------------------  RNNCell ------------------------------------------------
class RNNCell(nn.Module):
    """One vanilla-RNN step on a batch of vectors."""
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.W_xh = nn.Linear(n_in, n_hidden, bias=False)
        self.W_hh = nn.Linear(n_hidden, n_hidden, bias=True)

    def forward(self, x_t, h_prev):
        # h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h); b_h is inside W_hh
        h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))
        return h_t

print('RNNCell solution loaded.')
## ----------------------------  looping the cell through time ------------------------------------------------
torch.manual_seed(1337)
cell = RNNCell(n_in=10, n_hidden=8)

B, T = 4, 6
xseq = torch.randn(B, T, 10)
h = torch.zeros(B, 8)

for t in range(T):
    h = cell(xseq[:, t, :], h)          # the new state feeds back in

print('final hidden state shape:', tuple(h.shape))   # (4, 8)
assert h.shape == (B, 8)

## ----------------------------  RNNLanguageModel ------------------------------------------------
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_hidden=128):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.rnn = nn.GRU(n_embd, n_hidden, batch_first=True)
        self.lm_head = nn.Linear(n_hidden, vocab_size)

    def forward(self, idx, targets=None):
        emb = self.token_embedding_table(idx)
        out, _ = self.rnn(emb)
        logits = self.lm_head(out)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print('RNNLanguageModel (GRU solution) defined.')

## ----------------------------  data pipeline + model ------------------------------------------------
P = build_char_pipeline(text, block_size=32, batch_size=64, device=device)

gru = RNNLanguageModel(P['vocab_size']).to(device)

gru_losses    = P['estimate_loss'](gru)
print(f"GRU    | train {gru_losses['train']:.4f} | val {gru_losses['val']:.4f}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)   # start token = id 0

print()
print(P['decode'](gru.generate(context, 400)[0].tolist()))