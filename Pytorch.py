
# ======================== LAB 5 =====================================================
# === Run me first ===
# Colab already has these installed. If you run locally and something is missing:
#   pip install torch numpy matplotlib scikit-learn
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from data import X, y   # make_turnout_data(n=2000, seed=0), definido en data.py
print("X shape:", X.shape, "| y shape:", y.shape)
print("first 5 rows of X (age01, income01, voted2019):")
print(X[:5])
print("first 5 labels y:", y[:5])
print(f"fraction who turned out: {y.mean():.3f}")


X_t = torch.from_numpy(X)                       # (2000, 3) float32
y_t = torch.from_numpy(y).float().unsqueeze(1)  # (2000, 1) float32

# 2) Train / validation split (80 / 20). We shuffle indices with a fixed seed.
torch.manual_seed(1337)
n_total = X_t.shape[0]
n_train = int(0.8 * n_total)
perm    = torch.randperm(n_total)
train_idx, val_idx = perm[:n_train], perm[n_train:]

X_train, y_train = X_t[train_idx], y_t[train_idx]
X_val,   y_val   = X_t[val_idx],   y_t[val_idx]

print("X_train:", tuple(X_train.shape), "| y_train:", tuple(y_train.shape))
print("X_val  :", tuple(X_val.shape),   "| y_val  :", tuple(y_val.shape))
print("\nNote: age and income are already in [0,1] and voted2019 is 0/1,")
print("so our features are effectively normalised already (a good-practice box ticked).")

## ---------------------------- nn.Module ------------------------------------------------

# (a) The nn.Module form. This matches the TurnoutModel from Lecture 5.
class TurnoutNet(nn.Module):
    def __init__(self, in_features=3, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)   # 3 -> 16
        self.fc2 = nn.Linear(hidden, 1)             # 16 -> 1

    def forward(self, x):
        h = torch.relu(self.fc1(x))   # hidden layer + ReLU activation
        # TODO: return the output layer (fc2) applied to h. One line.
        #       NOTE: we do NOT apply a sigmoid here -- the loss will (see Section 3).
        return self.fc2(h)


torch.manual_seed(1337)
print(TurnoutNet())

## ---------------------------- Sequential model ----------------------------------------------------

torch.manual_seed(1337)
model_seq = nn.Sequential(
    nn.Linear(3, 16),   # 3 inputs -> 16 hidden
    nn.ReLU(),          # activation is now an explicit LAYER, not a function call
    nn.Linear(16, 1),   # 16 hidden -> 1 output (a logit)
)
print(model_seq)


##---------------------------- Model Train ----------------------------------------------------------


loss_fn   = nn.BCEWithLogitsLoss()                       # sigmoid + BCE, fused & stable
optimizer = torch.optim.Adam(model_seq.parameters(), lr=1e-2)
print("loss:", loss_fn)
print("optimiser:", optimizer.__class__.__name__, "| lr =", optimizer.param_groups[0]["lr"])

def train(model, loss_fn, optimizer, epochs=2000):
    history = {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()

        model.zero_grad()                                # TODO 1: clear the old gradients
        logits = model(X_train)                       # TODO 2: forward pass of model on X_train
        loss   = loss_fn(logits, y_train)   # (given) compute the loss
        loss.backward()                     # TODO 3: backpropagate (fill in .grad)
        optimizer.step()                                # TODO 4: one optimiser step (update weights)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val)
        history["train"].append(loss.item())
        history["val"].append(val_loss.item())
    return history

history = train(model_seq, loss_fn, optimizer, epochs=2000)
print("done. final val loss:", round(history["val"][-1], 4))


## ============================ LAB 6 ================================================


##-------------------------- CNN ------------------------------------------------------

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # -> (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))     # -> (B, 32,  7,  7)
        x = torch.flatten(x, 1)                  # -> (B, 1568)
        x = F.relu(self.fc1(x))                  # -> (B, 128)
        x = self.fc2(x)                          # -> (B, 10) logits
        return x


## ------------------------------ CNN ACCURACY ---------------------------------------

@torch.no_grad()
def accuracy(model, loader):
    model.eval()                          # evaluation mode
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1)      # the class with the highest score
        correct += (preds == y).sum().item()
        total   += y.size(0)
    model.train()                         # back to training mode
    return correct / total


## --------------------------- Training -------------------------------------------------

def train(model, train_loader, epochs=3, lr=1e-3):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()                # 1. clear gradients
            logits = model(X)                    # 2. forward
            loss = loss_fn(logits, y)            # 3. loss
            loss.backward()                      # 4. backward
            optimizer.step()                     # 5. update
            loss_history.append(loss.item())
        print(f"epoch {epoch+1}/{epochs} | last loss {loss.item():.4f} "
              f"| train acc {accuracy(model, train_loader):.3f} "
              f"| test acc {accuracy(model, test_loader):.3f}")
    return loss_history

## ============================ LAB 7 ====================================================

## ----------------------------- Autoencoder --------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden=HIDDEN):
        super().__init__()
        # Encoder: 784 -> hidden -> latent_dim   (the squeeze)
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        # Decoder: latent_dim -> hidden -> 784    (the expand)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 784), nn.Sigmoid(),   # sigmoid -> pixels in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)        # squeeze to the code
        xhat = self.decoder(z)     # expand back to an image
        return xhat, z

ae = Autoencoder().to(device)
print(ae)

## ------------------------------------------ Train AE -----------------------------------------

def train_ae(model, loader, epochs=10, lr=LR):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        running = 0.0
        for images, _ in loader:                       # labels ignored: no labels needed!
            x = images.view(images.size(0), -1).to(device)   # flatten to (batch, 784)
            xhat, z = model(x)                                # 2. forward
            loss = F.mse_loss(xhat, x, reduction="sum") / x.size(0)   # 3. loss (MSE per image)
            opt.zero_grad()                                  # 5a. clear old grads
            loss.backward()                                  # 4. backward
            opt.step()                                       # 5b. update weights
            running += loss.item() * x.size(0)
        print(f"epoch {epoch+1:2d}/{epochs} | recon loss {running/len(loader.dataset):.2f}")
    return model

# Train it. (Bump epochs up later for sharper reconstructions.)
train_ae(ae, train_loader, epochs=10)

## -----------------------------------------  Decoder AE -----------------------------------------
ae.eval()
with torch.no_grad():
    random_codes = torch.randn(16, LATENT_DIM).to(device)   # z ~ N(0, I)
    images_out = ae.decoder(random_codes)                   # decoder ONLY

show_images(images_out, title="Decoding random codes from a plain AE")

## ---------------------------------------- VAE ------------------------------------------------

class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden=HIDDEN):
        super().__init__()
        self.encoder   = nn.Sequential(nn.Linear(784, hidden), nn.ReLU())
        self.fc_mu     = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps          # the reparameterisation trick
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar, z

vae = VAE().to(device)
print(vae)

## ------------------------------------------- VAE loss ------------------------------------------------

BETA = 1.0

def vae_loss(xhat, x, mu, logvar, beta=BETA):
    recon = F.mse_loss(xhat, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kld, recon, kld

## ------------------------------------------- VAE training loops ---------------------------------------

def train_vae(model, loader, epochs=20, lr=LR):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        tot = tot_recon = tot_kld = 0.0
        for images, _ in loader:
            x = images.view(images.size(0), -1).to(device)      # flatten
            xhat, mu, logvar, z = model(x)                       # forward
            loss, recon, kld = vae_loss(xhat, x, mu, logvar)    # loss = recon + KL
            opt.zero_grad()
            loss.backward()
            opt.step()
            n = x.size(0)
            tot += loss.item() * n; tot_recon += recon.item() * n; tot_kld += kld.item() * n
        N = len(loader.dataset)
        print(f"epoch {epoch+1:2d}/{epochs} | total {tot/N:7.2f} | recon {tot_recon/N:7.2f} | KL {tot_kld/N:6.2f}")
    return model

train_vae(vae, train_loader, epochs=20)


## -------------------------------- sampling digits -------------------------------------------------------

vae.eval()
with torch.no_grad():
    z = torch.randn(64, LATENT_DIM).to(device)   # sample from N(0, I)
    new_digits = vae.decode(z)                   # decoder turns codes into images

show_images(new_digits, title="Brand-new digits sampled from N(0, I)", n=64, ncols=8)

## ------------------------------------- latent interpolation -------------------------------------------------

vae.eval()
with torch.no_grad():
    imgs, labels = next(iter(test_loader))
    x = imgs.view(imgs.size(0), -1).to(device)
    mu, logvar = vae.encode(x)            # use the means as the two endpoints

    z1 = mu[0:1]                          # first image's code
    z2 = mu[1:2]                          # second image's code

    steps = 10
    ts = torch.linspace(0, 1, steps).view(-1, 1).to(device)
    z_path = (1 - ts) * z1 + ts * z2     # straight line from z1 to z2
    morph = vae.decode(z_path)

print(f"morphing from a '{labels[0].item()}' to a '{labels[1].item()}'")
show_images(morph, title="Interpolation in latent space", n=steps, ncols=steps)


## ========================================= LAB 8 =================================================

## ----------------------------------- get_batch -----------------------------------------------------

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

## -------------------------------------- biagram model ----------------------------------------------

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)        # (B,T,vocab)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


    ## ----------------------------- top_next ----------------------------------------------------


emb = model.token_embedding_table.weight
print("embedding / logit table shape:", tuple(emb.shape))

def top_next(ch, k=5):
    row = emb[stoi[ch]]
    probs = F.softmax(row, dim=-1)
    vals, idx = torch.topk(probs, k)
    return [(itos[i.item()], round(v.item(), 3)) for v, i in zip(vals, idx)]

for ch in ['t', 'h', 'o', 'a', ' ']:
    print(f"after {ch!r:4} ->", top_next(ch))


