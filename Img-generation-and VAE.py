import torch
import torch.nn as nn
import torch.nn.functional as F
# MNIST (digits 0..9) from data.py: train 60000 imgs (batch 128), test 10000 imgs (batch 256), each (1, 28, 28) in [0, 1]
from data import train_dataMN, test_dataMN, train_loaderMN, test_loaderMN
import matplotlib.pyplot as plt



# Reproducibility: same seed -> same shuffles, same initial weights for everyone.
torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch version:", torch.__version__)
print("Using device :", device)

# Hyperparameters
LATENT_DIM = 2      # size of the code z (2 -> easy to visualise the latent space)
HIDDEN     = 400    # hidden layer width
LR         = 1e-3   # Adam learning rate



## ---------------------------- The autoencoder ------------------------------------------------

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

## ---------------------------- Train autoencoder ------------------------------------------------
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
train_ae(ae, train_loaderMN, epochs=10)
## ---------------------------- Decoding random codes from the AE ------------------------------------------------


def show_images(images, title="", n=16, ncols=8):
    """Show up to n images. `images` can be (N, 784) flat vectors or (N, 1, 28, 28)."""
    images = images.detach().cpu().view(-1, 28, 28)   # collapse to (N, 28, 28)
    n = min(n, images.shape[0])
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(ncols, nrows + 0.5))
    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Sanity check: look at a few real training digits.
example_imgs, example_labels = next(iter(train_loaderMN))
show_images(example_imgs, title="A few real MNIST digits")


## ---------------------------- Decoding random codes from the AE ------------------------------------------------


ae.eval()
with torch.no_grad():
    random_codes = torch.randn(16, LATENT_DIM).to(device)   # z ~ N(0, I)
    images_out = ae.decoder(random_codes)                   # decoder ONLY

show_images(images_out, title="Decoding random codes from a plain AE")

## ---------------------------- VAE Class------------------------------------------------

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

## ---------------------------- VAE loos ------------------------------------------------


BETA = 1.0

def vae_loss(xhat, x, mu, logvar, beta=BETA):
    recon = F.mse_loss(xhat, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kld, recon, kld

## ---------------------------- Train autoencoder ------------------------------------------------

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

train_vae(vae, train_loaderMN, epochs=20)


## ---------------------------- Ssampling new digits ------------------------------------------------
vae.eval()
with torch.no_grad():
    z = torch.randn(64, LATENT_DIM).to(device)   # sample from N(0, I)
    new_digits = vae.decode(z)                   # decoder turns codes into images

show_images(new_digits, title="Brand-new digits sampled from N(0, I)", n=64, ncols=8)

## ---------------------------- Latent interpolation  ------------------------------------------------
vae.eval()
with torch.no_grad():
    imgs, labels = next(iter(test_loaderMN))
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