"""This is a method that creates a small syntetic-voter-turnout-dataset. Each tow is one (made-up) voter described by three features, all scaled so theyre easy to compare.:

    -age01           #age,scaled to [0,1](age, scaled to [0, 1] (0 ≈ 18, 1 ≈ 80)
    -income01        #income, scaled to [0, 1] (0 ≈ £15k, 1 ≈ £100k)
    -voted2019       #did they vote in 2019? (0 = no, 1 = yes)
"""
import numpy as np



def make_turnout_data(n=2000, seed=0):
    """Synthetic turnout data with a known NONLINEAR signal.
    Features: age01, income01, voted2019  (all in [0,1] or {0,1}).
    Label:    turned out (0/1).  Returns numpy arrays X (n,3), y (n,).
    The age*income interaction makes a single linear model imperfect,
    so a 2-layer net can beat plain logistic regression."""
    rng = np.random.default_rng(seed)
    age01     = rng.uniform(0, 1, n)
    income01  = rng.uniform(0, 1, n)
    voted2019 = rng.integers(0, 2, n).astype(float)
    z = (1.6 * age01 + 1.2 * income01
         + 2.0 * voted2019
         - 3.0 * age01 * income01      # <-- interaction, nonlinearity
         - 1.0)
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p).astype(np.int64)
    X = np.stack([age01, income01, voted2019], axis=1).astype(np.float32)
    return X, y


def train_val_split(X, y, train_frac=0.8, seed=1337):
    """Converts numpy -> torch and performs a train/val split (80/20 by default).

    torch is imported inside this function so that scripts without PyTorch
    (NN-Repaso.py, MLP(no pytorch).py) can still import data.py.

    Returns: X_train, y_train, X_val, y_val  (float32 tensors)
    """
    import torch


    X_t = torch.from_numpy(X)                       # (n, 3) float32
    y_t = torch.from_numpy(y).float().unsqueeze(1)  # (n, 1) float32

    # 2) Train / validation split. We shuffle indices with a fixed seed
    #    (a local generator, so it doesn't touch the global torch seed).
    g = torch.Generator().manual_seed(seed)
    n_total = X_t.shape[0]
    n_train = int(train_frac * n_total)
    perm    = torch.randperm(n_total, generator=g)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_train, y_train = X_t[train_idx], y_t[train_idx]
    X_val,   y_val   = X_t[val_idx],   y_t[val_idx]
    return X_train, y_train, X_val, y_val


FASHION_CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_fashion_mnist(root="data", batch_size=64):
    """FashionMNIST dataset for image recognition (10 classes of clothing).

    Downloads the data the first time (cached in `root` afterwards) and
    converts each image to a tensor of shape (1, 28, 28) with values in [0, 1].

    torch/torchvision are imported inside the function so that scripts
    without PyTorch can still import data.py.

    Returns: train_data, test_data, train_loader, test_loader
      - each loader batch: images (B, 1, 28, 28), labels (B,)
      - labels 0..9 map to FASHION_CLASS_NAMES
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    train_data = datasets.FashionMNIST(root=root, train=True,  download=True, transform=ToTensor())
    test_data  = datasets.FashionMNIST(root=root, train=False, download=True, transform=ToTensor())

    # shuffle=True reshuffles the training set every epoch.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)
    return train_data, test_data, train_loader, test_loader


# Lets you do `from data import train_loader, test_loader` (and train_data,
# test_data). FashionMNIST is only loaded the first time one of these names is
# requested, so scripts that don't use it don't need torch/torchvision.
_FASHION_NAMES = ("train_data", "test_data", "train_loader", "test_loader")
_fashion_cache = None

# Declarations only (no value) so the editor/Pylance knows these names exist.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.utils.data import Dataset, DataLoader
train_data: "Dataset"
test_data: "Dataset"
train_loader: "DataLoader"
test_loader: "DataLoader"


def __getattr__(name):
    global _fashion_cache
    if name in _FASHION_NAMES:
        if _fashion_cache is None:
            _fashion_cache = dict(zip(_FASHION_NAMES, load_fashion_mnist()))
        return _fashion_cache[name]
    raise AttributeError(f"module 'data' has no attribute {name!r}")


X, y = make_turnout_data(n=2000, seed=0)


N_SMALL = 200
Xtr = [[float(v) for v in row] for row in X[:N_SMALL]]
ytr = [float(v) for v in y[:N_SMALL]]


if __name__ == "__main__":
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("first 5 rows of X (age01, income01, voted2019):")
    print(X[:5])
    print("first 5 labels y:", y[:5])
    print(f"fraction who turned out: {y.mean():.3f}")
    print(f"Xtr/ytr (listas para Value): {len(Xtr)} votantes")

    X_train, y_train, X_val, y_val = train_val_split(X, y)
    print("\nX_train:", tuple(X_train.shape), "| y_train:", tuple(y_train.shape))
    print("X_val  :", tuple(X_val.shape),   "| y_val  :", tuple(y_val.shape))
    print("\nNote: age and income are already in [0,1] and voted2019 is 0/1,")
    print("so our features are effectively normalised already (a good-practice box ticked).")
