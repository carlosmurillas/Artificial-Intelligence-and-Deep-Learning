import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from data import X, y, train_val_split   # make_turnout_data(n=2000, seed=0) + split 80/20

## ---------------------------- Forward pass ------------------------------------------------
class TurnoutNet(nn.Module):
    """Feed-forward network with one hidden layer for binary classification.

    Architecture: in_features (3 by default) -> hidden (16, ReLU) -> 1 output.
    The output is a raw logit; use BCEWithLogitsLoss for training or
    torch.sigmoid to get a probability.
    """
    def __init__(self, in_features=3, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)            # output layer -> a single logit per row

# Everything below only runs when this file is executed directly
# (py feed_forward_net.py), NOT when TurnoutNet is imported.
if __name__ == "__main__":

    torch.manual_seed(1337)

    print("torch       :", torch.__version__)
    print("numpy       :", np.__version__)
    print("CUDA (GPU)? :", torch.cuda.is_available(), "  <- False is totally fine for this lab")
    print("\nSetup complete. This lab's data is tiny, so CPU runs in seconds.")

    # Data: train / validation tensors from data.py
    X_train, y_train, X_val, y_val = train_val_split(X, y)

    print("\nX_train:", tuple(X_train.shape), "| y_train:", tuple(y_train.shape))
    print("X_val  :", tuple(X_val.shape),   "| y_val  :", tuple(y_val.shape))

    torch.manual_seed(1337)
    print(TurnoutNet())

    ## ---------------------------- Sequential model ------------------------------------------------
    # This is exactly the same model as above, but defined with nn.Sequential.
    torch.manual_seed(1337)
    model_seq = nn.Sequential(
        nn.Linear(3, 16),   # 3 inputs -> 16 hidden units
        nn.ReLU(),          # activation is now an explicit LAYER, not a function call
        nn.Linear(16, 1),   # 16 hidden units -> 1 output (a logit)
    )
    print(model_seq)

    ## ---------------------------- Training Loop ------------------------------------------------
    # Self-contained: re-creates a fresh model/loss/optimiser so this cell reproduces
    # the intended result on its own, regardless of what ran before it.
    torch.manual_seed(1337)
    model     = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer, epochs=2000):
        history = {"train": [], "val": []}

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()               # 1. clear the old gradients
            logits = model(X_train)             # 2. forward pass
            loss   = loss_fn(logits, y_train)   #    compute the loss
            loss.backward()                     # 3. backpropagate
            optimizer.step()                    # 4. update the weights

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val)
            history["train"].append(loss.item())
            history["val"].append(val_loss.item())
        return history

    history = train(model, loss_fn, optimizer, epochs=2000)
    print("done. final val loss:", round(history["val"][-1], 4)) #returns the final 