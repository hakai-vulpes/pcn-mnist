"""A trained PCN must be able to fit data.

On a tiny fixed subset (100 examples) the supervised PCN should reach ~100%
train accuracy within a modest number of epochs. This is an end-to-end smoke
test that inference + learning together actually drive the model toward the
data, catching sign errors that the gradient check (a single-step oracle) can
miss.
"""

import torch

from pcn.data import load_mnist, subset_loader
from pcn.eval import classify_accuracy
from pcn.model import PCN
from pcn.train import train_supervised_epoch


def test_overfit_100_examples():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_mnist(download=True)
    loader = subset_loader(data, n=100, batch_size=100, seed=0)

    # Inference must converge well before each weight update (enough steps T),
    # and xavier init keeps activations in a usable range -- both documented PCN
    # pitfalls. With those the network memorizes 100 examples.
    model = PCN(
        dims=[784, 256, 256, 10],
        activation="tanh",
        init="xavier",
        gamma=0.1,
        T=150,
        init_std=0.05,
        seed=0,
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(300):
        train_supervised_epoch(model, loader, optimizer=opt, device=device)

    acc = classify_accuracy(model, loader, device=device, T=300)
    assert acc >= 0.98, f"PCN failed to overfit 100 examples: train acc {acc:.3f}"
