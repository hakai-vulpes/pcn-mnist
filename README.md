# Predictive Coding Networks on MNIST

A small, clean, **correct** PyTorch implementation of a hierarchical
**Predictive Coding Network (PCN)** trained by **inference learning** — local
Hebbian weight updates with _no backpropagation for credit assignment_. The
headline result: a PCN matches a same-architecture backprop MLP on MNIST using
only local updates, and the same machinery runs as a generative model that
reconstructs and generates digits.

> Autograd is used **only inside the tests**, as a correctness oracle for the
> manually derived gradients. The model never calls `.backward()` to learn.

---

## The idea in one paragraph

Each layer `l` holds a vector of **value neurons** `x[l]` and predicts a
neighbouring layer through a weight matrix. The mismatch between a layer's
actual value and the prediction of it is the **error** `eps`. The network
defines a single scalar **energy**

```
F = sum_i  0.5 * || eps[i] ||^2 ,    eps[i] = x[target] - ( f(x[pred]) @ W[i].T + b[i] )
```

and _everything is gradient descent on F_:

- **Inference** relaxes the free value neurons `x` (fast inner loop), holding
  weights and clamped layers fixed, until F reaches a fixed point.
- **Learning** then takes **one** local weight step on the settled errors:
  `dF/dW[i] = -eps[i].T @ f(x[pred])` — an outer product of local error and
  local activity (Hebbian). No global error signal is propagated.

Both updates are derived by hand and verified against `torch.autograd` to ~1e-8.

## Two prediction directions, one convention

Layers are indexed `0 .. L` with **layer 0 = image** and **layer L = label**
everywhere. What changes per task is _which neighbour predicts which_
(`PCN(direction=...)`):

| direction        | prediction flow           | energy driven by                   | used for                              |
| ---------------- | ------------------------- | ---------------------------------- | ------------------------------------- |
| `discriminative` | image → label (bottom-up) | the **label** error                | supervised classification (Phase 4)   |
| `generative`     | label → image (top-down)  | the **image** reconstruction error | reconstruction & generation (Phase 5) |

This matters: in the generative (top-down) direction the energy is dominated by
the 784-dimensional image reconstruction, so training optimizes _reconstruction_
and classification caps around ~82%. The discriminative direction makes the
**output error** drive learning, so inference learning approximates backprop and
reaches MLP-level accuracy (Whittington & Bogacz 2017; Millidge et al. 2022).
Each model instance uses a single fixed direction — the two are never mixed
mid-computation (see `src/pcn/model.py`).

---

## Results

_MNIST, dims `784 → 256 → 256 → 10`, tanh, Xavier init._

### Supervised classification — PCN vs backprop

Same architecture and (for the apples-to-apples rows) the same L2 loss — only
the **credit-assignment algorithm** differs:

| model                   | credit assignment          | loss          | test accuracy |
| ----------------------- | -------------------------- | ------------- | ------------- |
| MLP baseline            | backprop                   | cross-entropy | **97.76%**    |
| MLP (same loss)         | backprop                   | L2 energy     | **97.82%**    |
| PCN — standard PC       | inference learning (local) | L2 energy     | **96.25%**    |
| PCN — incremental (iPC) | inference learning (local) | L2 energy     | **97.69%**    |

The decisive comparison is **same loss, different algorithm** (rows 2–4).
Backprop reaches 97.82%. _Standard_ PC — relax to equilibrium, then one weight
update — reaches 96.25%: at the fixed point the hidden layers "explain away" the
output error, so its weight update only ~0.67-cosine-approximates backprop
(measured directly; see the analysis in the git history). **Incremental PC** (a
weight update every inference step; Salvatori et al. 2022) follows the
relaxation trajectory, approximates backprop far more closely, and closes the
gap to MLP-level accuracy — all with purely local Hebbian updates and no
backprop.

![training curves](outputs/figures/training_curves.png)

The iPC model's per-class test confusion matrix (strong diagonal, ~97.7%):

![confusion matrix](outputs/figures/confusion_matrix.png)

### Inference settles the energy

Relaxing the value neurons drives F monotonically to a stable fixed point — the
"settling" at the heart of predictive coding.

![energy descent](outputs/figures/energy_descent.png)
![per-layer error norms](outputs/figures/layer_error_norms.png)

### Generative & unsupervised mode

The same energy machinery runs **top-down** (`direction="generative"`), in two
variants that together cover Phase 5.

**Supervised-generative** (train with image _and_ label clamped). It
reconstructs held-out digits crisply, and generates **class-conditioned**
samples — clamp a one-hot at the label end, free the image end, and let the
network "dream" the digit. Deterministic top-down generation is blurry (a known
property of generative PCNs), but the samples are genuinely class-conditioned:
**80%** are classified as their target digit by the discriminative classifier
above.

![reconstructions](outputs/figures/reconstructions.png)
![class-conditioned samples](outputs/figures/generations.png)

**Unsupervised** (train with image _only_ clamped; the top latent is free). With
a tight 16-d top latent, the only way to reconstruct through the bottleneck is
to encode the digit's generative factors — which correlate with class. A linear
probe on the **settled 16-d top latent** reaches **81.2%**, versus **68.4%** for
the _same untrained network's_ latent: a wide margin (a 40% relative error
reduction) showing the code captures class structure **without ever seeing
labels**. A t-SNE of the latent shows clear per-class clusters.

| metric                                       | value     |
| -------------------------------------------- | --------- |
| test reconstruction MSE (supervised-gen)     | **0.058** |
| generation class-match (discriminative)      | **80%**   |
| test reconstruction MSE (unsupervised, 16-d) | **0.053** |
| linear probe — trained 16-d latent           | **81.2%** |
| linear probe — random 16-d latent            | **68.4%** |

![t-SNE of unsupervised latents](outputs/figures/tsne_latents.png)

The model's first-layer receptive fields (columns of `W[0]`):

![receptive fields](outputs/figures/receptive_fields.png)

---

## Correctness tests

PCNs are notoriously easy to implement subtly wrong (an index/transpose slip
silently degrades accuracy), so correctness is tested before any training claim:

- **`tests/test_gradients.py`** — the central oracle. Builds F with autograd on
  a random batch and asserts the _manual_ `dF/dx`, `dF/dW`, `dF/db` match
  `torch.autograd.grad` to ~1e-8, for **both** prediction directions and for
  tanh and ReLU.
- **`tests/test_energy.py`** — during inference F is non-increasing and settles
  to a plateau.
- **`tests/test_overfit.py`** — the supervised PCN reaches ~100% on 100 examples
  (end-to-end check that inference + learning drive the model toward the data).

```bash
PYTHONPATH= uv run pytest -q
```

---

## Repository layout

```
src/pcn/
  functional.py   activations f, f', and the energy F
  model.py        the PCN: predict / grad_x / grad_w / infer / learn (both directions)
  data.py         MNIST load, standardize, one-hot targets, loaders
  train.py        supervised & generative training epochs (inference-then-learn)
  eval.py         classification, reconstruction, generation, linear probe
  baseline.py     same-architecture backprop MLP
  viz.py          all plotting helpers
  utils.py        seeding, device, config, CSV logging
configs/          supervised, supervised_ipc, generative, unsupervised YAML
experiments/      run_supervised, run_baseline, run_generative,
                  run_unsupervised, make_figures
tests/            the three correctness suites
```

## Reproducing

```bash
# 1. correctness
PYTHONPATH= uv run pytest -q

# 2. backprop baseline + standard-PC and incremental-PC classifiers
PYTHONPATH= uv run python experiments/run_baseline.py    --config configs/supervised.yaml --loss mse
PYTHONPATH= uv run python experiments/run_supervised.py  --config configs/supervised.yaml
PYTHONPATH= uv run python experiments/run_supervised.py  --config configs/supervised_ipc.yaml  # headline, >=97%

# 3a. supervised-generative: reconstruction + class-conditioned generation
PYTHONPATH= uv run python experiments/run_generative.py  --config configs/generative.yaml
# 3b. unsupervised: 16-d latent, linear probe vs random, t-SNE
PYTHONPATH= uv run python experiments/run_unsupervised.py --config configs/unsupervised.yaml

# 4. figures
PYTHONPATH= uv run python experiments/make_figures.py
```

> The `PYTHONPATH=` prefix clears a leaked ROS entry on this machine; drop it if
> your environment is clean.

## References

- Bogacz (2017), _A tutorial on the free-energy framework for modelling
  perception and learning._
- Whittington & Bogacz (2017), _An approximation of the error backpropagation
  algorithm in a predictive coding network with local Hebbian synaptic
  plasticity._
- Millidge, Tschantz, Buckley (2022), _Predictive coding approximates backprop
  along arbitrary computation graphs._
- Salvatori et al. (2022), _Incremental predictive coding._
