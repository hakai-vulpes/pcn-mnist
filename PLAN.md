# Predictive Coding Networks on MNIST — Build Plan

A spec for an autonomous coding agent. The goal is a small, clean, **correct**
PyTorch implementation of a hierarchical predictive coding network (PCN) trained
by inference learning (no backpropagation for credit assignment), evaluated on
MNIST in two modes: supervised classification and unsupervised/generative
representation learning. The headline result to reproduce is that a PCN matches
a same-architecture backprop MLP on MNIST (~97–98% test accuracy), while using
only local Hebbian weight updates.

---

## 0. Scope and deliverables

Build a repository that:

1. Implements a flexible PCN supporting arbitrary clamping of any layer.
2. Trains it on MNIST in **supervised** mode (clamp image + label) and
   reproduces near-MLP accuracy.
3. Trains it in **unsupervised** mode (clamp image only) and demonstrates
   learned representations + generation.
4. Includes a same-architecture **backprop MLP baseline** for comparison.
5. Has **correctness tests** (this matters more than usual — PCNs are easy to
   implement subtly wrong).
6. Produces plots and a short results writeup.

Non-goals: spiking neurons, GPU kernel optimization, distributed training. Keep
it readable.

---

## 1. The algorithm the agent MUST internalize before coding

This is the single most important section. The math below is the
discriminative/generative hierarchical PCN from Bogacz (2017) and Whittington &
Bogacz (2017). Implement the update rules **manually** — do not use
`loss.backward()` to assign credit. (Autograd is allowed only inside tests, as a
correctness oracle. See §3.)

### 1.1 Network structure

Layers indexed `l = 0 .. L`. Layer 0 is the data/image end; layer L is the
abstract/label end. Each layer has a vector of **value neurons** `x[l]` (shape
`[B, dim_l]` for batch size B). Each layer `0 .. L-1` also has **error neurons**
`eps[l]`.

Top-down prediction of layer `l` from the layer above:

```
mu[l]  = W[l] @ f(x[l+1]) + b[l]        for l = 0 .. L-1
eps[l] = x[l] - mu[l]                    # prediction error neuron activity
```

`f` is a nonlinearity (start with `tanh`). `W[l]` maps layer `l+1` -> layer `l`.

### 1.2 Energy (the thing that gets minimized)

```
F = sum_{l=0}^{L-1} 0.5 * ||eps[l]||^2     # sum of squared prediction errors
```

(Optionally add per-layer precision/variance weights later; start with identity
precisions.)

### 1.3 Inference phase — "pin and let it settle"

Clamp whichever layers are observed, then relax the **unclamped** value neurons
by gradient descent on F. For a hidden layer `l`:

```
dF/dx[l] = eps[l] - f'(x[l]) * (W[l-1].T @ eps[l-1])
x[l] <- x[l] - gamma * dF/dx[l]
```

The two terms are exactly the excitatory/inhibitory interplay from the video:

- `eps[l]` → corrective pull toward the top-down prediction (the node is being
  explained from above).
- `f'(x[l]) * (W[l-1].T @ eps[l-1])` → bottom-up drive from the unexplained
  error below.

Iterate for `T` steps (or until `||dF/dx||` plateaus). This is the "settle until
stable" step.

### 1.4 Learning phase — local Hebbian weight update

After inference converges, update weights once (standard PC) using the settled
errors:

```
W[l] <- W[l] + alpha * (eps[l].T @ f(x[l+1]))   # outer product, averaged over batch
b[l] <- b[l] + alpha * mean(eps[l], over batch)
```

Local: each synapse changes from its post-synaptic error and pre-synaptic
activity only.

### 1.5 Clamping conventions (how supervised vs unsupervised differ — same machinery)

| Mode                       | x[0] (image end) | x[L] (label end)       | Read out                        |
| -------------------------- | ---------------- | ---------------------- | ------------------------------- |
| Supervised train           | clamp = image    | clamp = one-hot label  | (learn W)                       |
| Supervised test / classify | clamp = image    | **free**               | argmax(x[L])                    |
| Generate a digit           | **free**         | clamp = one-hot class  | x[0] = "dreamed" image          |
| Unsupervised train         | clamp = image    | free latent (no label) | (learn W)                       |
| Representation probe       | clamp = image    | free                   | settled hidden x[l] as features |

### 1.6 Reference pseudocode (one training step, supervised)

```
for (image, label) in batch:
    x[0] = image (clamped); x[L] = onehot(label) (clamped)
    init hidden x[1..L-1]  (e.g. feedforward pass mu, or zeros)
    repeat T times:                       # inference / relaxation
        compute mu[l], eps[l] for all l
        for hidden l: x[l] -= gamma * (eps[l] - f'(x[l]) * (W[l-1].T @ eps[l-1]))
    for all l: W[l] += alpha * (eps[l].T @ f(x[l+1])) / B   # learning
               b[l] += alpha * mean(eps[l])
```

Everything is vectorized over the batch dimension B. No per-example Python loop.

---

## 2. Tech stack and environment

- **Python 3.11+**, **PyTorch** (CUDA build — target the homelab NVIDIA GPU; CPU
  also fine since MNIST is tiny).
- Env via **`uv`** (preferred) or `venv`. Provide a `pyproject.toml`.
- Core deps: `torch`, `torchvision` (MNIST loader), `numpy`, `matplotlib`,
  `tqdm`.
- Dev deps: `pytest`, `ruff`.
- Optional: `tensorboard` or CSV logging for curves; `scikit-learn` for the
  linear-probe / t-SNE.
- Determinism: seed everything; expose `--seed`.

---

## 3. Repository structure

```
pcn-mnist/
├── pyproject.toml
├── README.md
├── configs/
│   ├── supervised.yaml
│   └── unsupervised.yaml
├── src/pcn/
│   ├── __init__.py
│   ├── model.py        # PCN class: predict(), infer(), learn(), forward-init
│   ├── functional.py   # nonlinearities f and f', energy F
│   ├── data.py         # MNIST loaders, normalization, one-hot
│   ├── train.py        # training loops (supervised + unsupervised)
│   ├── baseline.py     # same-arch MLP trained with backprop (Adam/SGD)
│   ├── eval.py         # accuracy, reconstruction error, linear probe
│   └── viz.py          # energy-descent, reconstructions, generations, confusion matrix, receptive fields
├── experiments/
│   ├── run_supervised.py
│   ├── run_unsupervised.py
│   └── run_baseline.py
├── tests/
│   ├── test_gradients.py
│   ├── test_energy.py
│   └── test_overfit.py
└── notebooks/
    └── exploration.ipynb
```

---

## 4. Build phases (milestones, each independently verifiable)

**Phase 1 — Scaffolding & data.** Project skeleton, `pyproject.toml`, MNIST
download/normalize (`x in [0,1]` or standardized), train/val/test split, one-hot
labels, batching. Verify a batch loads with correct shapes.

**Phase 2 — Core PCN.** Implement `model.py`: parameter init (small random W,
zero b), the prediction pass, the inference loop, and the Hebbian weight update.
Implement F and f/f' in `functional.py`. Initialize hidden states with a
feedforward pass through `mu` for faster convergence.

**Phase 3 — Correctness tests (do this before any training claims).**

- `test_gradients.py`: build F with autograd on a random batch; assert the
  **manual** `dF/dx` and `dF/dW` match `torch.autograd.grad` to ~1e-5. This
  catches index/transpose errors.
- `test_energy.py`: during the inference phase F is **non-increasing** step to
  step (for small gamma).
- `test_overfit.py`: on 100 training examples, supervised PCN reaches ~100%
  train accuracy.

**Phase 4 — Supervised experiment + baseline.** Train full MNIST supervised.
Train the MLP baseline (same dims, same nonlinearity) with backprop. Report both
test accuracies; target PCN within ~0.5–1% of backprop. Plot training/val curves
for both on one axis.

**Phase 5 — Unsupervised / generative.** Train generative PCN (clamp image only,
free top latent). Then: (a) reconstruct test images and report MSE; (b) generate
by clamping a class one-hot at the top and relaxing with x[0] free; (c) freeze
the network and fit a **linear probe** on the settled top-layer representations
to measure representation quality (accuracy); (d) t-SNE/UMAP of the latents
colored by class.

**Phase 6 — Visualization & writeup.** Energy-vs-inference-step curve (show the
"settling"), per-layer error norms over inference, learned first-layer receptive
fields, confusion matrix, generated-digit grid, reconstruction grid. Write
`README.md` results section comparing PCN to backprop and showing both modes.

**Phase 7 — Stretch goals (optional).**

- **Incremental PC (iPC):** update weights every inference step instead of only
  at convergence; compare speed/accuracy (Salvatori et al. 2022/2024).
- Learnable per-layer **precisions** (variances) and observe effect on training
  stability.
- Ablations: number of inference steps T, depth, nonlinearity (tanh vs ReLU),
  init scheme.
- Convolutional PCN layers as a harder variant.

---

## 5. Starting hyperparameters

- Architecture: `784 -> 256 -> 256 -> 10`, nonlinearity `tanh`.
- Inference: `gamma = 0.1`, `T = 20` steps (raise to 50–100 if energy hasn't
  plateaued).
- Learning: `alpha = 1e-3` (or wrap weights in an optimizer like Adam stepping
  on the manual gradient).
- Batch size: 128. Epochs: 20–50. Weight init: `N(0, 0.05)` or Xavier.
- Clamp the label as a one-hot vector; if using tanh outputs, consider scaling
  targets to the nonlinearity's range.

These are starting points — tune in Phase 4.

---

## 6. Success criteria

- Supervised PCN test accuracy ≥ 97% and within ~1% of the backprop MLP
  baseline.
- All three correctness tests pass.
- Inference demonstrably reduces F to a stable fixed point (plot shows it).
- Unsupervised mode produces recognizable reconstructions and class-conditioned
  generations, and the linear probe on learned representations beats a
  random-features baseline by a wide margin.

---

## 7. Pitfalls to get right (PCNs are deceptively easy to break)

- **Do not use backprop for credit assignment.** Implement the manual updates;
  autograd appears only in tests as an oracle. Using `.backward()` to train
  defeats the entire point.
- **Index/transpose bugs** in `W[l-1].T @ eps[l-1]` are the #1 source of silent
  failure — the gradient-check test exists specifically to catch these.
- **Inference must converge before the weight update** in standard PC; too few
  steps T → bad gradients.
- **Numerical stability:** keep gamma small enough that F decreases
  monotonically; clamp/standardize inputs.
- **Convention discipline:** fix one indexing convention (this doc uses "layer
  above predicts layer below") and keep it everywhere. Mixing the discriminative
  and generative directions mid-code causes chaos.
- **Vectorize over the batch**; the outer-product weight update is
  `eps[l].T @ f(x[l+1]) / B`.
- Initialize hidden states with a feedforward `mu` pass, not zeros, to cut
  inference steps roughly in half.

---

## 8. References the agent should consult

- **Bogacz, R. (2017).** A tutorial on the free-energy framework for modelling
  perception and learning. _J. Math. Psych._ — cleanest derivation of the
  equations above; primary source.
- **Whittington & Bogacz (2017).** An Approximation of Error Backprop in a PCN
  with Local Hebbian Plasticity. _Neural Computation_ — the MNIST result this
  project reproduces.
- **Millidge, Tschantz & Buckley (2022).** Predictive Coding Approximates
  Backprop Along Arbitrary Computation Graphs. _Neural Computation_ —
  generalization + practical guidance.
- **Salvatori et al. (2024).** A Stable, Fast, and Fully Automatic Learning
  Algorithm for PCNs — for the iPC stretch goal.
- **Salvatori et al. (2025).** A Survey on Brain-Inspired Deep Learning via
  Predictive Coding — map of the field.

Reference implementations exist (e.g. Bogacz-group code and JAX libraries) — use
them only to **cross-check** behavior, not to copy. The point is to build and
understand it.
