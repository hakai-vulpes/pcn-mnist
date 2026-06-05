"""Predictive Coding Networks (PCN) trained by inference learning on MNIST.

A hierarchical PCN following Bogacz (2017) and Whittington & Bogacz (2017).
Credit assignment is done by manual local Hebbian updates -- never by
``loss.backward()`` (autograd is used only in tests as a correctness oracle).
"""

from pcn.functional import energy, tanh, tanh_grad

__all__ = ["energy", "tanh", "tanh_grad"]
