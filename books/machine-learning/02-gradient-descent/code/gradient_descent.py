"""Gradient descent for 1-input linear regression, from scratch.

Run:  python gradient_descent.py            (default learning rate 0.1)
      python gradient_descent.py 0.01       (too small: slow)
      python gradient_descent.py 0.5        (too big: diverges)

Needs only NumPy. If matplotlib is installed, it also plots the loss curve.
"""

import sys

import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 5.0])


def loss(w, b):
    return np.mean((w * x + b - y) ** 2)


def gradient(w, b):
    r = w * x + b - y  # residuals: guess minus truth
    return 2 * np.mean(r * x), 2 * np.mean(r)


def numerical_gradient(w, b, h=1e-6):
    """Check the calculus: nudge each knob and measure the change."""
    return (loss(w + h, b) - loss(w, b)) / h, (loss(w, b + h) - loss(w, b)) / h


def main():
    lr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    w, b = 0.0, 0.0

    print("analytic gradient at (0, 0): ", gradient(w, b))
    print("numerical gradient at (0, 0):", numerical_gradient(w, b))

    # Largest safe learning rate: 2 / (largest curvature of the bowl).
    hessian = 2 * np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])
    print(f"max stable learning rate ~ {2 / np.linalg.eigvalsh(hessian).max():.3f}\n")

    history = []
    print(f"{'step':>4} {'w':>9} {'b':>9} {'dL/dw':>9} {'dL/db':>9} {'loss':>10}")
    for step in range(201):
        gw, gb = gradient(w, b)
        history.append(loss(w, b))
        if step < 10 or step % 20 == 0:
            print(f"{step:>4} {w:>9.4f} {b:>9.4f} {gw:>9.4f} {gb:>9.4f} {history[-1]:>10.5f}")
        if not np.isfinite(history[-1]) or history[-1] > 1e8:
            print("diverged: the learning rate is too big")
            break
        w, b = w - lr * gw, b - lr * gb  # the update

    print(f"\nexact answer: w = 1.5, b = {2 / 3:.4f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.semilogy(history)
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    plt.title(f"Gradient descent, learning rate {lr}")
    plt.show()


if __name__ == "__main__":
    main()
