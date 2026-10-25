import math
import random

class Value:
    """A scalar that remembers how it was computed, so it can send gradients back."""
    def __init__(self, data, parents=(), local_grads=()):
        self.data, self.grad = data, 0.0
        self.parents, self.local_grads = parents, local_grads    # d(self)/d(parent) for each parent

    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data + o.data, (self, o), (1.0, 1.0))

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data * o.data, (self, o), (o.data, self.data))

    def __pow__(self, k):                      # k is a plain number
        return Value(self.data ** k, (self,), (k * self.data ** (k - 1),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        return Value(max(self.data, 0.0), (self,), (float(self.data > 0),))

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def __truediv__(self, o): return self * o ** -1

    def backward(self):
        """Reverse-mode autodiff: visit nodes in reverse topological order and apply the chain rule."""
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            for p, local in zip(v.parents, v.local_grads):
                p.grad += local * v.grad           # += : a node used twice gets gradient from both uses

# 1. A small expression, checked against the derivative worked out by hand
x, y = Value(2.0), Value(-1.0)
f = (x * y + x.exp()) * x                      # f = x^2 y + x e^x
f.backward()
print(f"f = {f.data:.4f}   df/dx = {x.grad:.4f} (by hand {2*2*-1 + math.exp(2)*(1+2):.4f})"
      f"   df/dy = {y.grad:.4f} (by hand {2**2:.4f})")

# 2. A neuron layer and a network built from Values
class MLP:
    def __init__(self, sizes, seed=0):
        # small random biases: with zero biases, input (0, 0) sits exactly on every ReLU's kink,
        # where the derivative is undefined and the finite-difference check below would fail
        r = random.Random(seed)
        self.layers = [[([Value(r.gauss(0, math.sqrt(2 / k))) for _ in range(k)], Value(r.gauss(0, 0.1))) for _ in range(m)]
                       for k, m in zip(sizes[:-1], sizes[1:])]
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            z = [sum((wi * xi for wi, xi in zip(w, x)), b) for w, b in layer]
            x = [zi.relu() for zi in z] if i < len(self.layers) - 1 else z
        return x[0]
    def params(self):
        return [p for layer in self.layers for w, b in layer for p in w + [b]]

def log_loss(F, y):                            # log(1 + e^F) - y F, written with Values
    return (F.exp() + 1).log() - F * y

# 3. Compare one backward pass with central finite differences
net = MLP([2, 4, 1], seed=1)
data = [((0.0, 0.0), 0), ((0.0, 1.0), 1), ((1.0, 0.0), 1), ((1.0, 1.0), 0)]    # XOR
def total_loss():
    return sum((log_loss(net([Value(a), Value(b)]), t) for (a, b), t in data), Value(0.0)) * 0.25
L = total_loss(); L.backward()
worst = 0.0
for p in net.params():
    old = p.data
    p.data = old + 1e-6; up = total_loss().data
    p.data = old - 1e-6; down = total_loss().data
    p.data = old
    worst = max(worst, abs((up - down) / 2e-6 - p.grad))
print(f"{len(net.params())} parameters, max |backprop - finite difference| = {worst:.1e}")

# 4. Train it on XOR with plain gradient descent
for step in range(501):
    for p in net.params():
        p.grad = 0.0                           # gradients accumulate, so reset them every step
    L = total_loss(); L.backward()
    for p in net.params():
        p.data -= 0.5 * p.grad
    if step % 100 == 0:
        probs = [1 / (1 + math.exp(-net([Value(a), Value(b)]).data)) for (a, b), _ in data]
        print(f"step {step:3d}  loss {L.data:.4f}  P(y=1) on 00,01,10,11: " + " ".join(f"{q:.2f}" for q in probs))

# f = 10.7781   df/dx = 18.1672 (by hand 18.1672)   df/dy = 4.0000 (by hand 4.0000)
# 17 parameters, max |backprop - finite difference| = 8.6e-11
# step   0  loss 0.6985  P(y=1) on 00,01,10,11: 0.48 0.54 0.50 0.54
# step 100  loss 0.0754  P(y=1) on 00,01,10,11: 0.04 0.92 0.90 0.08
# step 200  loss 0.0193  P(y=1) on 00,01,10,11: 0.01 0.98 0.97 0.02
# step 300  loss 0.0097  P(y=1) on 00,01,10,11: 0.01 0.99 0.99 0.01
# step 400  loss 0.0062  P(y=1) on 00,01,10,11: 0.00 0.99 0.99 0.01
# step 500  loss 0.0044  P(y=1) on 00,01,10,11: 0.00 1.00 0.99 0.00
