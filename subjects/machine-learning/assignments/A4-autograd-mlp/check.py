"""Autograder for assignment A4. Usage: python3 check.py [your_file.py]

Every gradient is compared with central finite differences, and every optimiser with
properties its update rule must have, so passing means your code is right, not merely
that it matches one particular implementation."""
import importlib.util
import sys
import traceback
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "starter.py"
spec = importlib.util.spec_from_file_location("student", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
T = S.Tensor
rng = np.random.default_rng(7)

results = []


def test(name, hint=""):
    def wrap(fn):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                fn()
            results.append((name, True, ""))
        except NotImplementedError:
            results.append((name, None, "not implemented yet"))
        except AssertionError as e:
            results.append((name, False, (str(e) or "wrong answer") + (f"\n        hint: {hint}" if hint else "")))
        except Exception:
            results.append((name, False, "crashed:\n" + "".join(traceback.format_exc(limit=-2))))
        return fn
    return wrap


def close(a, b, tol, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{what}: shape {a.shape}, expected {b.shape}"
    err = np.max(np.abs(a - b)) if a.size else 0
    assert err <= tol, f"{what}: off by {err:.3g} (tolerance {tol:g})"


def gradcheck(f, arrays, what, tol=1e-6):
    """f maps Tensors to a Tensor; compare backward() with central differences of sum(f * R)."""
    out = f(*[T(a.copy()) for a in arrays])
    R = np.random.default_rng(0).standard_normal(np.shape(out.data))
    ts = [T(a.copy()) for a in arrays]
    L = f(*ts) * T(R, requires_grad=False)
    L = L.sum() if np.ndim(L.data) else L
    L.backward()
    for k, a in enumerate(arrays):
        num = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            for s in (1, -1):
                b = [x.copy() for x in arrays]; b[k][idx] += s * 1e-6
                num[idx] += s * float(np.sum(f(*[T(x) for x in b]).data * R)) / 2e-6
        assert ts[k].grad is not None, f"{what}: input {k} got no gradient"
        close(ts[k].grad, num, tol * max(1, np.max(np.abs(num))), f"{what}: gradient of input {k}")


A, B = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
bias, col = rng.standard_normal(3), rng.standard_normal((4, 1))
M = rng.standard_normal((3, 5))


@test("unbroadcast", "first sum away extra leading axes, then sum (keepdims) every axis where shape has 1 but grad doesn't")
def _():
    G = rng.standard_normal((2, 4, 3))
    close(S.unbroadcast(G, (2, 4, 3)), G, 0, "same shape")
    close(S.unbroadcast(G, (4, 3)), G.sum(0), 1e-12, "(2,4,3) -> (4,3)")
    close(S.unbroadcast(G, (3,)), G.sum((0, 1)), 1e-12, "(2,4,3) -> (3,)")
    close(S.unbroadcast(G, (4, 1)), G.sum(0).sum(1, keepdims=True), 1e-12, "(2,4,3) -> (4,1)")
    close(S.unbroadcast(G, (1, 1, 3)), G.sum((0, 1), keepdims=True), 1e-12, "(2,4,3) -> (1,1,3)")
    close(S.unbroadcast(G, ()), G.sum(), 1e-12, "(2,4,3) -> ()")


@test("add / mul / sub / neg", "each parent gets unbroadcast(out.grad * local derivative, parent.shape), added with +=")
def _():
    gradcheck(lambda a, b: a + b, [A, B], "a + b")
    gradcheck(lambda a, b: a + b, [A, bias], "a + bias (broadcast)")
    gradcheck(lambda a, b: a * b, [A, col], "a * column (broadcast)")
    gradcheck(lambda a, b: a - b, [A, B], "a - b")
    gradcheck(lambda a: -a, [A], "-a")
    gradcheck(lambda a: a * 3.0 + 2.0, [A], "a * 3 + 2 (plain numbers)")
    gradcheck(lambda a: 2.0 * a, [A], "2 * a (number on the left)")


@test("matmul and transpose", "for C = A @ B: grad A = G @ B.T, grad B = A.T @ G; transpose's backward transposes")
def _():
    gradcheck(lambda a, m: a @ m, [A, M], "A @ M")
    gradcheck(lambda a: a.T, [A], "A.T")
    gradcheck(lambda a, b: a @ b.T, [A, B], "A @ B.T")


@test("sum and mean", "the backward of a sum broadcasts the gradient back; for axis without keepdims, re-insert the axis first")
def _():
    gradcheck(lambda a: a.sum(), [A], "sum()")
    gradcheck(lambda a: a.sum(axis=0), [A], "sum(axis=0)")
    gradcheck(lambda a: a.sum(axis=1), [A], "sum(axis=1)")
    gradcheck(lambda a: a.sum(axis=1, keepdims=True), [A], "sum(axis=1, keepdims=True)")
    gradcheck(lambda a: a.mean(), [A], "mean()")


@test("relu / exp / log", "relu passes grad where input > 0; exp multiplies by exp(x); log divides by x")
def _():
    Z = A + np.sign(A) * 0.1                         # keep away from the kink
    gradcheck(lambda a: a.relu(), [Z], "relu")
    gradcheck(lambda a: a.exp(), [A], "exp")
    gradcheck(lambda a: a.log(), [np.abs(A) + 0.5], "log")


@test("backward: shared nodes and depth", "accumulate with +=; visit nodes in reverse topological order with an explicit stack")
def _():
    a = T(A.copy())
    (a * a + a).sum().backward()
    close(a.grad, 2 * A + 1, 1e-10, "d/da sum(a*a + a)")
    a = T(A.copy()); b = a.exp(); c = (b * a + b.log()).sum(); c.backward()
    close(a.grad, np.exp(A) * A + np.exp(A) + 1, 1e-8, "d/da sum(e^a a + log e^a)")
    x = T(np.array([1.0])); y = x
    for _ in range(5000):
        y = y * 1.0001
    y.sum().backward()
    close(x.grad, [1.0001 ** 5000], 1e-6, "gradient through 5,000 chained operations")


@test("cross_entropy", "subtract the row max before exp; backward is (softmax - onehot) / N times out.grad")
def _():
    Z, t = rng.standard_normal((6, 4)) * 3, rng.integers(0, 4, 6)
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    close(S.cross_entropy(T(Z), t).data, -np.log(P[np.arange(6), t]).mean(), 1e-10, "value")
    gradcheck(lambda z: S.cross_entropy(z, t), [Z], "cross_entropy")
    big = S.cross_entropy(T(np.array([[1000.0, 0.0], [0.0, 1000.0]])), np.array([0, 0]))
    close(big.data, 500.0, 1e-9, "value at logits of 1000 (must not overflow)")
    z = T(Z.copy()); S.cross_entropy(z, t).backward()
    close(z.grad.sum(1), np.zeros(6), 1e-12, "each row of the logit gradient sums to 0")


@test("Linear and MLP", "W has shape (fan_in, fan_out) and std sqrt(2 / fan_in); MLP: ReLU between layers, none after the last")
def _():
    lin = S.Linear(512, 300, np.random.default_rng(0))
    assert lin.W.shape == (512, 300) and lin.b.shape == (300,), "Linear(512, 300): W should be (512, 300), b (300,)"
    sd = lin.W.data.std()
    assert abs(sd / np.sqrt(2 / 512) - 1) < 0.03, f"W std {sd:.4f}, expected about {np.sqrt(2 / 512):.4f} (He)"
    close(lin.b.data, np.zeros(300), 0, "bias starts at 0")
    net = S.MLP([5, 7, 3], np.random.default_rng(1))
    assert len(net.params()) == 4, "MLP([5, 7, 3]) should have 4 parameter tensors"
    X = rng.standard_normal((10, 5))
    W1, b1, W2, b2 = [p.data for p in net.params()]
    close(net(T(X, requires_grad=False)).data, np.maximum(X @ W1 + b1, 0) @ W2 + b2, 1e-10, "MLP forward")
    out = net(T(X, requires_grad=False)); S.cross_entropy(out, np.arange(10) % 3).backward()
    assert all(p.grad is not None and p.grad.shape == p.shape for p in net.params()), "every parameter needs a gradient of its own shape"


@test("dropout", "mask = rng.random(x.shape) >= p, scaled by 1 / (1 - p); return x itself when not training")
def _():
    x = T(np.ones((200, 300)))
    assert S.dropout(x, 0.5, False, np.random.default_rng(0)) is x, "not training: return x unchanged"
    y = S.dropout(x, 0.3, True, np.random.default_rng(4))
    mask = np.random.default_rng(4).random((200, 300)) >= 0.3
    close(y.data, mask / 0.7, 1e-12, "dropout output with the specified random mask")
    y.sum().backward()
    close(x.grad, mask / 0.7, 1e-12, "gradient flows only through survivors, with the same scale")
    net = S.MLP([5, 50, 3], np.random.default_rng(1), p_drop=0.5)
    X = T(rng.standard_normal((4, 5)), requires_grad=False)
    net.training = False
    a, b = net(X).data, net(X).data
    close(a, b, 0, "with training = False the MLP must be deterministic")


@test("SGD", "g = grad + wd * p; v = momentum * v + g; p -= lr * v; zero_grad sets every grad to None")
def _():
    p, q = T(np.array([1.0, -2.0])), T(np.array([5.0]))
    opt = S.SGD([p, q], lr=0.1, momentum=0.9, weight_decay=0.01)
    v, w = np.zeros(2), np.array([1.0, -2.0])
    for g in [np.array([1.0, 1.0]), np.array([-2.0, 0.5]), np.array([0.0, 3.0])]:
        p.grad = g.copy(); opt.step()
        v = 0.9 * v + g + 0.01 * w; w = w - 0.1 * v
    close(p.data, w, 1e-12, "parameter after three momentum steps")
    close(q.data, [5.0], 0, "a parameter with grad None must not move")
    opt.zero_grad()
    assert p.grad is None and q.grad is None, "zero_grad should set grads to None"


@test("AdamW", "bias-correct m and v with 1 - beta^t (t from 1); decay is lr * wd * p, outside the adaptive step")
def _():
    p = T(np.array([1.0, 1.0, 1.0]))
    opt = S.AdamW([p], lr=0.01)
    p.grad = np.array([1e-2, -3.0, 50.0]); opt.step()
    close(p.data, [0.99, 1.01, 0.99], 1e-6, "first Adam step moves each coordinate by lr * sign(g)")
    for _ in range(5):
        p.grad = np.array([1e-2, -3.0, 50.0]); opt.step()
    close(p.data, [0.94, 1.06, 0.94], 1e-6, "with a constant gradient every step has size lr")
    p = T(np.array([2.0, -4.0])); opt = S.AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = np.zeros(2); opt.step()
    close(p.data, [2.0 * 0.95, -4.0 * 0.95], 1e-9, "with zero gradient, decoupled decay multiplies p by (1 - lr * wd)")
    p = T(np.array([0.0])); opt = S.AdamW([p], lr=1.0, betas=(0.5, 0.75), eps=0.0)
    p.grad = np.array([2.0]); opt.step(); p.grad = np.array([-1.0]); opt.step()
    m, v = 0.5 * 0.5 * 2 + 0.5 * -1, 0.75 * 0.25 * 4 + 0.25 * 1
    close(p.data, [-1.0 - (m / 0.75) / np.sqrt(v / (1 - 0.75 ** 2))], 1e-9, "second step with bias correction")


@test("end to end", "if the parts pass but this fails, check that you call zero_grad before backward each step")
def _():
    r = np.random.default_rng(3)
    X = r.uniform(-1, 1, (400, 2)); y = ((X[:, 0] * X[:, 1]) > 0).astype(int)     # XOR quadrants
    net = S.MLP([2, 32, 32, 2], np.random.default_rng(0))
    opt = S.AdamW(net.params(), lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        loss = S.cross_entropy(net(T(X, requires_grad=False)), y)
        loss.backward(); opt.step()
    acc = np.mean(net(T(X, requires_grad=False)).data.argmax(1) == y)
    assert acc > 0.95, f"training accuracy {acc:.3f} on the XOR quadrants after 300 AdamW steps; expected > 0.95"


done = sum(ok is True for _, ok, _m in results)
for name, ok, msg in results:
    mark = {True: "PASS", False: "FAIL", None: "todo"}[ok]
    print(f"  {mark}  {name}" + (f"\n        {msg}" if msg else ""))
print(f"\n{done} / {len(results)} passed")
sys.exit(0 if done == len(results) else 1)
