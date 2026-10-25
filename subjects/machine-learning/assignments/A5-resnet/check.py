"""Autograder for assignment A5. Usage: python3 check.py [your_file.py]

Forward passes are compared with direct loops written from the definitions, and every
backward pass with central finite differences, so passing means your code is right, not
merely that it matches one particular implementation."""
import importlib.util
import sys
import traceback
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "starter.py"
spec = importlib.util.spec_from_file_location("student", path)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
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


def gradcheck(layer, x, what, tol=1e-6, samples=None):
    """Check layer.backward against central differences of sum(layer.forward(x) * R),
    for the input and for every parameter (all entries, or `samples` random ones each)."""
    R = np.random.default_rng(0).standard_normal(np.shape(layer.forward(x.copy())))
    f = lambda: float(np.sum(layer.forward(x) * R))
    f()
    dx = layer.backward(R)
    grads = {n: np.array(layer.grads[n]) for n in layer.params} if layer.params else {}
    targets = [("input", x, dx)] + [(f"parameter {n}", layer.params[n], grads.get(n)) for n in layer.params]
    for label, arr, g in targets:
        assert g is not None, f"{what}: no gradient stored for {label}"
        g = np.asarray(g)
        assert g.shape == arr.shape, f"{what}: gradient of {label} has shape {g.shape}, expected {arr.shape}"
        idxs = list(np.ndindex(arr.shape))
        if samples and len(idxs) > samples:
            idxs = [idxs[i] for i in np.random.default_rng(1).choice(len(idxs), samples, replace=False)]
        for idx in idxs:
            old = arr[idx]
            arr[idx] = old + 1e-6; up = f()
            arr[idx] = old - 1e-6; dn = f()
            arr[idx] = old
            num = (up - dn) / 2e-6
            assert abs(num - g[idx]) <= tol * max(1, abs(num)), \
                f"{what}: gradient of {label} at {idx} is {g[idx]:.6g}, finite differences give {num:.6g}"


def naive_conv(x, W, b, stride, pad):
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    OH, OW = (H + 2 * pad - k) // stride + 1, (Wd + 2 * pad - k) // stride + 1
    out = np.zeros((N, F, OH, OW))
    for n in range(N):
        for f in range(F):
            for i in range(OH):
                for j in range(OW):
                    out[n, f, i, j] = np.sum(xp[n, :, i * stride:i * stride + k, j * stride:j * stride + k] * W[f]) + b[f]
    return out


# ---------------------------------------------------------------- Part 1

@test("im2col: shape and contents", "row (n, i, j) holds the padded window starting at (i * stride, j * stride), flattened in (channel, row, column) order")
def _():
    x = rng.standard_normal((2, 3, 5, 6))
    for k, stride, pad in [(3, 1, 1), (3, 2, 0), (2, 2, 1), (1, 1, 0)]:
        cols = S.im2col(x, k, stride, pad)
        OH, OW = (5 + 2 * pad - k) // stride + 1, (6 + 2 * pad - k) // stride + 1
        assert cols.shape == (2 * OH * OW, 3 * k * k), f"k={k}, stride={stride}, pad={pad}: shape {cols.shape}, expected {(2 * OH * OW, 3 * k * k)}"
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        for n, i, j in [(0, 0, 0), (1, OH - 1, OW - 1), (1, OH // 2, 0), (0, 0, OW - 1)]:
            want = xp[n, :, i * stride:i * stride + k, j * stride:j * stride + k].ravel()
            close(cols[(n * OH + i) * OW + j], want, 1e-12, f"k={k}, stride={stride}, pad={pad}: row for (n={n}, i={i}, j={j})")


@test("col2im is the adjoint of im2col", "<im2col(x), Y> must equal <x, col2im(Y)> for all x and Y: add overlapping windows, drop the padding")
def _():
    for k, stride, pad in [(3, 1, 1), (3, 2, 1), (2, 2, 0), (5, 1, 2)]:
        x = rng.standard_normal((2, 2, 7, 6))
        cols = S.im2col(x, k, stride, pad)
        Y = rng.standard_normal(cols.shape)
        back = S.col2im(Y, x.shape, k, stride, pad)
        assert back.shape == x.shape, f"col2im returned shape {back.shape}, expected {x.shape}"
        close(np.sum(cols * Y), np.sum(x * back), 1e-9, f"k={k}, stride={stride}, pad={pad}: <im2col(x), Y> - <x, col2im(Y)>")


@test("Conv2d.forward matches the definition", "out[n, f, i, j] = sum of W[f] times the window at (i * stride, j * stride) of the padded input, plus b[f]")
def _():
    for C, F, k, stride, pad, H, W in [(1, 2, 3, 1, None, 5, 5), (3, 4, 3, 2, 1, 7, 6), (2, 3, 1, 2, 0, 4, 5), (2, 2, 5, 1, 2, 6, 6)]:
        conv = S.Conv2d(C, F, k, stride, pad)
        conv.params["b"] = rng.standard_normal(F)
        x = rng.standard_normal((2, C, H, W))
        close(conv.forward(x), naive_conv(x, conv.params["W"], conv.params["b"], stride, conv.pad), 1e-10, f"C={C}, F={F}, k={k}, stride={stride}")


@test("Conv2d.backward", "dW = g2.T @ cols reshaped like W; dx = col2im(g2 @ W as a matrix); g2 is g with channels last, flattened to rows")
def _():
    gradcheck(S.Conv2d(2, 3, 3, 1), rng.standard_normal((2, 2, 5, 4)), "conv 3x3 stride 1")
    gradcheck(S.Conv2d(3, 2, 3, 2, 1), rng.standard_normal((2, 3, 6, 5)), "conv 3x3 stride 2")
    gradcheck(S.Conv2d(2, 4, 1, 2, 0, bias=False), rng.standard_normal((2, 2, 4, 4)), "conv 1x1 stride 2, no bias")


@test("Conv2d is translation equivariant", "this follows from a correct forward pass")
def _():
    conv = S.Conv2d(1, 2, 3)
    x = np.zeros((1, 1, 12, 12)); x[0, 0, 3:6, 2:5] = rng.standard_normal((3, 3))
    y = np.roll(x, (4, 5), (2, 3))
    close(conv.forward(y)[..., 4:11, 5:11], conv.forward(x)[..., 0:7, 0:6], 1e-12, "shifted output against output of shifted input")


@test("MaxPool2d forward and backward", "with ties the whole gradient of a window goes to one entry")
def _():
    pool = S.MaxPool2d()
    x = rng.standard_normal((2, 3, 4, 6))
    out = pool.forward(x)
    close(out, x.reshape(2, 3, 2, 2, 3, 2).max((3, 5)), 1e-12, "forward")
    gradcheck(pool, x, "max pooling")
    t = np.ones((1, 1, 2, 2))
    pool.forward(t)
    gt = pool.backward(np.array([[[[5.0]]]]))
    assert np.sum(gt == 5.0) == 1 and np.sum(gt) == 5.0, f"a window of four equal entries should send its gradient to exactly one, got {gt.ravel()}"


@test("GlobalAvgPool forward and backward")
def _():
    gap = S.GlobalAvgPool()
    x = rng.standard_normal((3, 4, 5, 2))
    close(gap.forward(x), x.mean((2, 3)), 1e-12, "forward")
    gradcheck(gap, x, "global average pooling")


# ---------------------------------------------------------------- Part 2

@test("BatchNorm2d in training mode", "statistics are per channel, over axes (0, 2, 3); the backward pass is lesson 14's layer-norm formula with the means over those axes")
def _():
    bn = S.BatchNorm2d(3)
    x = 3 + 2 * rng.standard_normal((4, 3, 5, 5))
    y = bn.forward(x)
    close(y.mean((0, 2, 3)), np.zeros(3), 1e-10, "per-channel mean of the output with gamma 1, beta 0")
    close(y.var((0, 2, 3)), x.var((0, 2, 3)) / (x.var((0, 2, 3)) + 1e-5), 1e-10, "per-channel variance of the output")
    bn.params["gamma"] = rng.standard_normal(3); bn.params["beta"] = rng.standard_normal(3)
    gradcheck(bn, rng.standard_normal((3, 3, 4, 2)), "batch norm (training)", tol=1e-5)


@test("BatchNorm2d running statistics and evaluation mode", "running = 0.9 * running + 0.1 * batch statistic, with the unbiased variance; in evaluation mode use the running statistics")
def _():
    bn = S.BatchNorm2d(2)
    x = 1 + 3 * rng.standard_normal((5, 2, 3, 3))
    bn.forward(x)
    n = 5 * 3 * 3
    close(bn.running_mean, 0.1 * x.mean((0, 2, 3)), 1e-12, "running_mean after one step")
    close(bn.running_var, 0.9 + 0.1 * x.var((0, 2, 3)) * n / (n - 1), 1e-12, "running_var after one step")
    bn.training = False
    single = bn.forward(x[:1])
    close(single, bn.forward(x)[:1], 1e-12, "evaluation mode: an example's output must not depend on the rest of the batch")
    want = (x[:1] - bn.running_mean[None, :, None, None]) / np.sqrt(bn.running_var[None, :, None, None] + 1e-5)
    close(single, want, 1e-12, "evaluation mode output")
    rm = bn.running_mean.copy(); bn.forward(x); close(bn.running_mean, rm, 0, "running_mean must not change in evaluation mode")
    gradcheck(bn, rng.standard_normal((2, 2, 3, 3)), "batch norm (evaluation)")


@test("cross_entropy", "subtract each row's maximum before exponentiating; the gradient is (softmax - onehot) / N")
def _():
    L = rng.standard_normal((5, 4)); y = rng.integers(0, 4, 5)
    loss, G = S.cross_entropy(L, y)
    P = np.exp(L) / np.exp(L).sum(1, keepdims=True)
    close(loss, -np.mean(np.log(P[np.arange(5), y])), 1e-12, "loss")
    num = np.zeros_like(L)
    for idx in np.ndindex(L.shape):
        a, b = L.copy(), L.copy(); a[idx] += 1e-6; b[idx] -= 1e-6
        num[idx] = (S.cross_entropy(a, y)[0] - S.cross_entropy(b, y)[0]) / 2e-6
    close(G, num, 1e-7, "gradient")
    loss, G = S.cross_entropy(np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.array([0, 0]))
    close(loss, 500.0, 1e-9, "loss for logits of 1000")


# ---------------------------------------------------------------- Part 3

@test("ResidualBlock starts as relu(shortcut(x))", "bn2's gamma starts at 0, so main(x) is 0 at initialisation")
def _():
    blk = S.ResidualBlock(3, 3)
    x = rng.standard_normal((2, 3, 6, 6))
    close(blk.forward(x), np.maximum(x, 0), 1e-12, "output of an identity-shortcut block at initialisation")
    blk = S.ResidualBlock(3, 5, 2)
    out = blk.forward(x)
    assert out.shape == (2, 5, 3, 3), f"a stride-2 block from 3 to 5 channels on 6x6 should give (2, 5, 3, 3), got {out.shape}"


@test("ResidualBlock.backward", "the gradient splits at the addition and the two paths' gradients add; run each list of layers backwards")
def _():
    for C_in, C_out, stride in [(2, 2, 1), (2, 3, 2)]:
        blk = S.ResidualBlock(C_in, C_out, stride)
        for l in blk.sublayers():
            if "gamma" in l.params:
                l.params["gamma"] = 1 + 0.3 * rng.standard_normal(l.params["gamma"].shape)
                l.params["beta"] = 0.3 * rng.standard_normal(l.params["beta"].shape)
        x = rng.standard_normal((3, C_in, 4, 4))
        R = np.random.default_rng(0).standard_normal(blk.forward(x).shape)
        f = lambda: float(np.sum(blk.forward(x) * R))
        f(); dx = blk.backward(R)
        checks = [("input", x, dx)] + [(f"{type(l).__name__}.{n}", l.params[n], l.grads.get(n)) for l in blk.sublayers() for n in l.params]
        checks = [(lab, arr, None if g is None else np.array(g)) for lab, arr, g in checks]
        for label, arr, g in checks:
            assert g is not None, f"block {C_in}->{C_out}: no gradient for {label}"
            for idx in [tuple(rng.integers(0, s) for s in arr.shape) for _ in range(6)]:
                old = arr[idx]; arr[idx] = old + 1e-6; up = f(); arr[idx] = old - 1e-6; dn = f(); arr[idx] = old
                num = (up - dn) / 2e-6
                assert abs(num - g[idx]) <= 1e-5 * max(1, abs(num)), \
                    f"block {C_in}->{C_out}, stride {stride}: gradient of {label} at {idx} is {g[idx]:.6g}, finite differences give {num:.6g}"


# ---------------------------------------------------------------- Part 4

@test("SGD with momentum and weight decay", "g = grad + wd * p only for parameters named W; v = momentum * v + g; p -= lr * v")
def _():
    lin = S.Linear(3, 2)
    W0, b0 = lin.params["W"].copy(), lin.params["b"].copy()
    gs = [(rng.standard_normal((3, 2)), rng.standard_normal(2)) for _ in range(3)]
    opt = S.SGD([(lin, "W"), (lin, "b")], lr=0.1, momentum=0.9, weight_decay=0.01)
    W, b, vW, vb = W0.copy(), b0.copy(), 0, 0
    for gW, gb in gs:
        lin.grads = {"W": gW, "b": gb}
        opt.step()
        vW = 0.9 * vW + gW + 0.01 * W; W = W - 0.1 * vW
        vb = 0.9 * vb + gb; b = b - 0.1 * vb
    close(lin.params["W"], W, 1e-12, "W after three steps")
    close(lin.params["b"], b, 1e-12, "b after three steps (biases are not decayed)")


@test("augment", "pad by max_shift with zeros, then cut out an H x W window at a random offset, one offset per image")
def _():
    X = rng.random((200, 1, 8, 8)) + 1
    A = S.augment(X, np.random.default_rng(3), 2)
    assert A.shape == X.shape, f"shape {A.shape}, expected {X.shape}"
    seen = set()
    for n in range(200):
        found = None
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                P = np.pad(X[n], ((0, 0), (2, 2), (2, 2)))
                if np.array_equal(A[n], P[:, 2 - dy:10 - dy, 2 - dx:10 - dx]):
                    found = (dy, dx)
        assert found is not None, f"image {n} is not a shift of the original by at most 2 pixels with zero fill"
        seen.add(found)
    assert len(seen) == 25, f"over 200 images only {len(seen)} of the 25 possible shifts occurred"


@test("end to end: a small ResNet learns to recognise shapes anywhere in the image", "if every other check passes, look at Sequential's use of your layers, or at the learning rate")
def _():
    r = np.random.default_rng(0)
    glyphs = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                       [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 0], [1, 1, 1]]], float)
    def make(n):
        y = r.integers(0, 4, n); X = 0.2 * r.standard_normal((n, 1, 12, 12))
        for i in range(n):
            a, b = r.integers(0, 10, 2); X[i, 0, a:a + 3, b:b + 3] += glyphs[y[i]]
        return X, y
    S.rng = np.random.default_rng(1)
    model = S.resnet((8, 8, 16), 4)
    opt = S.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for t in range(400):
        X, y = make(64)
        loss, g = S.cross_entropy(model.forward(X), y)
        model.backward(g); opt.step()
    X, y = make(400)
    acc = S.evaluate(model, X, y)
    assert acc > 0.9, f"test accuracy {acc:.3f} after 400 steps; expected above 0.9"


if __name__ == "__main__":
    for name, ok, msg in results:
        print(("PASS " if ok else "todo " if ok is None else "FAIL ") + name)
        if ok is False:
            print("        " + msg.replace("\n", "\n        "))
    n = sum(ok is True for _, ok, _ in results)
    print(f"\n{n} of {len(results)} checks pass")
