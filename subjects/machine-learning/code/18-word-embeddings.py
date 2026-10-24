import numpy as np

rng = np.random.default_rng(0)

# A toy language with hidden structure: each person word has a gender, a status and an age,
# and the words around it depend on those attributes, never on the word itself.
people = {"king": "m royal adult", "queen": "f royal adult", "prince": "m royal child", "princess": "f royal child",
          "man": "m common adult", "woman": "f common adult", "boy": "m common child", "girl": "f common child"}
context = {"m": ["he", "his", "him"], "f": ["she", "her", "hers"],
           "royal": ["palace", "crown", "throne", "rules"], "common": ["village", "farm", "market", "works"],
           "adult": ["old", "tall", "married"], "child": ["young", "small", "school"]}
filler = ["the", "a", "and", "then", "very", "today"]

def sentence():
    w = rng.choice(list(people))
    attrs = people[w].split()
    words = [rng.choice(context[a]) for a in attrs for _ in range(2)] + list(rng.choice(filler, 3))
    rng.shuffle(words)
    words.insert(rng.integers(0, len(words) + 1), w)
    return words

corpus = [sentence() for _ in range(20000)]
vocab = sorted({w for s in corpus for w in s}); idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)

# (centre, context) pairs within a window of 2
pairs = np.array([(idx[s[i]], idx[s[j]]) for s in corpus for i in range(len(s))
                  for j in range(max(0, i - 2), min(len(s), i + 3)) if j != i])
counts = np.bincount(pairs[:, 1], minlength=V).astype(float)
noise = counts ** 0.75; noise /= noise.sum()                   # negatives drawn from unigram^0.75
print(f"{len(corpus)} sentences, {V} words, {len(pairs)} (word, context) pairs")

# Skip-gram with negative sampling: maximise log s(w.c) + sum_k log s(-w.c_k)
d, k, lr = 16, 5, 0.02
Wv = rng.standard_normal((V, d)) * 0.1; Cv = np.zeros((V, d))
sig = lambda z: 1 / (1 + np.exp(-z))
for epoch in range(3):
    total = 0.0
    for batch in np.array_split(rng.permutation(len(pairs)), len(pairs) // 512):
        w, c = pairs[batch, 0], pairs[batch, 1]
        neg = rng.choice(V, (len(batch), k), p=noise)
        ew, ec, en = Wv[w], Cv[c], Cv[neg]                        # (B, d), (B, d), (B, k, d)
        sp, sn = sig(np.sum(ew * ec, 1)), sig(np.einsum("bd,bkd->bk", ew, en))
        total += -np.sum(np.log(sp)) - np.sum(np.log(1 - sn))
        gp, gn = (sp - 1)[:, None], sn[:, :, None]                # gradients of the loss w.r.t. the scores
        dw = gp * ec + np.sum(gn * en, 1)
        np.add.at(Cv, c, -lr * gp * ew); np.add.at(Cv, neg, -lr * gn * ew[:, None, :]); np.add.at(Wv, w, -lr * dw)
    print(f"epoch {epoch + 1}: mean loss per pair {total / len(pairs):.3f}")

# 1. What does w.c converge to? Levy and Goldberg: PMI(w, c) - log k.
joint = np.zeros((V, V)); np.add.at(joint, (pairs[:, 0], pairs[:, 1]), 1)
pw, pc = joint.sum(1) / joint.sum(), joint.sum(0) / joint.sum()
seen = joint > 20                                              # pairs seen often enough to estimate PMI
pmi = np.log(joint[seen] / joint.sum() / np.outer(pw, pc)[seen])
dots = (Wv @ Cv.T)[seen]
print(f"correlation of w.c with PMI - log k over {seen.sum()} frequent pairs: {np.corrcoef(dots, pmi - np.log(k))[0, 1]:.3f}"
      f"   mean difference {np.mean(dots - (pmi - np.log(k))):+.2f}")

# 2. Nearest neighbours and analogies by cosine similarity
E = Wv / np.linalg.norm(Wv, axis=1, keepdims=True)
def nearest(v, exclude=(), n=3):
    sims = E @ (v / np.linalg.norm(v))
    return [vocab[i] for i in np.argsort(-sims) if vocab[i] not in exclude][:n]
for w in ["king", "girl", "palace"]:
    print(f"nearest to {w:7s}: {', '.join(nearest(E[idx[w]], exclude=[w]))}")
tests = [("man", "king", "woman", "queen"), ("boy", "man", "girl", "woman"), ("prince", "king", "princess", "queen"),
         ("man", "boy", "queen", "princess"), ("woman", "queen", "man", "king"), ("boy", "prince", "girl", "princess")]
right = 0
for a, b, c, want in tests:                                    # a is to b as c is to ?
    got = nearest(E[idx[b]] - E[idx[a]] + E[idx[c]], exclude=[a, b, c], n=1)[0]
    right += got == want
    print(f"{a} : {b} :: {c} : {got:9s} {'ok' if got == want else '(wanted ' + want + ')'}")
print(f"{right} of {len(tests)} analogies right")

# 20000 sentences, 34 words, 680000 (word, context) pairs
# epoch 1: mean loss per pair 2.601
# epoch 2: mean loss per pair 2.585
# epoch 3: mean loss per pair 2.585
# correlation of w.c with PMI - log k over 864 frequent pairs: 0.597   mean difference -0.01
# nearest to king   : prince, queen, man
# nearest to girl   : boy, princess, woman
# nearest to palace : rules, crown, throne
# man : king :: woman : queen     ok
# boy : man :: girl : woman     ok
# prince : king :: princess : queen     ok
# man : boy :: queen : princess  ok
# woman : queen :: man : king      ok
# boy : prince :: girl : princess  ok
# 6 of 6 analogies right
