import numpy as np
import re
from collections import Counter

rng = np.random.default_rng(0)

# 1. A toy language whose true entropy we can compute. Each sentence makes independent random
# choices: a subject, a verb, an object, and optionally a place. Words are distinct across slots,
# so a sentence determines its choices and H(sentence) = sum of the entropies of the choices.
SLOTS = [
    ("subject", ["the cat", "the dog", "a bird", "my sister", "the old man", "a child", "the teacher", "our neighbour"],
                [0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05]),
    ("verb", ["sees", "likes", "follows", "draws", "watches", "remembers"], [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
    ("object", ["the river", "a red kite", "the mountains", "the moon", "an empty house", "the morning train"],
               [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]),
    ("place", ["", " from the bridge", " in the rain", " every evening", " near the station"], [0.4, 0.2, 0.15, 0.15, 0.1]),
]
H = lambda p: -np.sum(np.array(p) * np.log2(p))
H_sentence = sum(H(p) for _, _, p in SLOTS)

def sentence():
    idx = [rng.choice(len(w), p=p) for _, w, p in SLOTS]
    s, q, o, pl = (w[i] for (_, w, _), i in zip(SLOTS, idx))
    return f"{s} {q} {o}{pl}. ", -sum(np.log2(p[i]) for (_, _, p), i in zip(SLOTS, idx))

train = "".join(sentence()[0] for _ in range(20000))
test_pairs = [sentence() for _ in range(2000)]
test = "".join(t for t, _ in test_pairs)
mean_len = len(test) / 2000
print(f"true entropy: {H_sentence:.3f} bits per sentence; sentences average {mean_len:.1f} characters,"
      f" so {H_sentence / mean_len:.4f} bits per character")
print(f"the true model's cross-entropy on this test set: {sum(b for _, b in test_pairs) / len(test):.4f} bits per character")

# 2. Character n-gram models with add-k smoothing, scored by cross-entropy on held-out text.
chars = sorted(set(train)); V = len(chars)
def ngram_counts(seq, n):
    counts, ctx = Counter(), Counter()
    for i in range(n - 1, len(seq)):
        counts[seq[i - n + 1:i + 1]] += 1; ctx[seq[i - n + 1:i]] += 1
    return counts, ctx

def ngram_bits_per_char(n, k=0.01):
    counts, ctx = ngram_counts(train, n)
    bits = [-np.log2((counts[test[i - n + 1:i + 1]] + k) / (ctx[test[i - n + 1:i]] + k * V)) for i in range(n - 1, len(test))]
    return np.mean(bits)
print(f"vocabulary: {V} characters (uniform model: {np.log2(V):.3f} bits per character)")
for n in [1, 2, 3, 5, 8, 12]:
    b = ngram_bits_per_char(n)
    print(f"{n:2d}-gram: {b:.4f} bits per character, perplexity per character {2 ** b:6.3f}")

# 3. Byte-pair encoding: start from characters, repeatedly merge the most frequent adjacent pair.
# As in GPT-2, the text is first split into words, each keeping the space in front of it, and
# merges never cross a word boundary.
pretokenise = lambda text: re.findall(r" ?[a-z-]+| ?\.", text)
def bpe_train(text, n_merges):
    words = Counter(pretokenise(text))
    seqs = {w: tuple(w) for w in words}
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, c in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))                   # ties go to the alphabetically last pair
        merges.append(best)
        for w in words:
            s, out, i = seqs[w], [], 0
            while i < len(s):
                if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                    out.append(s[i] + s[i + 1]); i += 2
                else:
                    out.append(s[i]); i += 1
            seqs[w] = tuple(out)
    return merges

def bpe_encode(word, merges):
    s = list(word)
    for a, b in merges:                                                  # apply merges in the order learned
        i = 0
        while i < len(s) - 1:
            if s[i] == a and s[i + 1] == b:
                s[i:i + 2] = [a + b]
            else:
                i += 1
    return s

merges = bpe_train(train, 150)
print("first 12 merges:", ["".join(m) for m in merges[:12]])
words = pretokenise(test)
for n_m in [0, 10, 40, 150]:
    toks = sum(len(bpe_encode(w, merges[:n_m])) for w in words)
    print(f"{n_m:3d} merges: {len(test) / toks:.2f} characters per token")
print("encoding 'the old man remembers the mountains':",
      [t for w in pretokenise("the old man remembers the mountains") for t in bpe_encode(w, merges)])

# 4. Perplexities are only comparable per character (or byte). A bigram model over BPE tokens,
# scored per token and then converted to bits per character.
def token_bigram(n_m, k=0.01):
    tr = [t for w in pretokenise(train) for t in bpe_encode(w, merges[:n_m])]
    te = [t for w in pretokenise(test) for t in bpe_encode(w, merges[:n_m])]
    V = len(set(tr)) + 1                                             # one extra for unseen tokens
    counts, ctx = Counter(zip(tr, tr[1:])), Counter(tr[:-1])
    bits = sum(-np.log2((counts[(a, b)] + k) / (ctx[a] + k * V)) for a, b in zip(te, te[1:]))
    return V - 1, bits / (len(te) - 1), bits / len(test)
for n_m in [0, 40, 150]:
    V_t, per_tok, per_char = token_bigram(n_m)
    print(f"token bigram, {n_m:3d} merges ({V_t:3d} types): {per_tok:.3f} bits per token,"
          f" perplexity {2 ** per_tok:6.2f}; {per_char:.4f} bits per character")

# 5. Sampling from the 8-gram model with a temperature: p_i^(1/tau), renormalised.
counts, _ = ngram_counts(train, 8)
nexts = {}
for g, c in counts.items():
    nexts.setdefault(g[:-1], []).append((g[-1], c))
def generate(tau, n_chars=120, seed=1):
    r = np.random.default_rng(seed); out = train[:7]
    while len(out) < n_chars:
        cs, c = zip(*nexts[out[-7:]])
        logp = np.log(np.array(c, float)) / tau
        p = np.exp(logp - logp.max()); p /= p.sum()
        out += cs[r.choice(len(cs), p=p)]
    return out
for tau in [1.0, 0.3, 3.0]:
    print(f"tau = {tau}: {generate(tau)!r}")

# true entropy: 9.868 bits per sentence; sentences average 37.3 characters, so 0.2647 bits per character
# the true model's cross-entropy on this test set: 0.2642 bits per character
# vocabulary: 24 characters (uniform model: 4.585 bits per character)
#  1-gram: 3.9925 bits per character, perplexity per character 15.917
#  2-gram: 2.3813 bits per character, perplexity per character  5.210
#  3-gram: 0.8606 bits per character, perplexity per character  1.816
#  5-gram: 0.3807 bits per character, perplexity per character  1.302
#  8-gram: 0.2646 bits per character, perplexity per character  1.201
# 12-gram: 0.2650 bits per character, perplexity per character  1.202
# first 12 merges: ['he', ' t', ' the', 'in', ' r', 'er', ' m', ' a', ' s', 'es', 'at', 'ver']
#   0 merges: 1.00 characters per token
#  10 merges: 1.37 characters per token
#  40 merges: 2.00 characters per token
# 150 merges: 4.58 characters per token
# encoding 'the old man remembers the mountains': ['t', 'he', ' old', ' man', ' remembers', ' the', ' mountains']
# token bigram,   0 merges ( 24 types): 2.381 bits per token, perplexity   5.21; 2.3813 bits per character
# token bigram,  40 merges ( 54 types): 1.723 bits per token, perplexity   3.30; 0.8628 bits per character
# token bigram, 150 merges ( 39 types): 1.686 bits per token, perplexity   3.22; 0.3677 bits per character
# tau = 1.0: 'my sister draws the moon in the rain. a child remembers a red kite from the bridge. a bird follows the river every eveni'
# tau = 0.3: 'my sister follows the mountains. the cat draws the moon. the cat likes the mountains. the dog sees the river every eveni'
# tau = 3.0: 'my sister draws the moon. a child sees an empty house in the rain. my sister sees the mountains near the station. my sis'
