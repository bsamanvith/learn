"""Extract each lesson's "Code from scratch" block to code/<lesson>.py and check its printed output.

    python3 tools/lesson_code.py machine-learning [lesson-file ...]

The block's trailing comment lines (after the last blank line) are the recorded output. The
script is run and its stdout is compared with them line by line, so the numbers quoted in a
lesson can't drift from what the code actually prints. Lines may end in "(...)" annotations,
which are ignored, and "..." matches anything after it."""
import html, re, subprocess, sys
from pathlib import Path

subj = sys.argv[1]
root = Path("subjects") / subj
only = set(sys.argv[2:])
(root / "code").mkdir(exist_ok=True)
bad = 0
for f in sorted((root / "lessons").glob("*.html")):
    if only and f.stem not in only:
        continue
    m = re.search(r'<section id="code">.*?<pre><code>(.*?)</code></pre>', f.read_text(), re.S)
    if not m:
        continue
    src = html.unescape(m.group(1))
    lines = src.rstrip("\n").split("\n")
    i = len(lines)
    while i > 0 and lines[i - 1].startswith("#"):
        i -= 1
    expected = [l[2:] if l.startswith("# ") else l[1:] for l in lines[i:]]
    out_py = root / "code" / f"{f.stem}.py"
    out_py.write_text(src.rstrip("\n") + "\n")
    if not expected:
        print(f"{f.stem}: no recorded output"); continue
    r = subprocess.run([sys.executable, out_py.name], cwd=out_py.parent, capture_output=True, text=True, timeout=600)
    got = [l.rstrip() for l in r.stdout.rstrip("\n").split("\n")]
    ok = len(got) == len(expected)
    for g, e in zip(got, expected):
        if " ".join(g.split()) == " ".join(e.split()):
            continue
        e = re.sub(r"\s+\(.*\)\s*$", "", e).rstrip()
        e = re.sub(r"\s+&lt;-.*$|\s+<-.*$", "", e).rstrip()
        g2 = g
        if "..." in e:
            e, g2 = e.split("...")[0].rstrip(), g[: len(e.split("...")[0].rstrip())]
        if " ".join(g2.split()) != " ".join(e.split()):
            ok = False; print(f"  {f.stem}\n    got:      {g}\n    recorded: {e}")
    if r.returncode:
        ok = False; print(r.stderr[-800:])
    bad += not ok
    print(f"{f.stem}: {'ok' if ok else 'MISMATCH'} ({len(got)} lines)")
sys.exit(1 if bad else 0)
