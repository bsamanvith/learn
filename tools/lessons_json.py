"""Rewrite a subject's lessons.json with one entry per line (stable, diff-friendly).
    python3 tools/lessons_json.py subjects/machine-learning/lessons.json"""
import json, sys

p = sys.argv[1]
d = json.load(open(p))
out = ["{", f'  "_about": {json.dumps(d["_about"], ensure_ascii=False)},']
for key in ("lessons", "pages"):
    rows = [json.dumps(e, ensure_ascii=False, separators=(", ", ": ")) for e in d.get(key, [])]
    out.append(f'  "{key}": [')
    out.append(",\n".join("    " + r for r in rows))
    out.append("  ]," if key == "lessons" else "  ]")
out.append("}")
open(p, "w").write("\n".join(out) + "\n")
