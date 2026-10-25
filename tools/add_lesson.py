"""Append (or replace) a lesson entry in a subject's lessons.json and reformat it.
    python3 tools/add_lesson.py <subject> <file> <chapter> <part> "<title>" "<prereqs comma-sep>" "<blurb>" """
import json, subprocess, sys
subj, file, chapter, part, title, pre, blurb = sys.argv[1:8]
p = f"subjects/{subj}/lessons.json"
d = json.load(open(p))
e = {"file": file, "chapter": chapter, "part": int(part), "title": title, "added": "2026-10-24",
     "prerequisites": [x for x in pre.split(",") if x], "blurb": blurb}
d["lessons"] = [x for x in d["lessons"] if x["file"] != file] + [e]
json.dump(d, open(p, "w"))
subprocess.run([sys.executable, "tools/lessons_json.py", p], check=True)
