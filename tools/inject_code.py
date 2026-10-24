"""Put a script and its real output into a lesson: replaces the @@CODE@@ marker with the
HTML-escaped source followed by its stdout as comment lines.
    python3 tools/inject_code.py <lesson.html> <script.py>"""
import html, subprocess, sys
from pathlib import Path

lesson, script = Path(sys.argv[1]), Path(sys.argv[2])
r = subprocess.run([sys.executable, script.name], cwd=script.parent, capture_output=True, text=True, timeout=900)
if r.returncode:
    sys.exit(r.stderr)
out = "\n".join("# " + l if l else "#" for l in r.stdout.rstrip("\n").split("\n"))
block = html.escape(script.read_text().rstrip("\n") + "\n\n" + out, quote=False)
s = lesson.read_text()
assert "@@CODE@@" in s, "no @@CODE@@ marker"
lesson.write_text(s.replace("@@CODE@@", block))
print(r.stdout)
