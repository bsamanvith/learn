"""Check that every internal link in the built site points at a page that exists.
    python3 tools/check_links.py site/dist /learn"""
import re, sys
from pathlib import Path
from urllib.parse import urlparse

dist, base = Path(sys.argv[1]), sys.argv[2].rstrip("/")
bad = 0
for page in dist.rglob("*.html"):
    if page.relative_to(dist).parts[0] == "subjects":   # raw source copies, not pages
        continue
    for href in re.findall(r'href="([^"#?]+)', page.read_text()):
        if href.startswith(("http:", "https:", "mailto:", "data:")):
            continue
        path = urlparse(href).path
        if not path.startswith(base + "/"):
            if path.startswith("/"):
                print(f"{page}: link without base path: {href}"); bad += 1
            continue
        target = dist / path[len(base) + 1:]
        if not (target.is_file() or (target / "index.html").is_file()):
            print(f"{page.relative_to(dist)}: broken link {href}"); bad += 1
print(f"{bad} broken links")
sys.exit(1 if bad else 0)
