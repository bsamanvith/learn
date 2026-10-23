// Turns the repo's plain Markdown conventions into site structure, so the source files
// stay readable on GitHub and in an editor and never need editing for the site:
//
//   > 🎛 **Play:** [name](../widgets/x.html) ...   -> an embedded, theme-synced widget
//   ../text/*.md, ../exercises/*.md, ../sims/*.py  -> site routes
//   ## 1. Hook: Title                              -> <section class="sec sec-hook"> with an eyebrow
//   ### 1.5.3 🟢 Title (exercise files)            -> problem card linked to its solution
//   <details><summary>1.5.3</summary>               -> gets id="sol-1.5.3"
import { visit } from "unist-util-visit";
import path from "node:path";

const KINDS = [
  ["hook", /^hook/i], ["intuition", /^intuition/i], ["definition", /^definition/i],
  ["worked", /^worked/i], ["faded", /^faded/i], ["checks", /^quick check/i],
  ["theory", /^theory/i], ["pitfalls", /^pitfall/i], ["simulation", /^simulation/i],
  ["connections", /^connections/i], ["reading", /^further reading/i], ["solutions", /^solutions/i],
];

const kindOf = (title) => (KINDS.find(([, re]) => re.test(title)) || ["other"])[0];

// Map a link written relative to a chapter file onto a site URL.
export function rewriteHref(href, fromFile, base = "") {
  if (!href || /^(https?:|mailto:|#|\/)/.test(href)) return href;
  const [rel, hash = ""] = href.split("#");
  const abs = path.resolve(path.dirname(fromFile), rel);
  const parts = abs.split(path.sep);
  const i = parts.lastIndexOf("subjects");
  if (i < 0 || !parts[i + 1]) return href;
  const subject = parts[i + 1];
  const root = base.replace(/\/$/, "");
  const BASE = `${root}/${subject}`;
  const [dir, file] = parts.slice(i + 2);
  const frag = hash ? `#${hash}` : "";
  if (dir === "widgets" && file) return `${root}/subjects/${subject}/widgets/${file}${frag}`;
  if (dir === "text" && file) return `${BASE}/${file.replace(/\.md$/, "")}/${frag}`;
  if (dir === "exercises" && file) return `${BASE}/${file.replace(/\.md$/, "")}/exercises/${frag}`;
  if (dir === "sims" && file) return `${BASE}/sims/${file.replace(/\.py$/, "")}/`;
  if (dir === "README.md" || dir === "PLAN.md" || !dir) return `${BASE}/`;
  return href;
}

const textOf = (node) =>
  node.type === "text" || node.type === "inlineCode" ? node.value
  : node.children ? node.children.map(textOf).join("") : "";

function widgetEmbed(quote, file, base) {
  const para = quote.children[0];
  if (!para || para.type !== "paragraph" || !textOf(para).trim().startsWith("🎛")) return false;
  let src = null, title = "";
  visit(para, "link", (link) => {
    if (!src && /widgets\/[^/]+\.html$/.test(link.url)) {
      src = rewriteHref(link.url, file, base);
      title = textOf(link);
    }
  });
  if (!src) return false;
  const name = src.split("/").pop().replace(/\.html$/, "");
  quote.data = { hName: "figure", hProperties: { className: ["widget"], dataWidget: name } };
  for (const child of quote.children) child.data = { ...child.data, hName: "figcaption" };
  quote.children.unshift({
    type: "html",
    value:
      `<div class="widget-frame"><iframe data-src="${src}" title="${title.replace(/"/g, "&quot;")}" ` +
      `loading="lazy"></iframe></div>` +
      `<a class="widget-open" href="${src}" target="_blank" rel="noopener">Open full size ↗</a>`,
  });
  return true;
}

// "1. Hook: what does 70% mean?" -> eyebrow "Hook", title "What does 70% mean?"
function splitHeading(heading) {
  const first = heading.children[0];
  if (!first || first.type !== "text") return { kind: kindOf(textOf(heading)) };
  const m = first.value.match(/^(\d+)\.\s+/);
  if (m) first.value = first.value.slice(m[0].length);
  const plain = textOf(heading);
  const kind = kindOf(plain);
  const colon = first.value.indexOf(": ");
  let eyebrow = null;
  if (colon > 0 && kind !== "other") {
    eyebrow = first.value.slice(0, colon);
    const rest = first.value.slice(colon + 2);
    first.value = rest.charAt(0).toUpperCase() + rest.slice(1);
  }
  return { kind, eyebrow };
}

export default function remarkLearn({ base = "" } = {}) {
  return (tree, vfile) => {
    const file = vfile.path || (vfile.history && vfile.history[0]) || "";
    const isExercises = file.includes(`${path.sep}exercises${path.sep}`);

    // Widget links are rewritten inside widgetEmbed, so embed first, then fix every other link.
    visit(tree, "blockquote", (quote) => { widgetEmbed(quote, file, base); });
    visit(tree, "link", (link) => { link.url = rewriteHref(link.url, file, base); });
    // KaTeX's blackboard font has no digits; draw the indicator 1 in bold instead.
    visit(tree, ["math", "inlineMath"], (m) => { m.value = m.value.replace(/\\mathbb\{1\}/g, "\\mathbf{1}"); });
    visit(tree, "html", (h) => {
      h.value = h.value.replace(/<details>\s*<summary>([\d.]+)<\/summary>/g,
        (_, id) => `<details id="sol-${id}" class="solution"><summary><span>Solution</span> ${id}</summary>`);
    });

    // Group top-level content into <section>s at each ## heading (and problems at ### in exercises).
    const out = [];
    let section = null, problem = null;
    for (const node of tree.children) {
      if (node.type === "heading" && node.depth === 2) {
        const { kind, eyebrow } = splitHeading(node);
        if (eyebrow) node.data = { hProperties: { dataEyebrow: eyebrow } };
        section = { type: "section", data: { hName: "section", hProperties: { className: ["sec", `sec-${kind}`] } }, children: [node] };
        problem = null;
        out.push(section);
        continue;
      }
      if (isExercises && node.type === "heading" && node.depth === 3) {
        const m = textOf(node).match(/^(\d+\.\d+\.\d+)\s*(\S+)?/);
        if (m) {
          node.data = { hProperties: { id: `p-${m[1]}` } };
          problem = {
            type: "problem",
            data: { hName: "article", hProperties: { className: ["problem"], dataTier: m[2] || "" } },
            children: [node, { type: "html", value: `<a class="to-solution" href="#sol-${m[1]}">Solution ↓</a>` }],
          };
          out.push(problem);
          continue;
        }
      }
      if (node.type === "thematicBreak" || (node.type === "heading" && node.depth <= 2)) problem = null;
      if (problem) problem.children.push(node);
      else if (section) section.children.push(node);
      else out.push(node);
    }
    tree.children = out;
  };
}
