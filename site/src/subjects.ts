// Subject-level data the chapter files don't carry, read from subjects/index.json and
// subjects/<slug>/meta.json. A subject's chapters come from two places:
//   - markdown chapters in subjects/<slug>/text/ (the "text" content collection), and
//   - hand-written HTML lessons in subjects/<slug>/lessons/, listed in lessons.json.
// Both become the same Entry, so navigation, subject pages and the home page treat them alike.
import { execFileSync } from "node:child_process";
import path from "node:path";
import { getCollection } from "astro:content";
import index from "../../subjects/index.json";

type Part = { title: string; blurb: string; label?: string };
export type SubjectMeta = { title: string; tagline: string; intro?: string; partLabel?: string; parts: Record<string, Part> };
export type Widget = { file: string; title: string; chapter: string; idea: string; status: string; notes: string };
type LessonInfo = { file: string; chapter: string; part: number; title: string; added?: string; prerequisites?: string[]; leads_to?: string[]; blurb?: string };
type PageInfo = { file: string; title: string; added?: string; idea?: string };
type Lessons = { lessons?: LessonInfo[]; pages?: PageInfo[] };

export type Entry = {
  id: string;          // "probability/1.3-counting"
  subject: string;
  slug: string;        // "1.3-counting"
  chapter: string;
  title: string;
  part: number;
  prerequisites: string[];
  leads_to: string[];
  status: string;
  kind: "md" | "html";
  added: string;       // YYYY-MM-DD
  blurb?: string;
};

const metas = import.meta.glob<SubjectMeta>("../../subjects/*/meta.json", { eager: true, import: "default" });
const registries = import.meta.glob<{ widgets: Widget[] }>("../../subjects/*/widgets/index.json", { eager: true, import: "default" });
const lessonLists = import.meta.glob<Lessons>("../../subjects/*/lessons.json", { eager: true, import: "default" });
const lessonHtml = import.meta.glob<string>("../../subjects/*/lessons/*.html", { eager: true, query: "?raw", import: "default" });
const pageHtml = import.meta.glob<string>("../../subjects/*/pages/*.html", { eager: true, query: "?raw", import: "default" });
const figureSvg = import.meta.glob<string>("../../subjects/*/figures/*.svg", { eager: true, query: "?raw", import: "default" });
const slugOf = (p: string) => {
  const parts = p.split("/");
  const i = parts.lastIndexOf("subjects");
  return parts[i + 1];
};
const forSubject = <T,>(files: Record<string, T>, slug: string) =>
  Object.entries(files).find(([p]) => slugOf(p) === slug)?.[1];

// ---------- URLs ----------
// The site is served from a sub-path on GitHub Pages (/learn/). Every internal link goes through u().
const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
export const u = (p: string) => BASE + (p.startsWith("/") ? p : `/${p}`);
// Hand-written HTML is authored with links from the site root ("/machine-learning/..."); prefix them.
export const withBase = (html: string) => html.replace(/(href|src|data-src)="\/(?!\/)/g, `$1="${BASE}/`);

export const SUBJECTS = index.subjects;
export const meta = (slug: string): SubjectMeta | undefined => forSubject(metas, slug);
export const widgetsOf = (slug: string): Widget[] => forSubject(registries, slug)?.widgets ?? [];
export const pagesOf = (slug: string): PageInfo[] => forSubject(lessonLists, slug)?.pages ?? [];
export const partLabel = (slug: string) => meta(slug)?.partLabel ?? "Part";
// "Stage 3", or a part's own label such as "Toolkit".
export const partName = (slug: string, part: number) => meta(slug)?.parts[String(part)]?.label ?? `${partLabel(slug)} ${part}`;
export const partTitle = (slug: string, part: number) => meta(slug)?.parts[String(part)]?.title ?? `${partLabel(slug)} ${part}`;
export const subjectTitle = (slug: string) => meta(slug)?.title ?? SUBJECTS.find((s) => s.slug === slug)?.title ?? slug;

// Collection ids look like "probability/1.3-counting".
export const subjectOf = (id: string) => id.split("/")[0];
export const slugIn = (id: string) => id.split("/")[1];
export const chapterUrl = (id: string) => u(`/${id}/`);

// Sort "1.10" after "1.9", "02" after "01", and lettered chapters ("M1", the math toolkit) after numbered ones.
const chapterKey = (c: string) => {
  const m = c.match(/^([A-Za-z]*)(.*)$/)!;
  return { prefix: m[1], nums: m[2].split(".").map(Number) };
};
export const byChapter = (a: { chapter: string; part?: number }, b: { chapter: string; part?: number }) => {
  if (a.part !== undefined && b.part !== undefined && a.part !== b.part) return a.part - b.part;
  const ka = chapterKey(a.chapter), kb = chapterKey(b.chapter);
  if (ka.prefix !== kb.prefix) return ka.prefix === "" ? -1 : kb.prefix === "" ? 1 : ka.prefix.localeCompare(kb.prefix);
  for (let i = 0; i < Math.max(ka.nums.length, kb.nums.length); i++) {
    const d = (ka.nums[i] ?? 0) - (kb.nums[i] ?? 0);
    if (d) return d;
  }
  return 0;
};

// The date a file first entered git history, so "Recently added" needs no bookkeeping.
const today = new Date().toISOString().slice(0, 10);
const addedCache = new Map<string, string>();
function addedDate(file: string): string {
  if (!addedCache.has(file)) {
    let d = "";
    try {
      const out = execFileSync("git", ["log", "--follow", "--diff-filter=A", "--format=%cs", "--", file], { cwd: path.dirname(file), stdio: ["ignore", "pipe", "ignore"] });
      d = out.toString().trim().split("\n").filter(Boolean).at(-1) ?? "";
    } catch {}
    addedCache.set(file, d || today);
  }
  return addedCache.get(file)!;
}

let all: Entry[] | undefined;
export async function entries(): Promise<Entry[]> {
  if (all) return all;
  const md: Entry[] = (await getCollection("text")).map((e) => ({
    id: e.id, subject: subjectOf(e.id), slug: slugIn(e.id), kind: "md",
    chapter: e.data.chapter, title: e.data.title, part: e.data.part,
    prerequisites: e.data.prerequisites, leads_to: e.data.leads_to, status: e.data.status,
    added: e.filePath ? addedDate(path.resolve(e.filePath)) : today,
  }));
  const html: Entry[] = Object.entries(lessonLists).flatMap(([p, list]) =>
    (list.lessons ?? []).map((l) => ({
      id: `${slugOf(p)}/${l.file}`, subject: slugOf(p), slug: l.file, kind: "html" as const,
      chapter: l.chapter, title: l.title, part: l.part,
      prerequisites: l.prerequisites ?? [], leads_to: l.leads_to ?? [], status: "published",
      added: l.added ?? today, blurb: l.blurb,
    })));
  all = [...md, ...html];
  return all;
}

export async function chaptersOf(slug: string): Promise<Entry[]> {
  return (await entries()).filter((e) => e.subject === slug).sort(byChapter);
}

export async function liveSubjects() {
  const es = await entries();
  return SUBJECTS.filter((s) => es.some((e) => e.subject === s.slug));
}

// Figures are SVG files in subjects/<s>/figures/, pulled inline with <!-- figure: name --> so they
// inherit the page's colours (they draw with currentColor and the theme's CSS variables).
const withFigures = (html: string, subject: string) =>
  html.replace(/<!--\s*figure:\s*([\w-]+)\s*-->/g, (all, name) =>
    Object.entries(figureSvg).find(([p]) => p.endsWith(`/subjects/${subject}/figures/${name}.svg`))?.[1]
      .replace(/<\?xml[^>]*>\s*/, "") ?? all);
const fragment = (files: Record<string, string>, subject: string, dir: string, file: string) =>
  withBase(withFigures(Object.entries(files).find(([p]) => p.endsWith(`/subjects/${subject}/${dir}/${file}.html`))?.[1] ?? "", subject));
export const lessonBody = (id: string) => fragment(lessonHtml, subjectOf(id), "lessons", slugIn(id));
export const pageBody = (subject: string, file: string) => fragment(pageHtml, subject, "pages", file);

// Section headings of a hand-written lesson, for its "On this page" list.
export const lessonHeadings = (html: string) =>
  [...html.matchAll(/<section id="([^"]+)"[^>]*>\s*<h2[^>]*>([\s\S]*?)<\/h2>/g)]
    .map(([, slug, text]) => ({ depth: 2, slug, text: text.replace(/<[^>]+>/g, "").trim() }));

// Newest first, across every subject: chapters, lessons and hand-written pages.
export async function recentlyAdded(limit = 8) {
  const es = await entries();
  const items = [
    ...es.map((e) => ({ date: e.added, subject: e.subject, href: chapterUrl(e.id), label: `${e.chapter} · ${e.title}` })),
    ...SUBJECTS.flatMap((s) => pagesOf(s.slug).map((p) => ({ date: p.added ?? today, subject: s.slug, href: u(`/${s.slug}/${p.file}/`), label: p.title }))),
  ];
  return items.sort((a, b) => b.date.localeCompare(a.date) || b.label.localeCompare(a.label)).slice(0, limit);
}
