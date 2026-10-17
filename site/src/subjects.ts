// Subject-level data the chapter files don't carry, read from subjects/index.json and
// subjects/<slug>/meta.json. Chapters and exercises come from the content collections.
import { getCollection, type CollectionEntry } from "astro:content";
import index from "../../subjects/index.json";

type Part = { title: string; blurb: string };
export type SubjectMeta = { title: string; tagline: string; intro?: string; parts: Record<string, Part> };
export type Widget = { file: string; title: string; chapter: string; idea: string; status: string; notes: string };

const metas = import.meta.glob<SubjectMeta>("../../subjects/*/meta.json", { eager: true, import: "default" });
const registries = import.meta.glob<{ widgets: Widget[] }>("../../subjects/*/widgets/index.json", { eager: true, import: "default" });
const slugOf = (p: string) => p.split("/").at(-2) === "widgets" ? p.split("/").at(-3)! : p.split("/").at(-2)!;

export const SUBJECTS = index.subjects;
export const meta = (slug: string): SubjectMeta | undefined =>
  Object.entries(metas).find(([p]) => slugOf(p) === slug)?.[1];
export const widgetsOf = (slug: string): Widget[] =>
  Object.entries(registries).find(([p]) => slugOf(p) === slug)?.[1].widgets ?? [];
export const partTitle = (slug: string, part: number) => meta(slug)?.parts[String(part)]?.title ?? `Part ${part}`;

// Collection ids look like "probability/1.3-counting".
export const subjectOf = (id: string) => id.split("/")[0];
export const slugIn = (id: string) => id.split("/")[1];
export const chapterUrl = (id: string) => `/${id}/`;

// Sort "1.10" after "1.9".
export const byChapter = (a: { data: { chapter: string } }, b: { data: { chapter: string } }) => {
  const [a1, a2] = a.data.chapter.split(".").map(Number);
  const [b1, b2] = b.data.chapter.split(".").map(Number);
  return a1 - b1 || a2 - b2;
};

export async function chaptersOf(slug: string): Promise<CollectionEntry<"text">[]> {
  return (await getCollection("text", (e) => subjectOf(e.id) === slug)).sort(byChapter);
}

export async function liveSubjects() {
  const text = await getCollection("text");
  return SUBJECTS.filter((s) => text.some((e) => subjectOf(e.id) === s.slug));
}
