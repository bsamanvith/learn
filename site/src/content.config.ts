import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

// Every subject's chapters are read straight from subjects/<slug>/; the site never copies
// or rewrites them. Ids look like "probability/1.3-counting".
const id = ({ entry }: { entry: string }) => entry.replace(/\/(text|exercises)\//, "/").replace(/\.md$/, "");

const text = defineCollection({
  loader: glob({ pattern: "*/text/*.md", base: "../subjects", generateId: id }),
  schema: z.object({
    title: z.string(),
    part: z.number(),
    chapter: z.string(),
    prerequisites: z.array(z.string()).default([]),
    leads_to: z.array(z.string()).default([]),
    status: z.string().default("draft"),
  }),
});

const exercises = defineCollection({
  loader: glob({ pattern: "*/exercises/*.md", base: "../subjects", generateId: id }),
  schema: z.object({ title: z.string(), chapter: z.string() }),
});

// Written problem-book entries (hard problems and assignments): subjects/<slug>/problems/<id>.md.
// The full list, including planned ones, is subjects/<slug>/problems/index.json.
const problems = defineCollection({
  loader: glob({ pattern: "*/problems/*.md", base: "../subjects", generateId: ({ entry }) => entry.replace("/problems/", "/").replace(/\.md$/, "") }),
  schema: z.object({
    title: z.string(),
    kind: z.enum(["hard", "assignment"]),
    stars: z.number().min(1).max(4).optional(),
    unlocks: z.string().optional(),
    uses: z.array(z.string()).default([]),
    time: z.string().optional(),
    after: z.string().optional(),
  }),
});

export const collections = { text, exercises, problems };
