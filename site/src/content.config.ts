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

export const collections = { text, exercises };
