import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

// The chapters are read straight from subjects/; the site never copies or rewrites them.
const id = ({ entry }: { entry: string }) => entry.replace(/\.md$/, "");

const text = defineCollection({
  loader: glob({ pattern: "*.md", base: "../subjects/probability/text", generateId: id }),
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
  loader: glob({ pattern: "*.md", base: "../subjects/probability/exercises", generateId: id }),
  schema: z.object({ title: z.string(), chapter: z.string() }),
});

export const collections = { text, exercises };
