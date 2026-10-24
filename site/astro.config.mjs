import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkLearn from "./src/plugins/remark-learn.mjs";

// Served by GitHub Pages at https://bsamanvith.github.io/learn/. Override with BASE=/ for a root deploy.
const base = process.env.BASE ?? "/learn";

export default defineConfig({
  site: "https://bsamanvith.github.io",
  base,
  trailingSlash: "always",
  markdown: {
    remarkPlugins: [remarkMath, [remarkLearn, { base }]],
    rehypePlugins: [[rehypeKatex, { throwOnError: false, strict: "ignore" }]],
    shikiConfig: { themes: { light: "github-light", dark: "github-dark" }, defaultColor: false },
  },
  vite: {
    // Chapter sources and sims live outside site/.
    server: { fs: { allow: [".."] } },
  },
});
