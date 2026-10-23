import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkLearn from "./src/plugins/remark-learn.mjs";

export default defineConfig({
  trailingSlash: "always",
  markdown: {
    remarkPlugins: [remarkMath, remarkLearn],
    rehypePlugins: [[rehypeKatex, { throwOnError: false, strict: "ignore" }]],
    shikiConfig: { themes: { light: "github-light", dark: "github-dark" }, defaultColor: false },
  },
  vite: {
    // Chapter sources and sims live outside site/.
    server: { fs: { allow: [".."] } },
  },
});
