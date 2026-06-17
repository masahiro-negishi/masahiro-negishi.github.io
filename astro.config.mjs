import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

export default defineConfig({
  site: "https://masahiro-negishi.github.io",
  integrations: [mdx()],
  output: "static",
  trailingSlash: "always",
});
