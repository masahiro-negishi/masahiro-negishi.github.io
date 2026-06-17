# Masahiro Negishi

Personal website for <https://masahiro-negishi.github.io>.

This site is a static Astro project. It replaced the previous Jekyll/al-folio
layout so the repository stays small and the UI can be maintained directly.

## Local Development

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

## Check Before Deploying

Use the production build preview before deploying:

```bash
npm run build
npm run preview
```

Open the local URL printed by Astro. This serves the generated `dist/` output,
so it is the closest local check to the deployed GitHub Pages site.

## Content

- Homepage content lives in `src/pages/index.astro`.
- Publication data lives in `src/data/publications.ts`.
- Public blog posts belong in `src/content/blog/` with `draft: false`.
- Web slide decks belong in `src/content/slides/` as MDX files.
- Old blog posts are archived in `archive/blog-posts/` and are not published.

## Deployment

GitHub Actions builds the site on pull requests and deploys GitHub Pages only
from pushes to `main` or manual workflow runs.
