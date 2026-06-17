export function contentSlug(entry: { id: string; slug?: string }) {
  return entry.slug ?? entry.id.replace(/\.(md|mdx)$/i, "");
}
