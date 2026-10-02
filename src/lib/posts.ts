interface Frontmatter {
  title: string;
  description: string;
  date: string;
  draft?: boolean;
}
interface PostModule {
  frontmatter: Frontmatter;
  Content: any;
}
const modules = import.meta.glob<PostModule>('../content/posts/*.{md,mdx}', { eager: true });

export const posts = Object.entries(modules)
  .filter(([, entry]) => import.meta.env.DEV || entry.frontmatter.draft !== true)
  .map(([path, entry]) => {
    const slug = path.split('/').pop()!.replace(/\.(md|mdx)$/, '');
    const { title, description, date } = entry.frontmatter;
    if (!title || !description || !date || Number.isNaN(Date.parse(String(date)))) {
      throw new Error(`Post ${slug} needs a title, description, and valid date.`);
    }
    return { ...entry, slug };
  })
  .sort((a, b) => Date.parse(String(b.frontmatter.date)) - Date.parse(String(a.frontmatter.date)));

export function formatDate(date: string) {
  return new Date(date).toLocaleDateString('en-US', {
    month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC',
  });
}
