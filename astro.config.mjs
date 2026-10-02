import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://sujalsingh.me',
  output: 'static',
  trailingSlash: 'always',
  integrations: [mdx()],
});
