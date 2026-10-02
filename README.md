# Sujal Singh — personal website

The portfolio at https://sujalsingh.me, built with Astro and Markdown/MDX. The original design is preserved: light background, serif headings, subtle blue accents, and expandable project details.

## Local development

Use Node.js 24 (or a supported version >=22.12.0).

```sh
npm ci
npm run dev
```

Production check:

```sh
npm run build
npm run preview
```

## Publish with GitHub Pages

1. In this repository, open **Settings → Pages**.
2. Select **GitHub Actions** as the build/deployment source. Private repositories need an eligible GitHub paid plan for Pages; otherwise make this repository public if you want to publish its source publicly.
3. In **Actions → Deploy website**, run the workflow on `main` if it has not already run after your push.
4. In **Settings → Pages → Custom domain**, enter `sujalsingh.me` and save. Configure the custom domain in GitHub before changing DNS. The Astro configuration already uses this domain and root paths.
5. In GoDaddy, select **sujalsingh.me → DNS**. Replace the root parking/old-host A records with these four A records:

| Type | Name | Value |
| --- | --- | --- |
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |
| CNAME | www | sujalsin.github.io |

Use the default TTL. Do not change mail, nameserver, or unrelated verification records. Replace conflicting `www` records if present. If old root AAAA records exist, remove the old host's values or replace them with GitHub's documented IPv6 values.

The earlier ChatGPT Sites A records (`162.159.143.30` and `172.66.3.26`) are not used for this GitHub deployment. The `_openai-site-verification` and `_cf-custom-hostname` TXT records are not needed for GitHub Pages; their exact records can be removed if you previously added them solely for this site's setup.

6. When GitHub's DNS check and certificate finish, enable **Enforce HTTPS**. Certificate availability can take up to 24 hours.

The default project URL is `https://sujalsin.github.io/personal-site/`. This build is configured for the custom domain, so root-relative links are intended to work at `https://sujalsingh.me/`, not under the project subpath. Test with local preview until the custom domain is connected.

For extra domain protection, GitHub account **Settings → Pages → Add a domain** can generate a TXT ownership-verification record. Use the exact value GitHub provides; it cannot be precomputed.

## Add a blog post

1. Copy `templates/post.md` into `src/content/posts/your-post-slug.md`.
2. Fill in the title, description, and quoted `YYYY-MM-DD` date.
3. Write the post with Markdown. Use `.mdx` instead if you need embedded Astro components.
4. Change `draft: true` to `draft: false` when ready.
5. Commit and push to `main`. The workflow builds and publishes automatically.

Posts appear on the homepage, newest first, and at `/writing/your-post-slug/`. Drafts are excluded from the generated site. Draft source is still visible to anyone who can read the repository, so keep confidential drafts outside a public repository. There is no scheduled publication: a non-draft post is published on the next build, regardless of its date.

Put images in `public/images/` and use Markdown like `![Descriptive alt text](/images/result.png)`. Code fences are highlighted. Equation rendering is not installed yet; ordinary Markdown and MDX are ready.

## Edit the profile

- `src/pages/index.astro`: homepage/profile content.
- `public/style.css`: shared site and article styling.
- `src/layouts/PostLayout.astro`: article layout.
- `src/lib/posts.ts`: post discovery, sorting, and draft filtering.
- `.github/workflows/deploy.yml`: automatic deployment.

No Notion integration, newsletter service, analytics, or admin editor is configured.

## Official setup references

- https://docs.astro.build/en/guides/deploy/github/
- https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site
