import {themes as prismThemes} from 'prism-react-renderer';
import {writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const githubUrl = 'https://github.com/daisytuner/docc';
const siteUrl = 'https://docc.daisytuner.com';
// Keep search engines out until docc.daisytuner.com is live; build with DOCS_NOINDEX=false to allow indexing.
const noIndex = process.env.DOCS_NOINDEX !== 'false';

const config: Config = {
  title: 'docc',
  tagline: 'The Daisytuner Optimizing Compiler Collection',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: siteUrl,
  baseUrl: '/',
  trailingSlash: false,
  noIndex,

  onBrokenLinks: 'throw',
  markdown: {
    // .md is plain CommonMark so included READMEs aren't parsed as JSX; .mdx stays MDX.
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: `${githubUrl}/tree/main/docs-site/`,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    () => ({
      name: 'robots-txt',
      async postBuild({outDir}) {
        const robots = noIndex
          ? 'User-agent: *\nDisallow: /\n'
          : `User-agent: *\nAllow: /\n\nSitemap: ${siteUrl}/sitemap.xml\n`;
        await writeFile(join(outDir, 'robots.txt'), robots);
      },
    }),
  ],

  themes: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['en'],
        indexBlog: false,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'docc',
      logo: {
        alt: 'docc',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          // pathname:// bypasses the SPA router; Doxygen output is plain static HTML.
          href: 'pathname:///api/index.html',
          label: 'C++ API',
          position: 'left',
          target: '_self',
        },
        {
          href: githubUrl,
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Introduction', to: '/docs/intro'},
            {label: 'Install for C/C++', to: '/docs/getting-started-docc/install'},
            {label: 'Install for Python', to: '/docs/getting-started-python/install'},
            {label: 'C++ API', href: 'pathname:///api/index.html', target: '_self'},
          ],
        },
        {
          title: 'Daisytuner',
          items: [
            {label: 'Website', href: 'https://daisytuner.com'},
            {label: 'Daisytuner Docs', href: 'https://docs.daisytuner.com'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: githubUrl},
            {label: 'Discord', href: 'https://discord.gg/jpQ2h3f8jN'},
            {label: 'LinkedIn', href: 'https://www.linkedin.com/company/daisytuner/'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Daisytuner. Code and documentation are published under the BSD 3-Clause license.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['cmake'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
