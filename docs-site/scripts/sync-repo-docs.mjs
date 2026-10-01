// Copies docs that live elsewhere in the docc repo into docs/generated/ so Docusaurus can render them.
import {existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync} from 'node:fs';
import {dirname, join, posix, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const siteDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoDir = resolve(siteDir, '..');
const outDir = join(siteDir, 'docs', 'generated');
const githubUrl = 'https://github.com/daisytuner/docc';

const sources = [
  {
    src: 'python/README.md',
    id: 'python-build-from-source',
    slug: '/getting-started-python/build-from-source',
    title: 'Build from Source & Usage',
    description: 'How to build docc for Python from source and use the @native decorator',
  },
  {
    src: 'tutorial/printf_target/target_tutorial.md',
    id: 'custom-offload-targets',
    slug: '/structured-sdfgs/custom-offload-targets',
    title: 'Adding Custom Offload Targets',
    description: 'How to implement a custom offload target, using a Printf debug target as example',
  },
];

function rewriteLinks(markdown, srcFile) {
  // Relative links would point into the site; send them to the file on GitHub instead.
  return markdown.replace(/(!?)\[([^\]]*)\]\(([^)\s]+)\)/g, (match, bang, text, target) => {
    if (/^([a-z]+:|#|\/)/i.test(target)) return match;
    const [path, hash = ''] = target.split('#');
    const repoPath = posix.normalize(posix.join(posix.dirname(srcFile), path));
    const abs = join(repoDir, repoPath);
    if (!abs.startsWith(repoDir) || !existsSync(abs)) {
      throw new Error(`${srcFile}: broken relative link "${target}"`);
    }
    const kind = bang ? 'raw' : statSync(abs).isDirectory() ? 'tree' : 'blob';
    const url = bang
      ? `https://raw.githubusercontent.com/daisytuner/docc/main/${repoPath}`
      : `${githubUrl}/${kind}/main/${repoPath}`;
    return `${bang}[${text}](${url}${hash ? `#${hash}` : ''})`;
  });
}

rmSync(outDir, {recursive: true, force: true});
mkdirSync(outDir, {recursive: true});

for (const {src, id, slug, title, description} of sources) {
  const body = readFileSync(join(repoDir, src), 'utf8').replace(/^\s*# .*\n/, '');
  const frontMatter = [
    '---',
    `id: ${id}`,
    `slug: ${slug}`,
    `title: ${JSON.stringify(title)}`,
    `description: ${JSON.stringify(description)}`,
    `custom_edit_url: ${githubUrl}/edit/main/${src}`,
    '---',
  ].join('\n');
  const note = `:::note\n\nThis page is generated from [\`${src}\`](${githubUrl}/blob/main/${src}) in the docc repository.\n\n:::\n`;
  writeFileSync(join(outDir, `${id}.md`), `${frontMatter}\n\n${note}\n${rewriteLinks(body, src)}`);
  console.log(`synced ${src} -> ${relative(siteDir, join(outDir, `${id}.md`))}`);
}
