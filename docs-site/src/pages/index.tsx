import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

type Feature = {
  title: string;
  description: string;
  to: string;
};

const features: Feature[] = [
  {
    title: 'C/C++',
    description: 'Use docc as a drop-in replacement for clang and gcc, and offload to GPUs with one flag.',
    to: '/docs/getting-started-docc/install',
  },
  {
    title: 'Python & PyTorch',
    description: 'Compile NumPy functions with @native, or use docc as a torch.compile backend.',
    to: '/docs/getting-started-python/install',
  },
  {
    title: 'C++ API',
    description: 'Reference documentation for the SDFG IR, passes, transformations, and runtimes.',
    to: 'pathname:///api/index.html',
  },
];

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="Documentation" description={siteConfig.tagline}>
      <header className={styles.hero}>
        <div className="container">
          <Heading as="h1" className={styles.title}>
            {siteConfig.title}
          </Heading>
          <p className={styles.tagline}>{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link className="button button--primary button--lg" to="/docs/intro">
              Read the docs
            </Link>
            <Link
              className="button button--secondary button--lg"
              to="pathname:///api/index.html"
              target="_self">
              C++ API reference
            </Link>
          </div>
        </div>
      </header>
      <main className="container margin-vert--xl">
        <div className="row">
          {features.map((f) => (
            <div key={f.title} className="col col--4 margin-bottom--lg">
              <Link
                className={styles.card}
                to={f.to}
                target={f.to.startsWith('pathname:') ? '_self' : undefined}>
                <Heading as="h3">{f.title}</Heading>
                <p>{f.description}</p>
              </Link>
            </div>
          ))}
        </div>
      </main>
    </Layout>
  );
}
