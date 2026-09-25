import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started (C/C++)',
      collapsed: false,
      items: [
        'getting-started-docc/install',
        'getting-started-docc/first-program',
        'getting-started-docc/libraries',
      ],
    },
    {
      type: 'category',
      label: 'Getting Started (Python)',
      collapsed: false,
      items: ['getting-started-python/install', 'generated/python-build-from-source'],
    },
    {
      type: 'category',
      label: 'Getting Started (PyTorch)',
      items: ['getting-started-pytorch/usage'],
    },
    {
      type: 'category',
      label: 'Using docc for C/C++',
      items: [
        'using-docc/commandline',
        'using-docc/targets',
        'using-docc/runtime-instrumentation',
        'using-docc/plugins',
      ],
    },
    {
      type: 'category',
      label: 'Working with Structured SDFGs',
      items: ['structured-sdfgs/overview', 'generated/custom-offload-targets'],
    },
    {
      type: 'category',
      label: 'Reference',
      items: ['reference/api', 'reference/citing'],
    },
  ],
};

export default sidebars;
