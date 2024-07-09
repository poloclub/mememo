# MeMemo <a href="https://poloclub.github.io/mememo/"><img align="right" src="./examples/rag-playground/src/images/icon-logo.svg" height="38"></img></a>

[![build](https://github.com/poloclub/mememo/actions/workflows/build.yml/badge.svg)](https://github.com/poloclub/mememo/actions/workflows/build.yml)
[![npm](https://img.shields.io/npm/v/mememo?color=orange)](https://www.npmjs.com/package/mememo)
[![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/poloclub/mememo/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arXiv-2407.01972-red)](https://arxiv.org/abs/2407.01972)
[![DOI:10.1145/3543873.3587362](https://img.shields.io/badge/DOI-10.1145/3626772.3657662-blue)](https://doi.org/10.1145/3626772.3657662)

A JavaScript library that brings vector search and RAG to your browser!

<table>
  <tr>
    <td colspan="3"><a href="https://poloclub.github.io/mememo"><img src='https://i.imgur.com/4cDZQSz.png' width="100%"></a></td>
  </tr>
  <tr></tr>
  <tr>
     <td><a href="https://poloclub.github.io/mememo/?dataset=paper">🤖 ML Paper Reviewer</a></td>
     <td><a href="https://poloclub.github.io/mememo/?dataset=diffusiondb">🌠 Prompt Enhancer</a></td>
     <td><a href="https://poloclub.github.io/mememo/?dataset=accident">🌱 Responsible AI Assistant</a></td>
  </tr>
</table>

## What is MeMemo?

MeMemo is a JavaScript library that adapts the state-of-the-art approximate nearest neighbor search technique HNSW to browser environments.
Developed with modern and native Web technologies, such as IndexedDB and Web Workers, our toolkit leverages client-side hardware capabilities to enable researchers and developers to efficiently search through millions of high-dimensional vectors in browsers.
MeMemo enables exciting new design and research opportunities, such as private and personalized content creation and interactive prototyping, as demonstrated in our example application RAG Playground.✨

### Features

<video src="https://github.com/poloclub/mememo/assets/15007159/081ab670-a90a-464b-a9e6-e70b308314c9"></video>

## Getting Started

### Installation

MeMemo supports both browser and Node.js environments. To install MeMemo, you can use `npm`:

```bash
npm install mememo
```

### Vector Search and Storage in Browsers

Then, you can create a vector index and do an approximate nearest neighbor search through two functions:

```typescript
// Import the HNSW class from the MeMemo module
import { HNSW } from 'mememo';

// Creating a new index
const index = new HNSW({ distanceFunction: 'cosine' });

// Inserting elements into our index in batches
let keys: string[];
let values: number[][];
await index.bulkInsert(keys, values);

// Find k-nearest neighbors
let query: number[];
const { keys, distances } = await index.query(query, k);
```

## Developing MeMemo

Clone or download this repository:

```bash
git clone git@github.com:poloclub/mememo.git
```

Install the dependencies:

```bash
npm install
```

Use Vitest for unit testing:

```
npm run test
```

## Developing the RAG Playground Examples

Clone or download this repository:

```bash
git clone git@github.com:poloclub/mememo.git
```

Navigate to the example folder:

```bash
cd ./examples/rag-playground
```

Install the dependencies:

```bash
npm install
```

Then run Loan Explainer:

```
npm run dev
```

Navigate to localhost:3000. You should see three Explainers running in your browser :)

## Credits

MeMemo is created by <a href='https://zijie.wang/' target='_blank'>Jay Wang</a> and <a href='' target='_blank'>Polo Chau</a>.

## Citation

To learn more about MeMemo, check out our [research paper](https://arxiv.org/abs/2407.01972) published at SIGIR'24.

```bibtex
@inproceedings{wangMeMemoOndeviceRetrieval2024,
  title = {{{MeMemo}}: {{On-device Retrieval Augmentation}} for {{Private}} and {{Personalized Text Generation}}},
  booktitle = {Proceedings of the 47th {{International ACM SIGIR Conference}} on {{Research}} and {{Development}} in {{Information Retrieval}}},
  author = {Wang, Zijie J. and Chau, Duen Horng},
  year = {2024},
  urldate = {2024-06-26},
  langid = {english}
}
```

## License

The software is available under the [MIT License](https://github.com/poloclub/mememo/blob/main/LICENSE).

## Contact

If you have any questions, feel free to [open an issue](https://github.com/poloclub/mememo/issues/new) or contact [Jay Wang](https://zijie.wang).
