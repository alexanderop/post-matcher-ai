import { pipeline, FeatureExtractionPipeline } from '@huggingface/transformers';
import chalk from 'chalk';
import fs from 'fs';
import { glob } from 'glob';
import matter from 'gray-matter';
import { remark } from 'remark';
import strip from 'strip-markdown';
import path from 'path';

// --------- Configurations ---------
const GLOB = 'src/content/**/*.{md,mdx}';              // Where to find Markdown content
const OUT = 'src/assets/similarities.json';             // Output file for results
const TOP_N = 5;                                        // Number of similar docs to keep
const MODEL = 'Snowflake/snowflake-arctic-embed-m-v2.0';// Embedding model

// --------- Type Definitions ---------
interface Frontmatter { slug: string; [k: string]: unknown }
interface Document { path: string; content: string; frontmatter: Frontmatter }
interface SimilarityResult extends Frontmatter { path: string; similarity: number }

// --------- Utils ---------

/**
 * Normalizes a vector to unit length (L2 norm == 1)
 * This makes cosine similarity a simple dot product!
 */
function normalize(vec: Float32Array): Float32Array {
  let len = Math.hypot(...vec);         // L2 norm
  if (!len) return vec;
  return new Float32Array(vec.map(x => x / len));
}

/**
 * Computes dot product of two same-length vectors.
 * Vectors MUST be normalized before using this for cosine similarity!
 */
const dot = (a: Float32Array, b: Float32Array) => a.reduce((sum, ai, i) => sum + ai * b[i], 0);

/**
 * Strips markdown formatting, import/export lines, headings, tables, etc.
 * Returns plain text for semantic analysis.
 */
const getPlainText = async (md: string) => {
  let txt = String(await remark().use(strip).process(md))
    .replace(/^import .*?$/gm, '')
    .replace(/^export .*?$/gm, '')
    .replace(/^\s*(TLDR|Introduction|Conclusion|Summary|Quick Setup Guide|Rules?)\s*$/gim, '')
    .replace(/^[A-Z\s]{4,}$/gm, '')
    .replace(/^\|.*\|$/gm, '')
    .replace(/(Rule\s\d+:.*)(?=\s*Rule\s\d+:)/g, '$1\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/\n{2}/g, '\n\n')
    .replace(/\n/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
  return txt;
};

/**
 * Parses and validates a single Markdown file.
 * - Extracts frontmatter (slug, etc.)
 * - Converts content to plain text
 * - Skips drafts or files with no slug
 */
async function processFile(path: string): Promise<Document | null> {
  try {
    const { content, data } = matter(fs.readFileSync(path, 'utf-8'));
    if (!data.slug || data.draft) return null;
    const plain = await getPlainText(content);
    return { path, content: plain, frontmatter: data as Frontmatter };
  } catch { return null; }
}

/**
 * Processes an array of Markdown file paths into Documents
 */
async function loadDocs(paths: string[]) {
  const docs: Document[] = [];
  for (const p of paths) {
    const d = await processFile(p);
    if (d) docs.push(d);
  }
  return docs;
}

/**
 * Generates vector embeddings for each document's plain text.
 * - Uses HuggingFace model
 * - Normalizes each vector for fast cosine similarity search
 */
async function embedDocs(docs: Document[], extractor: FeatureExtractionPipeline) {
  if (!docs.length) return [];
  // Don't let the model normalize, we do it manually for safety
  const res = await extractor(docs.map(d => d.content), { pooling: 'mean', normalize: false }) as any;
  const [n, dim] = res.dims;
  // Each embedding vector is normalized for performance
  return Array.from({ length: n }, (_, i) => normalize(res.data.slice(i * dim, (i + 1) * dim)));
}

/**
 * Computes the top-N most similar documents for the given document index.
 * - Uses dot product of normalized vectors for cosine similarity
 * - Returns only the top-N
 */
function topSimilar(idx: number, docs: Document[], embs: Float32Array[], n: number): SimilarityResult[] {
  return docs.map((d, j) => j === idx ? null : ({
    ...d.frontmatter, path: d.path,
    similarity: +dot(embs[idx], embs[j]).toFixed(2) // higher = more similar
  }))
    .filter(Boolean)
    .sort((a, b) => (b as any).similarity - (a as any).similarity)
    .slice(0, n) as SimilarityResult[];
}

/**
 * Computes all similarities for every document, returns as {slug: SimilarityResult[]} map.
 */
function allSimilarities(docs: Document[], embs: Float32Array[], n: number) {
  return Object.fromEntries(docs.map((d, i) => [d.frontmatter.slug, topSimilar(i, docs, embs, n)]));
}

/**
 * Saves result object as JSON file.
 * - Ensures output directory exists.
 */
async function saveJson(obj: any, out: string) {
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(obj, null, 2));
}

// --------- Main Execution Flow ---------
async function main() {
  try {
    // 1. Load transformer model for embeddings
    const extractor = await pipeline('feature-extraction', MODEL) as FeatureExtractionPipeline;

    // 2. Find all Markdown files
    const files = await glob(GLOB);
    if (!files.length) return console.log(chalk.yellow('No content files found.'));

    // 3. Parse and process all files
    const docs = await loadDocs(files);
    if (!docs.length) return console.log(chalk.red('No documents loaded.'));

    // 4. Generate & normalize embeddings
    const embs = await embedDocs(docs, extractor);
    if (!embs.length) return console.log(chalk.red('No embeddings.'));

    // 5. Calculate similarities for each doc
    const results = allSimilarities(docs, embs, TOP_N);

    // 6. Save results to disk
    await saveJson(results, OUT);
    console.log(chalk.green(`Similarity results saved to ${OUT}`));
  } catch (e) {
    console.error(chalk.red('Error:'), e);
    process.exitCode = 1;
  }
}

main();
