import { pipeline, FeatureExtractionPipeline } from '@huggingface/transformers';
import chalk from 'chalk';
import fs from 'fs';
import { glob } from 'glob';
import matter from 'gray-matter';
import { remark } from 'remark';
import strip from 'strip-markdown';
import path from 'path';

const CONTENT_GLOB_PATTERN = 'src/content/**/*.{md,mdx}';
const SIMILARITIES_OUTPUT_PATH = 'src/assets/similarities.json';
const TOP_N_RESULTS = 5;
const FEATURE_EXTRACTOR_MODEL_NAME = 'Snowflake/snowflake-arctic-embed-m-v2.0';

interface Frontmatter {
  slug: string;
  // Add other known/expected frontmatter properties here for better type safety
  // e.g., title?: string; date?: string;
  [key: string]: unknown; // Use unknown for other dynamic properties
}

interface Document {
  path: string;
  content: string; // Plain text content after strip-markdown
  frontmatter: Frontmatter;
}

// SimilarityResult includes properties from the target document's frontmatter,
// plus the path of the similar document and the similarity score.
interface SimilarityResult extends Frontmatter {
  path: string;
  similarity: number;
}

// Type for the Hugging Face feature-extraction pipeline result's relevant parts
interface EmbeddingOutput {
  dims: number[]; // [numberOfDocuments, embeddingDimension]
  data: Float32Array;   // Flattened array of all embeddings
}

// --- Global Extractor ---
// To be initialized asynchronously in the main execution flow
let featureExtractor: FeatureExtractionPipeline;

async function runSimilarityAnalysis() {
    try {
      featureExtractor = await initializeFeatureExtractor();
  
      const filePaths = await glob(CONTENT_GLOB_PATTERN);
      console.log(chalk.cyan(`Found ${filePaths.length} content files matching pattern: ${CONTENT_GLOB_PATTERN}`));
  
      if (filePaths.length === 0) {
        console.log(chalk.yellow.bold('No content files found. Exiting.'));
        return;
      }
  
      const documents = await loadAllDocuments(filePaths);
      if (documents.length === 0) {
        console.log(chalk.red.bold('No documents were successfully processed. Exiting.'));
        return;
      }
  
      const embeddings = await generateDocumentEmbeddings(documents, featureExtractor);
      if (embeddings.length === 0 && documents.length > 0) {
        // This condition implies an issue in embedding generation if documents were present
        console.log(chalk.red.bold('Failed to generate embeddings for processed documents. Exiting.'));
        return;
      }
       if (embeddings.length === 0 && documents.length == 0) {
        console.log(chalk.yellow.bold('No documents to process, so no embeddings generated. Exiting.'));
        return;
      }
  
  
      const similarityResults = computeAllSimilarities(documents, embeddings, TOP_N_RESULTS);
  
      console.log(chalk.green.bold('Final Results (Top Similar Documents for Each):'));
      console.log(JSON.stringify(similarityResults, null, 2));
  
      await saveResultsToFile(similarityResults, SIMILARITIES_OUTPUT_PATH);
  
    } catch (error) {
      console.error(chalk.red.bold('An unhandled error occurred during the similarity analysis:'), error);
      process.exitCode = 1; // Indicate failure to the shell
    }
}


async function getPlainText(markdownContent: string): Promise<string> {
    const processed = await remark().use(strip).process(markdownContent);
    let text = String(processed);
  
    // Remove import/export lines (they often sneak in at top)
    text = text.replace(/^import .*?$/gm, '');
    text = text.replace(/^export .*?$/gm, '');
  
    // Remove headings like "Introduction", "TLDR", etc.
    text = text.replace(/^\s*(TLDR|Introduction|Conclusion|Summary|Quick Setup Guide|Rules?)\s*$/gim, '');
  
    // Remove headings in all-caps (common for section labels)
    text = text.replace(/^[A-Z\s]{4,}$/gm, '');
  
    // Remove Markdown tables (if any remain)
    text = text.replace(/^\|.*\|$/gm, '');
  
    // Collapse repeated section markers (e.g. multiple Rule headings)
    text = text.replace(/(Rule\s\d+:.*)(?=\s*Rule\s\d+:)/g, '$1\n');
  
    // Normalize line breaks
    text = text
      .replace(/\n{3,}/g, '\n\n')   // excessive breaks to paragraph breaks
      .replace(/\n{2}/g, '\n\n')    // keep paragraph spacing
      .replace(/\n/g, ' ')          // all other line breaks to space
      .replace(/\s{2,}/g, ' ')      // collapse whitespace
      .trim();
  
    return text;
  }

function cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
  let dotProduct = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
  }
  return dotProduct;
}

// --- Core Logic Functions ---
async function processDocumentFile(filePath: string): Promise<Document | null> {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const { content: markdownContent, data: parsedFrontmatter } = matter(fileContent);

    if (typeof parsedFrontmatter.slug !== 'string' || !parsedFrontmatter.slug) {
      console.warn(chalk.yellow(`Document at ${filePath} is missing a valid slug. Skipping.`));
      return null;
    }

    if (parsedFrontmatter.draft === true) {
      console.warn(chalk.yellow(`Document at ${filePath} is a draft. Skipping.`));
      return null;
    }

    const plainText = await getPlainText(markdownContent);
    return {
      path: filePath,
      content: plainText,
      frontmatter: parsedFrontmatter as Frontmatter, // Asserting shape after validation
    };
  } catch (error) {
    console.error(chalk.red(`Error processing file ${filePath}:`), error);
    return null;
  }
}

async function loadAllDocuments(filePaths: string[]): Promise<Document[]> {
  const documents: Document[] = [];
  console.log(chalk.blue(`Processing ${filePaths.length} document files...`));
  for (const filePath of filePaths) {
    const doc = await processDocumentFile(filePath);
    if (doc) {
      documents.push(doc);
      console.log(chalk.gray(`Successfully processed and added: ${filePath}`));
    }
  }
  console.log(chalk.green(`Successfully loaded ${documents.length} documents.`));
  return documents;
}

async function generateDocumentEmbeddings(
  docs: Document[],
  extractor: FeatureExtractionPipeline
): Promise<Float32Array[]> {
  console.log(chalk.blue(`Generating embeddings for ${docs.length} documents...`));
  if (docs.length === 0) {
    console.log(chalk.yellow('No documents to generate embeddings for.'));
    return [];
  }
  const documentTexts = docs.map(doc => doc.content);

  const embeddingsOutput = (await extractor(documentTexts, {
    pooling: 'mean',
    normalize: true,
  })) as unknown as EmbeddingOutput; // Cast to unknown first, then to EmbeddingOutput

  console.log(chalk.green('Embeddings generated.'));

  // Add a check for the dimensions length
  if (!embeddingsOutput.dims || embeddingsOutput.dims.length < 2) {
    console.error(chalk.red('Embeddings output does not have the expected dimensions.'));
    return [];
  }

  const numDocsOutput = embeddingsOutput.dims[0];
  const embeddingDim = embeddingsOutput.dims[1];

  if (numDocsOutput !== docs.length) {
    console.error(chalk.red(`Mismatch between number of documents (${docs.length}) and embeddings generated (${numDocsOutput}).`));
    // Potentially throw an error here or return empty, depending on desired robustness
    return [];
  }

  const allEmbeddings: Float32Array[] = [];
  for (let i = 0; i < numDocsOutput; i++) {
    allEmbeddings.push(
      embeddingsOutput.data.slice(i * embeddingDim, (i + 1) * embeddingDim) as Float32Array
    );
  }
  return allEmbeddings;
}

function calculateSingleDocSimilarities(
  sourceDocIndex: number,
  allDocuments: Document[],
  allEmbeddings: Float32Array[],
  topN: number
): SimilarityResult[] {
  const sourceEmbedding = allEmbeddings[sourceDocIndex];
  const similarities: SimilarityResult[] = [];

  for (let j = 0; j < allDocuments.length; j++) {
    if (sourceDocIndex === j) continue; // Don't compare a document with itself

    const targetDoc = allDocuments[j];
    const targetEmbedding = allEmbeddings[j];
    const similarityScore = cosineSimilarity(sourceEmbedding, targetEmbedding);

    const similarityEntry: SimilarityResult = {
      ...targetDoc.frontmatter, // Spread frontmatter of the target document
      path: targetDoc.path,     // Override path to be the target document's path
      similarity: parseFloat(similarityScore.toFixed(2)), // Add similarity score
    };
    similarities.push(similarityEntry);
  }

  similarities.sort((a, b) => b.similarity - a.similarity);
  return similarities.slice(0, topN);
}

function computeAllSimilarities(
  documents: Document[],
  embeddings: Float32Array[],
  topN: number
): Record<string, SimilarityResult[]> {
  const resultsBySlug: Record<string, SimilarityResult[]> = {};
  console.log(chalk.blue(`Calculating similarities for ${documents.length} documents...`));

  documents.forEach((doc, i) => {
    const sourceSlug = doc.frontmatter.slug;
    const topSimilarDocs = calculateSingleDocSimilarities(i, documents, embeddings, topN);
    resultsBySlug[sourceSlug] = topSimilarDocs;
    console.log(chalk.gray(`Processed similarities for: ${sourceSlug} (${i + 1}/${documents.length})`));
  });

  console.log(chalk.green('All similarity calculations complete.'));
  return resultsBySlug;
}

async function saveResultsToFile(
  results: Record<string, SimilarityResult[]>,
  outputFilePath: string
): Promise<void> {
  console.log(chalk.blue(`Attempting to save similarity results to ${outputFilePath}...`));
  try {
    const outputDir = path.dirname(outputFilePath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
      console.log(chalk.gray(`Created directory: ${outputDir}`));
    }
    fs.writeFileSync(outputFilePath, JSON.stringify(results, null, 2));
    console.log(chalk.green.bold(`Successfully saved similarity results to: ${outputFilePath}`));
  } catch (error) {
    console.error(chalk.red.bold(`Error writing similarity results to file ${outputFilePath}:`), error);
    throw error; // Re-throw to be caught by the main error handler
  }
}

async function initializeFeatureExtractor(): Promise<FeatureExtractionPipeline> {
  console.log(chalk.blue(`Initializing feature extractor model (${FEATURE_EXTRACTOR_MODEL_NAME})...`));
  const extractorInstance = await pipeline('feature-extraction', FEATURE_EXTRACTOR_MODEL_NAME);
  console.log(chalk.green('Feature extractor initialized successfully.'));
  return extractorInstance;
}



runSimilarityAnalysis();


