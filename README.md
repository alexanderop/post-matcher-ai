# post-matcher-ai

**post-matcher** is a Node.js/TypeScript tool that analyzes a collection of Markdown documents, computes semantic similarities between them using state-of-the-art embeddings from Hugging Face Transformers, and outputs the top similar documents for each entry. This is useful for content recommendation, clustering, or knowledge base navigation.

## Features

- **Automatic Markdown Parsing:** Reads all `.md` and `.mdx` files in `src/content/`.
- **Frontmatter Support:** Extracts metadata (slug, title, date, etc.) from each document.
- **Semantic Embedding:** Uses the `Snowflake/snowflake-arctic-embed-m-v2.0` model via Hugging Face Transformers to generate embeddings for each document.
- **Similarity Calculation:** Computes cosine similarity between all document pairs.
- **Top-N Recommendations:** For each document, finds the top 5 most similar documents.
- **JSON Output:** Saves results to `src/assets/similarities.json` for further use.

## Example Output

The output is a JSON object mapping each document's slug to an array of its most similar documents, e.g.:

```json
{
  "vue-introduction": [
    {
      "slug": "typescript-advanced-types",
      "title": "Advanced Types in TypeScript",
      "date": "2024-06-03T00:00:00.000Z",
      "path": "src/content/typescript-advanced-types.md",
      "similarity": 0.35
    },
    ...
  ],
  ...
}
```

## Project Structure

```
src/
  postMatcher.ts         # Main script for similarity analysis
  content/               # Markdown documents to analyze
  assets/
    similarities.json    # Output: similarity results
```

## Getting Started

### Prerequisites

- Node.js (v18+ recommended)
- npm

### Installation

1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd post-matcher
   ```

2. Install dependencies:
   ```sh
   npm install
   ```

### Usage

To run the similarity analysis:

```sh
npm run postMatcher
```

This will process all Markdown files in `src/content/`, compute similarities, and write the results to `src/assets/similarities.json`.

### Adding Content

- Place your Markdown (`.md` or `.mdx`) files in `src/content/`.
- Each file should have a frontmatter section with at least a `slug` (unique identifier):

  ```markdown
  ---
  slug: my-unique-slug
  title: My Document Title
  date: 2024-06-01
  ---
  ```

### Configuration

- **Model:** The embedding model is set to `Snowflake/snowflake-arctic-embed-m-v2.0` in the code.
- **Top N Results:** Change the `TOP_N_RESULTS` constant in `src/postMatcher.ts` to adjust how many similar documents are returned per file.

## Development

- TypeScript configuration is in `tsconfig.json`.
- Main logic is in `src/postMatcher.ts`.
- Uses `ts-node` for running TypeScript directly.

## Dependencies

- [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers)
- [`chalk`](https://www.npmjs.com/package/chalk)
- [`glob`](https://www.npmjs.com/package/glob)
- [`gray-matter`](https://www.npmjs.com/package/gray-matter)
- [`remark`](https://www.npmjs.com/package/remark)
- [`strip-markdown`](https://www.npmjs.com/package/strip-markdown)
- [`ts-node`](https://www.npmjs.com/package/ts-node)
- [`typescript`](https://www.npmjs.com/package/typescript)

## Example Content

Some example Markdown files are included in `src/content/`:

- `vue.md` — Introduction to Vue.js
- `animals.md` — The Fascinating World of Animals
- `cat.md` — All About Cats
- `typescript-advanced-types.md` — Advanced Types in TypeScript
- `typescript-generics.md` — Generics in TypeScript

## License

ISC
