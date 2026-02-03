# MT-Aligner-Pipeline

A document alignment pipeline for building parallel corpora from multilingual document pairs. It takes documents in two languages (e.g., an English book and its Finnish translation), aligns them at both paragraph and sentence level using semantic embeddings and Dynamic Time Warping, and outputs aligned text chunks suitable for training machine translation models.

## How it works

The pipeline performs hierarchical alignment in two stages:

1. **Coarse alignment (paragraph-level)** — Embeds all paragraphs using a Sentence Transformers model (e.g., LaBSE), computes a cosine distance matrix, and applies Dynamic Time Warping (DTW) to find the optimal paragraph-to-paragraph mapping. This handles merged/split paragraphs across translations by supporting many-to-one and one-to-many mappings.

2. **Fine alignment (sentence-level)** — Within each aligned paragraph group, sentences are split (via Spacy), embedded, and aligned using a greedy fuzzy span matching algorithm that supports 1-to-1, 1-to-2, and 2-to-1 sentence alignments.

3. **Chunk building** — Aligned sentence pairs are grouped into output chunks of up to 2500 words per language side.

```
Input JSONL document pairs
    |
    v
Paragraph embedding (LaBSE / Sentence Transformers)
    |
    v
DTW paragraph alignment (approx or exact)
    |
    v
Path collapsing (group into paragraph ranges)
    |
    v
Sentence splitting (Spacy) + sentence embedding
    |
    v
Greedy span matching (sentence-level alignment)
    |
    v
Chunk building (word-limit aware)
    |
    v
Output: aligned parallel chunks (JSONL)
```

## Project structure

```
MT-aligner-pipeline/
├── run_pipeline.py              # CLI entry point
├── launcher.sh                  # SLURM job submission script
├── stats.py                     # Corpus statistics tool
├── requirements.txt
├── configs/
│   └── default.yaml             # Pipeline configuration
├── src/
│   ├── schema.py                # Pydantic data models
│   ├── input_handling.py        # Input format handling
│   ├── align/
│   │   ├── coarse_align.py      # Paragraph-level DTW alignment
│   │   ├── fine_align_spans.py  # Sentence-level span alignment
│   │   └── filters.py           # Alignment validation filters
│   ├── encode/
│   │   ├── embed_paragraphs.py  # Paragraph embedding
│   │   └── embed_sentences.py   # Sentence embedding
│   ├── pipeline/
│   │   ├── runner.py            # Pipeline orchestrator (multiprocessing)
│   │   ├── document_loader.py   # Document I/O and embedding cache
│   │   ├── aligner.py           # High-level alignment coordinator
│   │   └── output_builder.py    # Result formatting and saving
│   ├── postprocess/
│   │   └── rebuild.py           # Chunk building from alignments
│   └── utils/
│       ├── io.py                # JSON/embedding I/O
│       ├── sentence_split.py    # Spacy sentence tokenization
│       ├── collapse_path.py     # DTW path simplification
│       └── debug.py
└── data/
    ├── raw/                     # Input document pairs (JSONL)
    ├── preprocessed/            # Cached embeddings
    └── aligned/                 # Output aligned chunks
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
```

## Configuration

Edit `configs/default.yaml`:

```yaml
# Output directory (created if missing)
output: "data/aligned"

# Embedding model (any Sentence Transformers model)
model: "labse"
model_path: "/path/to/local/model"   # optional, for offline use

# DTW mode: "approx" (fast, O(N)) or "exact" (accurate, O(N*M))
dtw: "approx"

# Input
input: "data/raw"
input_format: "SingleDocument"
```

## Usage

### Local

```bash
python run_pipeline.py --config configs/default.yaml
```

### SLURM (HPC)

Edit `launcher.sh` to match your cluster setup, then:

```bash
sbatch launcher.sh
```

## Input format

The pipeline expects input as subdirectories under the input path, each containing JSONL files for the two languages of a document pair:

```
data/raw/
└── 1399_49487/
    ├── en.jsonl
    └── fi.jsonl
```

Each JSONL file contains one line per paragraph:

```json
{"doc_id": "anna_karenina.en", "para_id": 0, "lang": "en", "text": "Chapter 1"}
{"doc_id": "anna_karenina.en", "para_id": 1, "lang": "en", "text": "Happy families are all alike..."}
```

## Output format

For each document pair, the pipeline produces:

- `{name}_chunks.jsonl` — Aligned parallel text chunks (the main output):

```json
{
  "paragraph_pairs": [[[0], [0]], [[1, 2], [1]]],
  "src_text": "Chapter 1\n\nHappy families are all alike...",
  "tgt_text": "Luku 1\n\nOnnelliset perheet ovat kaikki samanlaisia...",
  "src_words": 2450,
  "tgt_words": 2380
}
```

- `{name}_meta.jsonl` — Alignment metadata (DTW path, sentence alignments, input paths).

## Multiprocessing

The pipeline supports parallel processing via Python `multiprocessing` (spawn mode). By default it uses up to 8 workers, bounded by available CPUs. Each worker loads its own copy of the embedding model, so GPU memory is the main constraint. Sequential mode is available and recommended for single-GPU setups — set `use_multiprocessing=False` in the runner.

## Dependencies

- [sentence-transformers](https://www.sbert.net/) — Multilingual embedding models
- [fastdtw](https://github.com/slaypni/fastdtw) — Fast approximate DTW
- [dtaidistance](https://github.com/wannesm/dtaidistance) — Exact DTW
- [Spacy](https://spacy.io/) + `xx_ent_wiki_sm` — Multilingual sentence splitting
- NumPy, PyYAML, tqdm
