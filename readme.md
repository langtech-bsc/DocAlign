# DocAlign

A document alignment pipeline for building parallel corpora from multilingual document pairs. It takes documents in two languages (e.g., an English book and its Finnish translation), aligns them at both paragraph and sentence level using semantic embeddings and Dynamic Time Warping, and outputs aligned text chunks suitable for training machine translation models.

## How it works

The pipeline performs hierarchical alignment in two stages:

1. **Coarse alignment (paragraph-level)** — Embeds all paragraphs using a Sentence Transformers model (e.g., LaBSE), computes a cosine distance matrix, and applies Dynamic Time Warping (DTW) to find the optimal paragraph-to-paragraph mapping. This handles merged/split paragraphs across translations by supporting many-to-one and one-to-many mappings. Paragraph groups with low similarity are filtered out before proceeding.

2. **Fine alignment (sentence-level)** — Within each aligned paragraph group, sentences are split (via Spacy), embedded, and aligned using a greedy fuzzy span matching algorithm that supports 1-to-1, 1-to-2, and 2-to-1 sentence alignments.

3. **Chunk building** — Aligned sentence pairs are grouped into output chunks of up to 2500 words per language side. Chunks with low average alignment scores can be filtered out.

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
Paragraph similarity filtering (drop non-parallel groups)
    |
    v
Sentence splitting (Spacy) + sentence embedding
    |
    v
Greedy span matching (sentence-level alignment)
    |
    v
Chunk building (word-limit aware) + score filtering
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
│   │   ├── filters.py           # Alignment validation filters
│   │   └── thresholds.py        # Data mode presets (parallel/comparable)
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
input_format: "document_pair"

# Input type: "paragraph" (default) or "sentence"
# Controls how aligned blocks are joined in the output: paragraphs are
# separated by double newlines, sentences are joined with spaces.
input_type: "paragraph"

# Data mode: "parallel" or "comparable"
# See "Data modes" section below for details.
data_mode: "parallel"
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

The pipeline expects input as subdirectories under the input path, each containing exactly two JSONL files (one per language):

```
data/raw/
└── 1399_49487/
    ├── en.jsonl
    └── fi.jsonl
```

Each JSONL file contains one line per paragraph (or per sentence if using `input_type: "sentence"`):

```json
{"doc_id": "anna_karenina.en", "para_id": 0, "lang": "en", "text": "Chapter 1"}
{"doc_id": "anna_karenina.en", "para_id": 1, "lang": "en", "text": "Happy families are all alike..."}
```

### Sentence-level input

When working with data that has been pre-segmented at sentence level, each JSONL entry is a single sentence. Set `input_type: "sentence"` in the config so that the pipeline joins aligned blocks with spaces rather than paragraph breaks, producing continuous text in the output.

## Data modes

The pipeline supports two data modes, controlled by `data_mode` in the config:

### `"parallel"` (default)

For known-parallel document pairs (e.g., published translations). Uses lenient filtering thresholds — paragraphs are still filtered at a low similarity threshold (0.30) to catch structural elements like footnotes, tables of contents, or image captions that may appear in one language but not the other.

### `"comparable"`

For document pairs that may share a topic but are not necessarily direct translations. Uses stricter filtering at every stage: paragraph-level similarity filtering (0.45), tighter sentence alignment windows, higher match thresholds, and chunk-level score filtering. This mode discards non-parallel portions rather than force-aligning them.

### Threshold overrides

Each mode sets sensible defaults for all filtering thresholds. Individual thresholds can be overridden in the config:

```yaml
data_mode: "comparable"
thresholds:
  para_sim_threshold: 0.50   # stricter paragraph filtering
  min_chunk_score: 0.70      # stricter chunk filtering
```

Available threshold keys:

| Key | Description |
|-----|-------------|
| `para_sim_threshold` | Minimum cosine similarity for paragraph groups |
| `sent_window` | Sentence alignment search window size |
| `sent_baseline_threshold` | Minimum similarity for baseline sentence matches |
| `sent_neighbour_margin` | Relaxation margin for neighbor expansion |
| `sent_fuzzy_threshold` | Threshold for 1-to-2 and 2-to-1 span matching |
| `sent_min_span_score` | Floor for greedy span selection |
| `min_chunk_score` | Minimum average score per output chunk |

## Output format

For each document pair, the pipeline produces:

- `{name}_chunks.jsonl` — Aligned parallel text chunks (the main output):

```json
{
  "paragraph_pairs": [[[0], [0]], [[1, 2], [1]]],
  "src_text": "Chapter 1\n\nHappy families are all alike...",
  "tgt_text": "Luku 1\n\nOnnelliset perheet ovat kaikki samanlaisia...",
  "src_words": 2450,
  "tgt_words": 2380,
  "avg_score": 0.8234
}
```

- `{name}_meta.jsonl` — Alignment metadata (DTW cost, DTW path, sentence alignments, input paths).

## Multiprocessing

The pipeline supports parallel processing via Python `multiprocessing` (spawn mode). By default it uses up to 8 workers, bounded by available CPUs. Each worker loads its own copy of the embedding model, so GPU memory is the main constraint. Sequential mode is available and recommended for single-GPU setups — set `use_multiprocessing=False` in the runner.

## Dependencies

- [sentence-transformers](https://www.sbert.net/) — Multilingual embedding models
- [fastdtw](https://github.com/slaypni/fastdtw) — Fast approximate DTW
- [dtaidistance](https://github.com/wannesm/dtaidistance) — Exact DTW
- [Spacy](https://spacy.io/) + `xx_ent_wiki_sm` — Multilingual sentence splitting
- [GlotLID](https://github.com/cisnlp/GlotLID) — Language identification model, licensed under Apache 2.0 with notices (see `THIRD_PARTY_LICENSES/GlotLID_LICENSE`)
- NumPy, PyYAML, tqdm
