# news-deframe

> **Objective structural analysis and comparative framing dissection for Chinese and English news articles.**

`news-deframe` performs structural framing analysis on news coverage. It supports two complementary workflows:

1. **Pairwise Diff (`diff`)**: Compare two articles covering the same event to surface sentence-level alignments, omissions, active/passive voice differences, and entity modifier contrasts.
2. **Event-Level Analysis (`analyze`)**: Analyse a whole folder or collection of articles covering an event to discover shared claim clusters, entity agency/patient matrices across outlets, unsupervised framing clusters, and consensus/outlier distributions.

Language is **detected automatically** from the article text — no `--lang` flag required. Both Traditional/Simplified Chinese and English are supported out of the box.

---

## Philosophy & Interpretation Boundary

> **news-deframe exposes framing evidence; humans interpret it.**

The software reports objective structural facts:
- *8 of 10 articles contain this claim.*
- *Outlet A places Entity X in agent position more frequently than Outlet B.*
- *Articles A, C, and F share similar structural framing profiles.*
- *This claim appears only in Outlet D (coverage difference).*

The software **never** makes automatic normative inferences such as:
- *Outlet D is biased.*
- *Outlet A is more objective.*
- *The rare claim is false or suspicious.*
- *Outlet C intentionally omitted the information.*
- *Framing Cluster 2 represents politically biased media.*

---

## Features

| Feature | Scope | Description |
|---|---|---|
| **Bilingual Support** | `diff` & `analyze` | Automatically detects Chinese (zh) or English (en) per article; routes to the correct spaCy pipeline |
| **SVO Extraction** | `diff` & `analyze` | Subject-Verb-Object triples with active/passive voice detection (`被`, `遭`, `受到`, `was … by`, `aux:pass`, …) |
| **Entity Framing Analysis** | `diff` & `analyze` | Named entities paired with evaluative adjectives/adverbs and structural roles (agent vs. patient) |
| **Sentence Semantic Alignment** | `diff` | Cosine similarity matrix to align sentences between two articles |
| **Multi-Document Claim Clustering** | `analyze` | Generalizes semantic alignment across N articles; deduplicates within-article repetitions |
| **Entity × Outlet Framing Matrix** | `analyze` | Cross-outlet agent/patient ratios, passive counts, associated verbs, and modifiers per entity |
| **Unsupervised Framing Clusters** | `analyze` | Groups articles by structural framing similarity with neutral labels (`Framing Cluster 1`, …) |
| **Consensus / Outlier View** | `analyze` | Frequency classification (`Widely shared`, `Commonly reported`, `Minority coverage`, `Rare claim`) with absent outlet lists |
| **Rich Terminal Output** | `diff` & `analyze` | Colour-coded reports rendered in the terminal via [Rich](https://github.com/Textualize/rich) |
| **JSON Export** | `diff` & `analyze` | Full structured Pydantic v2 data for downstream visualization, research, or pipelines |

---

## Requirements

- **Python** 3.10+
- **spaCy model (Chinese)**: `zh_core_web_md`
- **spaCy model (English)**: `en_core_web_md`
- **sentence-transformers**: `paraphrase-multilingual-MiniLM-L12-v2` — auto-downloaded on first run

---

## Installation

```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Download the required spaCy models
python -m spacy download zh_core_web_md   # Traditional / Simplified Chinese
python -m spacy download en_core_web_md   # English
```

> **Note**: Only the model(s) matching the language(s) you actually analyse need to be downloaded.
> The loader is lazy — neither model is imported at package load time.

---

## Usage

### 1. Multi-Article Event Analysis (`analyze`)

The `analyze` command operates on a directory containing news articles about the same event or multiple explicit files:

```bash
# Analyze an event directory (conventional workflow)
news-deframe analyze articles/event_001/

# Analyze multiple explicit files directly
news-deframe analyze outlet_a.txt outlet_b.txt outlet_c.txt

# External folders outside the repository are fully supported
news-deframe analyze "/path/to/my/event_articles/"

# Export event analysis to JSON
news-deframe analyze articles/event_001/ --format json

# Save JSON report to a file
news-deframe analyze articles/event_001/ --format json --output event_report.json

# Custom similarity threshold and number of framing clusters
news-deframe analyze articles/event_001/ --threshold 0.55 --clusters 3
```

#### Event Folder Convention

```text
news-deframe/
├── articles/
│   └── event_001/
│       ├── outlet_a.txt
│       ├── outlet_b.txt
│       ├── outlet_c.txt
│       └── outlet_d.txt
├── src/
├── tests/
├── README.md
└── pyproject.toml
```

- Each directory represents one **event corpus** (e.g. `event_001`).
- Files are loaded **non-recursively** by default.
- Supported file type: `.txt` (UTF-8 encoded).
- Article IDs are derived from filename stems (`outlet_a`, …).
- Discovery is sorted deterministically in lexicographical order.
- Hidden files (`.gitkeep`, `.DS_Store`) and non-`.txt` files are safely ignored.
- Minimum 2 usable articles required per analysis.

### 2. Pairwise Diff (`diff`)

The original pairwise command remains fully operational for granular 2-article comparisons:

```bash
# Rich terminal diff report (default) – language is auto-detected per file
news-deframe diff article_a.txt article_b.txt

# Works for English articles too
news-deframe diff article_en_a.txt article_en_b.txt

# Adjust the similarity threshold (default: 0.60)
news-deframe diff article_a.txt article_b.txt --threshold 0.5

# JSON output to stdout or file
news-deframe diff article_a.txt article_b.txt --format json -o diff_report.json
```

---

## Data Models

All data models are strongly typed with **Pydantic v2**.

### Event Analysis Schema (`EventAnalysis`)

```python
class EventAnalysis(BaseModel):
    event_id: str                          # e.g. "event_001"
    articles: list[ParsedArticle]          # full parsed article records
    article_ids: list[str]                 # ordered list of outlet IDs
    claim_clusters: list[ClaimCluster]     # multi-document claim clusters
    entity_outlet_matrix: EntityOutletMatrix # entity agency / patient profiles
    framing_clusters: list[FramingCluster] # unsupervised structural framing groups
    consensus_view: ConsensusView          # claim frequency & absent outlet view
```

### Pairwise Diff Schema (`DiffReport`)

```python
class DiffReport(BaseModel):
    article_a_id: str
    article_b_id: str
    alignments: list[SentenceAlignment]    # per-sentence best-match pairs
    unshared_claims_a: list[str]           # sentences unique to A
    unshared_claims_b: list[str]           # sentences unique to B
    passive_ratio_a: float                 # fraction of passive SVO records in A
    passive_ratio_b: float                 # fraction of passive SVO records in B
```

---

## Project Structure

```
news-deframe/
├── pyproject.toml
├── README.md
├── articles/
│   ├── .gitkeep
│   └── event_001/
├── tests/
│   ├── fixtures/
│   ├── test_svo.py               # SVO extraction + passive detection tests (zh + en)
│   ├── test_entities.py          # NER & modifier extraction tests
│   ├── test_diff.py              # Pairwise alignment + coverage diff tests
│   ├── test_article_loader.py    # Folder discovery, ordering, validation tests
│   ├── test_analysis.py          # Claim clustering, entity matrix, framing clusters, consensus tests
│   ├── test_cli_analyze.py       # CLI analyze command & diff backward compatibility tests
│   └── test_corpora.py           # Chinese, English, and mixed corpora tests
└── src/
    └── news_deframe/
        ├── __init__.py
        ├── cli.py                # Click CLI entry point (diff + analyze commands)
        ├── schemas.py            # Pydantic v2 schemas
        ├── parser/
        │   ├── spacy_loader.py   # Language detector + lazy per-lang model cache
        │   ├── svo.py            # Bilingual SVO extractor + passive detector
        │   ├── entities.py       # NER + modifier extractor
        │   └── article_loader.py # Folder discovery & multi-file loader
        ├── diff/
        │   ├── aligner.py        # Embedding + cosine similarity matrix
        │   └── coverage.py       # Unshared claim detection
        ├── analysis/
        │   ├── claims.py         # Multi-document claim clustering
        │   ├── entity_matrix.py  # Entity × outlet structural framing matrix
        │   ├── framing_clusters.py # Unsupervised framing clustering
        │   ├── consensus.py      # Consensus & outlier view
        │   ├── event.py          # Top-level event analysis orchestrator
        │   └── schemas.py        # Event-level Pydantic schemas
        └── formatters/
            ├── console.py        # Rich terminal diff renderer
            ├── event_console.py  # Rich terminal event analysis renderer
            └── json_export.py    # JSON serialiser for diff & event reports
```

---

## Running Tests

Tests run **offline without model downloads** using deterministic mocking:

```bash
# Run the full test suite
pytest

# With coverage report
pytest --cov=news_deframe --cov-report=term-missing
```

---

## License

[GPL-v3](LICENSE)
