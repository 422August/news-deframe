# news-deframe

> **Objective structural analysis and comparative framing dissection for Chinese and English news articles.**

`news-deframe` takes two articles covering the same event and surfaces exactly how they differ — not just *what* they say, but *how* they frame it: who is positioned as agent or victim, what entities are foregrounded or buried, and which claims appear in one outlet but not the other.

Language is **detected automatically** from the article text — no `--lang` flag required. Both Traditional/Simplified Chinese and English are supported out of the box.

---

## Features

| Feature | Description |
|---|---|
| **Bilingual Support** | Automatically detects Chinese (zh) or English (en) per article; routes to the correct spaCy pipeline |
| **SVO Extraction** | Subject-Verb-Object triples with active/passive voice detection (`被`, `遭`, `受到`, `was … by`, `aux:pass`, …) |
| **Entity Modifier Analysis** | Named entities paired with their associated adjective/adverb descriptors; noise types (`CARDINAL`, `DATE`, etc.) filtered out |
| **Semantic Alignment** | Sentence-level cosine similarity matrix to find shared and divergent claims |
| **Unshared Claim Detection** | Sentences unique to one article, exposing omissions and framing gaps |
| **Rich Terminal Output** | Colour-coded diff report rendered in the terminal via [Rich](https://github.com/Textualize/rich) |
| **JSON Export** | Machine-readable output for downstream pipelines or archiving |

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

### CLI

```bash
# Rich terminal diff report (default) – language is auto-detected per file
news-deframe diff article_a.txt article_b.txt

# Works for English articles too
news-deframe diff article_en_a.txt article_en_b.txt

# Adjust the similarity threshold (default: 0.60)
news-deframe diff article_a.txt article_b.txt --threshold 0.5

# JSON output to stdout
news-deframe diff article_a.txt article_b.txt --format json

# Save JSON report to a file
news-deframe diff article_a.txt article_b.txt --format json --output report.json
```

**Arguments**

| Argument / Option | Default | Description |
|---|---|---|
| `FILE_A` | — | Path to the first article (UTF-8 plain text) |
| `FILE_B` | — | Path to the second article (UTF-8 plain text) |
| `--threshold FLOAT` | `0.60` | Cosine similarity cutoff for sentence matching |
| `--format console\|json` | `console` | Output format |
| `--output / -o PATH` | — | Write JSON output to file (only with `--format json`) |

### Python API

```python
from news_deframe.parser.spacy_loader import get_nlp
from news_deframe.parser.svo import extract_svo
from news_deframe.parser.entities import extract_entity_modifiers
from news_deframe.diff.coverage import compute_coverage
from news_deframe.schemas import ParsedArticle

def parse(text: str, article_id: str) -> ParsedArticle:
    # Language is auto-detected from text (CJK proportion heuristic)
    nlp = get_nlp(text)
    doc = nlp(text)
    return ParsedArticle(
        article_id=article_id,
        sentences=[s.text.strip() for s in doc.sents if s.text.strip()],
        svo_records=extract_svo(doc),
        entity_modifiers=extract_entity_modifiers(doc),
    )

article_a = parse(open("article_a.txt").read(), "article_a")
article_b = parse(open("article_b.txt").read(), "article_b")

report = compute_coverage(article_a, article_b, threshold=0.60)

print(f"Passive ratio A: {report.passive_ratio_a:.1%}")
print(f"Passive ratio B: {report.passive_ratio_b:.1%}")
print(f"Unshared claims in A: {len(report.unshared_claims_a)}")
print(f"Unshared claims in B: {len(report.unshared_claims_b)}")
```

---

## Output Schema

The core output is a `DiffReport` Pydantic model:

```python
class DiffReport(BaseModel):
    article_a_id: str
    article_b_id: str
    alignments: list[SentenceAlignment]   # per-sentence best-match pairs
    unshared_claims_a: list[str]          # sentences only in A
    unshared_claims_b: list[str]          # sentences only in B
    passive_ratio_a: float                # fraction of passive SVO records in A
    passive_ratio_b: float                # fraction of passive SVO records in B
```

Each `SentenceAlignment` record contains:

| Field | Type | Description |
|---|---|---|
| `sent_a` | `str` | Sentence from article A |
| `sent_b` | `str \| None` | Best-matching sentence from B, or `None` if below threshold |
| `similarity_score` | `float [0, 1]` | Cosine similarity between the two sentences |

---

## Project Structure

```
news-deframe/
├── pyproject.toml
├── README.md
├── tests/
│   ├── fixtures/
│   │   ├── incident_01_a.txt        # Chinese news sample (police framing)
│   │   ├── incident_01_b.txt        # Chinese news sample (community framing)
│   │   ├── incident_en_01_a.txt     # English news sample (active/official framing)
│   │   └── incident_en_01_b.txt     # English news sample (passive/community framing)
│   ├── test_svo.py                  # SVO extraction + passive detection tests (zh + en)
│   └── test_diff.py                 # Alignment + coverage diff tests
└── src/
    └── news_deframe/
        ├── __init__.py
        ├── cli.py                # Click CLI entry point (auto language detection)
        ├── schemas.py            # Pydantic v2 models
        ├── parser/
        │   ├── spacy_loader.py   # Language detector + lazy per-lang model cache
        │   ├── svo.py            # Bilingual SVO extractor + passive detector
        │   └── entities.py       # NER + modifier extractor (with entity type filter)
        ├── diff/
        │   ├── aligner.py        # Embedding + cosine similarity matrix
        │   └── coverage.py       # Unshared claim detection
        └── formatters/
            ├── console.py        # Rich terminal renderer
            └── json_export.py    # JSON serialiser
```

---

## Running Tests

Tests run **without** downloading any models — all NLP and embedding calls are mocked with deterministic stubs.

```bash
# Run the full test suite
pytest

# With coverage report
pytest --cov=news_deframe --cov-report=term-missing
```

---

## Design Principles

- **No side effects on import** — model loading is lazy and thread-safe; importing any module never triggers a download.
- **Automatic language detection** — `detect_language(text)` uses a dependency-free CJK proportion heuristic; no `--lang` flag needed anywhere.
- **Strict type safety** — all public interfaces are typed with Pydantic v2 models.
- **Modular** — parsing, diffing, and formatting are fully isolated concerns; swap any layer independently.
- **Testable** — NLP and embedding layers are designed to be easily mocked, keeping the test suite fast and offline-safe.

---

## License

[GPL-v3](LICENSE)
