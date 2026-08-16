# news-deframe

> **Objective structural analysis and comparative framing dissection for Chinese news articles.**

`news-deframe` takes two articles covering the same event and surfaces exactly how they differ — not just *what* they say, but *how* they frame it: who is positioned as agent or victim, what entities are foregrounded or buried, and which claims appear in one outlet but not the other.

---

## Features

| Feature | Description |
|---|---|
| **SVO Extraction** | Subject-Verb-Object triples with active/passive voice detection (`被`, `遭`, `受到`, …) |
| **Entity Modifier Analysis** | Named entities paired with their associated adjective/adverb descriptors |
| **Semantic Alignment** | Sentence-level cosine similarity matrix to find shared and divergent claims |
| **Unshared Claim Detection** | Sentences unique to one article, exposing omissions and framing gaps |
| **Rich Terminal Output** | Colour-coded diff report rendered in the terminal via [Rich](https://github.com/Textualize/rich) |
| **JSON Export** | Machine-readable output for downstream pipelines or archiving |

---

## Requirements

- **Python** 3.10+
- **spaCy model**: `zh_core_web_md` (Chinese medium pipeline)
- **sentence-transformers**: `paraphrase-multilingual-MiniLM-L12-v2` — auto-downloaded on first run

---

## Installation

```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Download the required spaCy Chinese model
python -m spacy download zh_core_web_md
```

---

## Usage

### CLI

```bash
# Rich terminal diff report (default)
news-deframe diff article_a.txt article_b.txt

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

nlp = get_nlp()

def parse(text: str, article_id: str) -> ParsedArticle:
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
│   │   ├── incident_01_a.txt     # Chinese news sample (police framing)
│   │   └── incident_01_b.txt     # Chinese news sample (community framing)
│   ├── test_svo.py               # SVO extraction + passive detection tests
│   └── test_diff.py              # Alignment + coverage diff tests
└── src/
    └── news_deframe/
        ├── __init__.py
        ├── cli.py                # Click CLI entry point
        ├── schemas.py            # Pydantic v2 models
        ├── parser/
        │   ├── spacy_loader.py   # Lazy, thread-safe model loader
        │   ├── svo.py            # SVO extractor + passive detector
        │   └── entities.py       # NER + modifier extractor
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
- **Strict type safety** — all public interfaces are typed with Pydantic v2 models.
- **Modular** — parsing, diffing, and formatting are fully isolated concerns; swap any layer independently.
- **Testable** — NLP and embedding layers are designed to be easily mocked, keeping the test suite fast and offline-safe.

---

## License

[GPL-v3](LICENSE)
