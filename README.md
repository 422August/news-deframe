# news-deframe

> **Structural comparison of how news outlets frame the same event.**

`news-deframe` is a bilingual NLP tool for examining how news coverage differs across outlets. Rather than assigning bias scores or deciding which source is more objective, it surfaces **observable structural differences**: who is positioned as an agent or patient, which claims are shared or absent, how entities are described, and which articles exhibit similar framing patterns.

It supports two complementary workflows:

* **`diff`** — detailed pairwise comparison between two articles.
* **`analyze`** — event-level analysis across multiple articles.

Traditional Chinese, Simplified Chinese, and English are supported. Language is detected automatically from article text; no `--lang` option is required.

---

## Philosophy

> **news-deframe exposes framing evidence; humans interpret it.**

Framing is contextual. A structural difference alone does not establish bias, intent, accuracy, or truthfulness.

`news-deframe` therefore reports observations such as:

* *8 of 10 articles contain this claim.*
* *Outlet A places Entity X in agent position more frequently than Outlet B.*
* *Articles A, C, and F have similar structural framing profiles.*
* *This claim appears only in Outlet D.*

It deliberately does **not** automatically conclude that:

* an outlet is biased or objective;
* a rare claim is false or suspicious;
* a widely reported claim is necessarily true;
* an absent claim was intentionally omitted;
* a framing cluster represents a political or ideological position.

The output is evidence for human interpretation, not a verdict.

---

## Features

| Feature                    | `diff` | `analyze` | Description                                                                 |
| -------------------------- | :----: | :-------: | --------------------------------------------------------------------------- |
| **Bilingual NLP**          |    ✓   |     ✓     | Automatic Chinese/English detection with language-specific spaCy pipelines  |
| **SVO Extraction**         |    ✓   |     ✓     | Extracts subject–verb–object relationships and active/passive constructions |
| **Entity Framing**         |    ✓   |     ✓     | Examines named entities, modifiers, and structural roles                    |
| **Semantic Alignment**     |    ✓   |     —     | Aligns semantically similar sentences between two articles                  |
| **Unshared Claims**        |    ✓   |     —     | Surfaces claims found in only one side of a pairwise comparison             |
| **Claim Clustering**       |    —   |     ✓     | Groups semantically related claims across multiple articles                 |
| **Entity × Outlet Matrix** |    —   |     ✓     | Compares entity agency/patient patterns across outlets                      |
| **Framing Clusters**       |    —   |     ✓     | Groups articles with similar structural framing profiles                    |
| **Consensus / Outliers**   |    —   |     ✓     | Shows claim coverage and identifies outlets where claims are absent         |
| **Rich Terminal Reports**  |    ✓   |     ✓     | Human-readable colour terminal output                                       |
| **JSON Export**            |    ✓   |     ✓     | Structured Pydantic v2 output for research and downstream processing        |

---

## Requirements

* Python 3.10+
* `zh_core_web_md` for Chinese
* `en_core_web_md` for English
* `paraphrase-multilingual-MiniLM-L12-v2` for multilingual semantic embeddings

The sentence-transformer model is downloaded automatically on first use.

spaCy models are loaded lazily, so only the model required for the language being analysed needs to be installed.

---

## Installation

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

Install the spaCy model(s) you need:

```bash
# Chinese
python -m spacy download zh_core_web_md

# English
python -m spacy download en_core_web_md
```

---

# Usage

## Event-Level Analysis

Use `analyze` when comparing several articles covering the same event.

The recommended workflow is to place the articles for one event in a directory:

```text
articles/
└── event_001/
    ├── outlet_a.txt
    ├── outlet_b.txt
    ├── outlet_c.txt
    └── outlet_d.txt
```

Then run:

```bash
news-deframe analyze articles/event_001/
```

The directory name becomes the default event ID, while each filename stem becomes its article/outlet ID.

For example:

```text
articles/election_debate/
├── reuters.txt
├── outlet_a.txt
└── outlet_b.txt
```

produces the event ID `election_debate` and article IDs `reuters`, `outlet_a`, and `outlet_b`.

Directories outside the repository work as well:

```bash
news-deframe analyze "/path/to/event_articles/"
```

Multiple files may also be supplied explicitly:

```bash
news-deframe analyze a.txt b.txt c.txt
```

### Folder Loading Rules

Event directories are loaded non-recursively.

* `.txt` files are read as UTF-8.
* Files are processed in deterministic lexicographical order.
* Hidden and unsupported files are ignored.
* At least two usable articles are required.
* Article IDs are derived from filename stems.

One directory should contain articles covering the **same event**. Mixing unrelated events will reduce the usefulness of claim clustering and framing comparisons.

### Options

Adjust the semantic similarity threshold:

```bash
news-deframe analyze articles/event_001/ --threshold 0.55
```

Set the requested number of framing clusters:

```bash
news-deframe analyze articles/event_001/ --clusters 3
```

Export JSON:

```bash
news-deframe analyze articles/event_001/ --format json
```

Save the report:

```bash
news-deframe analyze articles/event_001/ \
    --format json \
    --output event_report.json
```

---

## Pairwise Diff

Use `diff` for a focused comparison between exactly two articles:

```bash
news-deframe diff article_a.txt article_b.txt
```

The pairwise report includes sentence alignment, unshared claims, passive-voice statistics, and entity framing information.

Adjust the semantic matching threshold:

```bash
news-deframe diff article_a.txt article_b.txt --threshold 0.5
```

Export the result as JSON:

```bash
news-deframe diff article_a.txt article_b.txt --format json
```

Or write it directly to a file:

```bash
news-deframe diff article_a.txt article_b.txt \
    --format json \
    --output diff_report.json
```

The original `diff` workflow remains independent from event-level analysis. `analyze` extends the project to multi-document corpora rather than replacing pairwise comparison.

---

# How Event Analysis Works

Given a collection of articles describing the same event, `news-deframe analyze` builds an event-level representation:

```text
Event
│
├── Articles
│
├── Claim Clusters
│   ├── shared claims
│   ├── minority-coverage claims
│   └── article coverage
│
├── Entities
│   ├── agent roles
│   ├── patient roles
│   ├── passive constructions
│   ├── associated verbs
│   └── modifiers
│
├── Entity × Outlet Framing Matrix
│
├── Framing Clusters
│
└── Consensus / Outlier View
```

## Claim Clustering

Sentences across all articles are embedded and compared semantically. Related claims are grouped into clusters representing information shared across the event corpus.

A cluster might look conceptually like:

```text
Claim C01
Representative:
"Police arrested three protesters."

Present in:
- outlet_a
- outlet_b
- outlet_d

Coverage: 3 / 5
```

Multiple similar sentences from the same article do not increase article-level coverage.

This allows `news-deframe` to move beyond pairwise omissions and examine how widely each claim appears across an entire corpus.

## Consensus / Outlier View

Claim clusters are classified by their coverage across the analysed articles.

For example:

```text
Widely shared        9/10
Commonly reported    7/10
Minority coverage    3/10
Rare claim           1/10
```

The report also identifies outlets where a claim is absent.

These categories measure **coverage frequency only**.

`1/10` means that one analysed article contains the claim. It does not mean the claim is false.

Likewise, `10/10` means every analysed article contains the claim. It does not independently verify that claim.

## Entity × Outlet Framing Matrix

The entity matrix compares how important entities are grammatically positioned across outlets.

It can examine signals such as:

* total mentions;
* appearances as subject or agent;
* appearances as object or patient;
* passive constructions;
* associated actions and verbs;
* associated modifiers;
* normalized agent and patient ratios.

Conceptually:

```text
                    Outlet A    Outlet B    Outlet C

Police
  Agent ratio          0.72        0.31        0.55
  Patient ratio        0.08        0.29        0.12

Protesters
  Agent ratio          0.61        0.24        0.40
  Patient ratio        0.19        0.67        0.38
```

These measurements describe grammatical positioning. They do not by themselves establish an outlet's attitude toward an entity.

## Framing Clusters

Articles can be grouped according to similarities in their structural framing profiles.

Signals may include:

* claim coverage;
* entity agency patterns;
* entity patient patterns;
* passive voice usage;
* entity distributions;
* sentence-level structural features.

Clusters receive neutral identifiers:

```text
Framing Cluster 1
Framing Cluster 2
Framing Cluster 3
```

No political or ideological meaning is automatically assigned to a cluster.

---

# Data Models

All public analysis structures use **Pydantic v2** models.

## Event Analysis

```python
class EventAnalysis(BaseModel):
    event_id: str
    articles: list[ParsedArticle]
    article_ids: list[str]
    claim_clusters: list[ClaimCluster]
    entity_outlet_matrix: EntityOutletMatrix
    framing_clusters: list[FramingCluster]
    consensus_view: ConsensusView
```

## Pairwise Diff

```python
class DiffReport(BaseModel):
    article_a_id: str
    article_b_id: str
    alignments: list[SentenceAlignment]
    unshared_claims_a: list[str]
    unshared_claims_b: list[str]
    passive_ratio_a: float
    passive_ratio_b: float
```

JSON export exposes these structures for visualization, statistical analysis, archival use, or integration with other tools.

---

# Project Structure

```text
news-deframe/
├── articles/
│   └── .gitkeep
├── tests/
│   ├── fixtures/
│   ├── test_svo.py
│   ├── test_entities.py
│   ├── test_diff.py
│   ├── test_article_loader.py
│   ├── test_analysis.py
│   ├── test_cli_analyze.py
│   └── test_corpora.py
├── src/
│   └── news_deframe/
│       ├── __init__.py
│       ├── cli.py
│       ├── schemas.py
│       ├── parser/
│       │   ├── spacy_loader.py
│       │   ├── svo.py
│       │   ├── entities.py
│       │   └── article_loader.py
│       ├── diff/
│       │   ├── aligner.py
│       │   └── coverage.py
│       ├── analysis/
│       │   ├── claims.py
│       │   ├── entity_matrix.py
│       │   ├── framing_clusters.py
│       │   ├── consensus.py
│       │   ├── event.py
│       │   └── schemas.py
│       └── formatters/
│           ├── console.py
│           ├── event_console.py
│           └── json_export.py
├── pyproject.toml
└── README.md
```

The `articles/` directory is a conventional local workspace. Article corpora may be stored anywhere on the filesystem and passed to `news-deframe analyze`.

---

# Python API

The parsing and analysis layers are modular and can be used independently of the CLI.

Pairwise parsing continues to use the existing parser:

```python
from news_deframe.parser.spacy_loader import get_nlp
from news_deframe.parser.svo import extract_svo
from news_deframe.parser.entities import extract_entity_modifiers
from news_deframe.diff.coverage import compute_coverage
from news_deframe.schemas import ParsedArticle


def parse(text: str, article_id: str) -> ParsedArticle:
    nlp = get_nlp(text)
    doc = nlp(text)

    return ParsedArticle(
        article_id=article_id,
        sentences=[
            sentence.text.strip()
            for sentence in doc.sents
            if sentence.text.strip()
        ],
        svo_records=extract_svo(doc),
        entity_modifiers=extract_entity_modifiers(doc),
    )
```

Event-level components are separated into dedicated analysis modules so that claim clustering, entity matrices, consensus analysis, and framing clustering can evolve independently.

---

# Testing

The test suite is designed to run without downloading NLP or embedding models. External model calls are replaced with deterministic test doubles where appropriate.

Run all tests:

```bash
pytest
```

Generate a coverage report:

```bash
pytest --cov=news_deframe --cov-report=term-missing
```

Tests cover both the original pairwise workflow and event-level functionality, including folder loading, multi-document claim clustering, entity matrices, consensus analysis, framing clustering, Chinese/English corpora, JSON serialization, and backward compatibility.

---

# Design Principles

**Human interpretation first.** Structural measurements are evidence, not judgments about bias, intent, or truth.

**Additive workflows.** `analyze` extends `news-deframe`; it does not replace the focused `diff` workflow.

**No import side effects.** NLP and embedding models are loaded lazily rather than during package import.

**Automatic language detection.** Chinese and English articles are routed to the appropriate NLP pipeline without a language flag.

**Modular analysis.** Parsing, pairwise diffing, event analysis, and presentation remain separate concerns.

**Strict schemas.** Public data structures use Pydantic v2 models.

**Testability.** NLP and embedding layers can be mocked so the test suite remains fast and offline-safe.

**Machine-readable results.** Console output is designed for humans, while JSON output preserves structured evidence for downstream analysis.

---

# Limitations

`news-deframe` performs computational linguistic analysis and should not be treated as an automated fact-checker or political-bias classifier.

Semantic similarity does not guarantee that two sentences make exactly the same claim. Entity recognition and grammatical-role extraction can also produce errors, particularly with ambiguous language or complex sentence structures.

Results should therefore be treated as **analytical signals to inspect**, with the original articles remaining the authoritative context for interpretation.

---

## License

[GPL-v3](LICENSE)
