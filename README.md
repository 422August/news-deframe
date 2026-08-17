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
| **Claim Coverage**         |    —   |     ✓     | Integrated table of shared/single-outlet claims and absent outlets          |
| **Actor Framing Blocks**   |    —   |     ✓     | Compares entity agency/patient patterns and associated actions by outlet    |
| **Framing Clusters**       |    —   |     ✓     | Groups articles with similar structural framing profiles                    |
| **Presentation Tiers**     |    —   |     ✓     | Concise default summary, `--details` for coding, `--verbose` for diagnostics|
| **Rich Terminal Reports**  |    ✓   |     ✓     | Terminal-width-aware, human-readable colour reports                         |
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

### Presentation Levels & CLI Options

`news-deframe analyze` provides three presentation tiers tailored for different analytical needs:

1. **Default View**: A concise, research-oriented summary designed for social-science and media studies.
2. **`--details`**: Detailed claim-level evidence, present/absent outlets, and full source sentences for human coding and manual inspection.
3. **`--verbose` (`-v`)**: Technical diagnostics including cluster centroid feature values, sentence similarity scores, exact ratio denominators, and all actors.
4. **`--format json`**: Full structured Pydantic v2 data model export for downstream statistical processing and archival.

#### Default Concise Analysis

```bash
news-deframe analyze articles/event_001/
```

Displays:
* Compact header with event ID and article/cluster counts.
* Integrated **Claim Coverage** table with aggregate counts (`Shared by all outlets`, `Shared by majority`, `Single-outlet claims`).
* **Actor Framing by Outlet** vertical blocks showing agent/patient grammatical ratios and associated actions for top actors.
* **Framing Clusters** listing structural cluster membership.
* **Research Interpretation Notes** clarifying scientific guardrails.

#### Human Coding / Manual Inspection (`--details`)

```bash
news-deframe analyze articles/event_001/ --details
```

Adds claim-level panels containing:
* Representative claim sentence
* Coverage ratio and percentage
* Present outlets and absent outlets
* Extracted source sentences grouped by article ID (without technical similarity noise)

#### Technical Diagnostics (`--verbose` or `-v`)

```bash
news-deframe analyze articles/event_001/ --verbose
```

Exposes engineering diagnostics:
* Framing cluster centroid feature values (`passive_ratio`, `mean_agent_ratio`, `mean_patient_ratio`, etc.)
* Exact denominator definitions (`role_occurrence_count = agent_count + patient_count`)
* Passive patient rate (`Passive Pt`) and evaluative modifiers
* Complete actor matrix beyond the default top 5

#### Combined Inspection & Diagnostics

```bash
news-deframe analyze articles/event_001/ --details --verbose
```

Shows full source sentences with similarity scores (`sim=0.76`) along with centroid feature values and complete diagnostics.

#### Additional Flags

Adjust semantic similarity threshold:

```bash
news-deframe analyze articles/event_001/ --threshold 0.55
```

Set the requested number of framing clusters:

```bash
news-deframe analyze articles/event_001/ --clusters 3
```

Export structured JSON:

```bash
news-deframe analyze articles/event_001/ --format json
```

Save JSON to file:

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
├── Integrated Claim Coverage
│   ├── summary counts (all / majority / single-outlet)
│   ├── descriptive coverage categories
│   ├── absent outlets
│   └── representative claims
│
├── Actor Framing by Outlet
│   ├── agent ratios
│   ├── patient ratios
│   ├── role observations (agent + patient occurrences)
│   └── associated actions / predicates
│
├── Framing Clusters
│   ├── structural profile membership
│   └── cluster centroids (in --verbose)
│
└── Research Interpretation Notes
```

## Integrated Claim Coverage

Sentences across all articles are embedded and verified for propositional equivalence. Related claims are grouped into clusters and classified by descriptive coverage frequency:

* **Widely shared** (e.g. 3/3 outlets)
* **Commonly reported** (e.g. 2/3 outlets)
* **Minority coverage** / **Single-outlet** (e.g. 1/3 outlets)

```text
Shared by all outlets:     3
Shared by majority:        1
Single-outlet claims:      20

Claim   Coverage   Category            Missing      Representative
C01     3/3        Widely shared       —            Police and organizers confirmed...
C02     3/3        Widely shared       —            According to police, demonstrators...
C04     2/3        Commonly reported   outlet_a     An officer sustained minor injuries...
C05     1/3        Single-outlet       b, c         A rally took place in downtown square...
```

Coverage measures **reporting frequency across the corpus**. It does not establish truth, and absence does not imply intentional omission.

## Actor Framing by Outlet

Rather than combining disparate metrics into cramped matrix cells, actor framing is presented in clean vertical blocks that separate **grammatical positioning** from **lexical actions**:

```text
Actor: Police

Outlet     Agent     Patient     Role observations
a          1.00       0.00              13
b          0.85       0.15              13
c          0.93       0.07              14

Associated actions:
  a: arrest, state, note, demand
  b: arrest, require, state, obstruct
  c: require, investigate, arrest
```

* **Agent / Patient ratios** describe how an actor is grammatically positioned when appearing in extracted clause roles (`agent / (agent + patient)`).
* **Associated actions** list the specific verbs tied to the actor across each outlet.
* Top actors are ranked deterministically by cross-outlet importance.

## Framing Clusters

Articles are grouped by unsupervised structural feature similarity (syntactic voice, agency/patient distributions, entity density, and claim participation).

* Clusters are assigned neutral labels (`Framing Cluster 1`, `Framing Cluster 2`).
* When each article forms a separate cluster in small corpora, a neutral notice is displayed.
* Numerical centroid vectors are accessible via `--verbose` or `--format json`.

## Research Interpretation Notes

Every report concludes with concise methodology reminders:
* Claim coverage reflects reporting frequency across outlets, not factual verification.
* Absence of a claim indicates a reporting difference, not intentional omission.
* Agent/patient ratios describe grammatical positioning in extracted clauses.
* Framing clusters group articles by structural feature similarity only.

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
│   ├── test_event_console_formatter.py
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
