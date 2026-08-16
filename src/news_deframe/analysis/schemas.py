"""Event-level Pydantic v2 schemas for the multi-article analysis pipeline.

All types in this module are additive \u2014 the existing :class:`ParsedArticle`,
:class:`DiffReport`, and related schemas are unchanged.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from news_deframe.schemas import ParsedArticle


# \u2500\u2500 Claim Clustering \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


class SourceSentence(BaseModel):
    """A single sentence contributing to a claim cluster."""

    article_id: str = Field(..., description="Outlet / article identifier")
    text: str = Field(..., description="Raw sentence text")
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity to the cluster representative",
    )


class ClaimCluster(BaseModel):
    """A group of semantically similar sentences from different articles.

    Coverage is defined as the number of *distinct* articles that contributed
    at least one sentence to this cluster.  Duplicate paraphrases within a
    single article do not inflate the coverage count.
    """

    cluster_id: str = Field(..., description="Unique cluster identifier (e.g. 'C01')")
    representative: str = Field(
        ..., description="The sentence chosen as the cluster representative"
    )
    sources: list[SourceSentence] = Field(
        default_factory=list,
        description="All sentences contributing to this cluster",
    )
    article_ids: list[str] = Field(
        default_factory=list,
        description="Distinct article IDs that contain this claim",
    )
    coverage_count: int = Field(
        ..., ge=0, description="Number of distinct articles containing this claim"
    )
    total_articles: int = Field(
        ..., ge=1, description="Total articles in the event corpus"
    )
    coverage_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="coverage_count / total_articles"
    )


# \u2500\u2500 Entity \u00d7 Outlet Framing Matrix \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


class EntityOutletProfile(BaseModel):
    """Structural framing profile for one entity in one article/outlet.

    All ratio fields are in ``[0.0, 1.0]`` relative to *total_mentions*.
    They are 0.0 when *total_mentions* is 0.
    """

    entity_name: str
    article_id: str
    total_mentions: int = Field(default=0, ge=0)
    subject_count: int = Field(
        default=0, ge=0, description="Appearances as grammatical subject/agent"
    )
    object_count: int = Field(
        default=0, ge=0, description="Appearances as grammatical object/patient"
    )
    passive_count: int = Field(
        default=0, ge=0, description="Appearances in passive constructions"
    )
    agent_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="subject_count / total_mentions"
    )
    patient_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="object_count / total_mentions"
    )
    passive_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="passive_count / total_mentions"
    )
    modifiers: list[str] = Field(
        default_factory=list,
        description="Evaluative modifiers associated with this entity in this article",
    )
    associated_verbs: list[str] = Field(
        default_factory=list,
        description="Verbs for which this entity acts as subject or object",
    )


class EntityOutletMatrix(BaseModel):
    """Complete entity \u00d7 outlet framing matrix for an event."""

    entity_names: list[str] = Field(
        default_factory=list,
        description="Distinct entity names (sorted) found across the corpus",
    )
    article_ids: list[str] = Field(
        default_factory=list,
        description="Outlet / article IDs (sorted) included in the matrix",
    )
    profiles: list[EntityOutletProfile] = Field(
        default_factory=list,
        description="One profile per (entity, article) combination",
    )


# \u2500\u2500 Framing Clusters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


class ArticleFramingFeatures(BaseModel):
    """Numeric framing feature vector for one article.

    Used as input to the unsupervised clustering algorithm.
    """

    article_id: str
    passive_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_agent_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean agent ratio across all entities"
    )
    mean_patient_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean patient ratio across all entities"
    )
    entity_count: int = Field(
        default=0, ge=0, description="Number of distinct named entities"
    )
    sentence_count: int = Field(default=0, ge=0)
    claim_coverage_vector: list[float] = Field(
        default_factory=list,
        description=(
            "Per-claim coverage participation flag/weight \u2014 "
            "1.0 if article contains the claim, 0.0 otherwise"
        ),
    )


class FramingCluster(BaseModel):
    """A group of articles with structurally similar framing profiles.

    Labels are neutral (e.g. 'Framing Cluster 1') \u2014 no ideological inference
    is made from the structural similarity.
    """

    cluster_id: int = Field(..., description="Numeric cluster identifier (1-indexed)")
    label: str = Field(..., description="Human-readable neutral label")
    article_ids: list[str] = Field(
        default_factory=list, description="Articles belonging to this cluster"
    )
    centroid_description: dict[str, float] = Field(
        default_factory=dict,
        description="Mean feature values for this cluster (for interpretability)",
    )


# \u2500\u2500 Consensus / Outlier View \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


class ClaimConsensus(BaseModel):
    """Consensus metadata for a single claim cluster.

    Coverage categories describe *frequency only* and carry no truth or bias
    inference.  A rare claim is not automatically suspicious; a widely-shared
    claim is not automatically verified.
    """

    cluster_id: str
    representative: str
    coverage_count: int
    total_articles: int
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    coverage_category: str = Field(
        ...,
        description=(
            "Descriptive frequency category: "
            "'Widely shared', 'Commonly reported', 'Minority coverage', or 'Rare claim'"
        ),
    )
    outlets_present: list[str] = Field(
        default_factory=list, description="Article IDs that contain this claim"
    )
    outlets_absent: list[str] = Field(
        default_factory=list,
        description=(
            "Article IDs that do not contain this claim \u2014 "
            "reported as a coverage difference, not intentional omission"
        ),
    )


class ConsensusView(BaseModel):
    """Aggregated consensus / outlier analysis for all claim clusters."""

    total_articles: int
    claims: list[ClaimConsensus] = Field(default_factory=list)


# \u2500\u2500 Top-level Event Analysis \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


class EventAnalysis(BaseModel):
    """Complete event-level analysis result.

    This is the root schema returned by the ``analyze`` command.  It can be
    serialised to JSON via :meth:`model_dump_json` for downstream processing,
    visualisation, or archival.

    The *event_id* defaults to the folder name when loading from a directory::

        articles/event_001/ → event_id = 'event_001'
    """

    event_id: str = Field(..., description="Identifier for this event corpus")
    articles: list[ParsedArticle] = Field(
        default_factory=list, description="Parsed articles included in the analysis"
    )
    article_ids: list[str] = Field(
        default_factory=list, description="Ordered list of article IDs analysed"
    )
    claim_clusters: list[ClaimCluster] = Field(default_factory=list)
    entity_outlet_matrix: EntityOutletMatrix = Field(
        default_factory=EntityOutletMatrix
    )
    framing_clusters: list[FramingCluster] = Field(default_factory=list)
    consensus_view: ConsensusView = Field(
        default_factory=lambda: ConsensusView(total_articles=0)
    )


EventAnalysis.model_rebuild()
