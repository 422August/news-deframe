"""
Pydantic v2 schemas for news-deframe.

All public types are exported from this module.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SVORecord(BaseModel):
    """A single Subject-Verb-Object extraction from a sentence."""

    sentence: str = Field(..., description="Original sentence text")
    verb: str = Field(..., description="Head verb lemma or surface form")
    subjects: list[str] = Field(
        default_factory=list,
        description="Nominal/clausal subjects (nsubj, csubj deps)",
    )
    objects: list[str] = Field(
        default_factory=list,
        description="Direct/prepositional objects (dobj, pobj deps)",
    )
    is_passive: bool = Field(
        default=False,
        description="True when passive construction is detected",
    )
    voice_markers: list[str] = Field(
        default_factory=list,
        description="Passive markers found (e.g. 被, 遭, 受到, pass-tagged tokens)",
    )


class FramingDescriptor(BaseModel):
    """A framing target (named entity, action verb, or event noun) with evaluative modifiers.

    This is the canonical modifier record returned by the entities pipeline.

    ``target_type`` values
    ----------------------
    NER labels     ``PERSON``, ``ORG``, ``GPE``, ``NORP``, ``EVENT``, ``FAC``, ``PER``, ``LOC``
    Predicates     ``VERB_ACTION``  – an action verb carrying evaluative adverbs
    Event nouns    ``EVENT_NOUN``   – a key noun bearing adjectival / compound modifiers
    """

    target: str = Field(..., description="Surface form of the entity, verb, or event noun")
    target_type: str = Field(
        ...,
        description=(
            "spaCy NER label (PERSON/ORG/GPE/…) or synthetic tag "
            "(VERB_ACTION, EVENT_NOUN)"
        ),
    )
    modifiers: list[str] = Field(
        default_factory=list,
        description=(
            "Evaluative adjectives (amod) and adverbs (advmod) attached to the target"
        ),
    )


class EntityModifier(BaseModel):
    """A named entity paired with its descriptive modifiers.

    .. deprecated::
        Use :class:`FramingDescriptor` for new code.  ``EntityModifier`` is
        retained for backwards-compatibility with existing consumers and tests.
    """

    entity_name: str = Field(..., description="Surface form of the named entity")
    entity_type: str = Field(..., description="spaCy NER label (e.g. PERSON, ORG)")
    modifiers: list[str] = Field(
        default_factory=list,
        description="Adjectives (amod) and adverbs (advmod) associated with the entity",
    )

    def to_framing_descriptor(self) -> FramingDescriptor:
        """Convert to the canonical :class:`FramingDescriptor` representation."""
        return FramingDescriptor(
            target=self.entity_name,
            target_type=self.entity_type,
            modifiers=self.modifiers,
        )


class SentenceAlignment(BaseModel):
    """Cosine-similarity alignment between one sentence from article A and its best match in B."""

    sent_a: str = Field(..., description="Sentence from article A")
    sent_b: str | None = Field(
        default=None,
        description="Best-matching sentence from article B, or None if below threshold",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score [0, 1]",
    )

    @model_validator(mode="after")
    def check_none_has_low_score(self) -> "SentenceAlignment":
        if self.sent_b is None and self.similarity_score > 0.0:
            # Allow non-zero only when explicitly set (e.g. best available score kept for debug)
            pass
        return self


class ParsedArticle(BaseModel):
    """Full parse output for one article."""

    article_id: str = Field(..., description="Unique identifier (e.g. filename stem)")
    svo_records: list[SVORecord] = Field(default_factory=list)
    entity_modifiers: list[EntityModifier] = Field(
        default_factory=list,
        description=(
            "Framing descriptors for this article.  "
            "The field name is kept for backwards compatibility; entries may now "
            "include VERB_ACTION and EVENT_NOUN targets in addition to NER entities."
        ),
    )
    sentences: list[str] = Field(default_factory=list, description="Raw sentence strings")


class DiffReport(BaseModel):
    """Comparative framing analysis output for a pair of articles."""

    article_a_id: str
    article_b_id: str
    alignments: list[SentenceAlignment] = Field(default_factory=list)
    unshared_claims_a: list[str] = Field(
        default_factory=list,
        description="Sentences in A without a match in B above the threshold",
    )
    unshared_claims_b: list[str] = Field(
        default_factory=list,
        description="Sentences in B without a match in A above the threshold",
    )
    passive_ratio_a: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of SVO records in A that are passive",
    )
    passive_ratio_b: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of SVO records in B that are passive",
    )
