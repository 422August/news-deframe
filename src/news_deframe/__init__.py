"""news_deframe – public API surface."""
from news_deframe.schemas import (
    SVORecord,
    EntityModifier,
    FramingDescriptor,
    SentenceAlignment,
    ParsedArticle,
    DiffReport,
)

__all__ = [
    "SVORecord",
    "EntityModifier",
    "FramingDescriptor",
    "SentenceAlignment",
    "ParsedArticle",
    "DiffReport",
]
