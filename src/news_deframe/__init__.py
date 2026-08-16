"""news_deframe – public API surface."""
from news_deframe.schemas import (
    SVORecord,
    EntityModifier,
    SentenceAlignment,
    ParsedArticle,
    DiffReport,
)

__all__ = [
    "SVORecord",
    "EntityModifier",
    "SentenceAlignment",
    "ParsedArticle",
    "DiffReport",
]
