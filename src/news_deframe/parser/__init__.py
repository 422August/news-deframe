"""Parser sub-package public API."""
from news_deframe.parser.svo import extract_svo, passive_ratio
from news_deframe.parser.entities import extract_entity_modifiers
from news_deframe.parser.spacy_loader import get_nlp, get_nlp_for_lang, get_nlp_model, detect_language
from news_deframe.parser.article_loader import (
    ArticleFile,
    ArticleLoadError,
    clean_article_text,
    discover_articles,
    load_article_files,
)

__all__ = [
    "extract_svo",
    "passive_ratio",
    "extract_entity_modifiers",
    "get_nlp",
    "get_nlp_for_lang",
    "get_nlp_model",
    "detect_language",
    "ArticleFile",
    "ArticleLoadError",
    "clean_article_text",
    "discover_articles",
    "load_article_files",
]
