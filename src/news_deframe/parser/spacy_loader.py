"""Lazy-loading helper for spaCy models with automatic language detection.

Importing this module does *not* load any model; call ``get_nlp(text)`` or
``get_nlp_for_lang(lang)`` explicitly.

Language Detection
------------------
``detect_language(text)`` uses a lightweight, dependency-free heuristic:
if the proportion of CJK Unified Ideographs in the text exceeds a threshold
the text is classified as ``'zh'`` (Chinese); otherwise ``'en'`` (English).

Model Routing
-------------
- Chinese (``zh``) → ``zh_core_web_md``
- English  (``en``) → ``en_core_web_md``

Both pipelines are cached independently in a thread-safe manner so each is
loaded at most once per process lifetime.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from spacy.language import Language

# ── Constants ─────────────────────────────────────────────────────────────────

ZH_MODEL = "zh_core_web_md"
EN_MODEL = "en_core_web_md"

#: Proportion of CJK characters above which text is classed as Chinese.
_CJK_THRESHOLD = 0.15

# ── Thread-safe per-language cache ────────────────────────────────────────────

_lock = threading.Lock()
_cache: dict[str, "Language"] = {}

Lang = Literal["zh", "en"]

# ── Public API ────────────────────────────────────────────────────────────────


def detect_language(text: str) -> Lang:
    """Return ``'zh'`` if *text* is predominantly Chinese, else ``'en'``.

    The detection is purely character-level — no external library required.
    CJK Unified Ideographs (U+4E00–U+9FFF) and common CJK extensions are
    counted; if they make up more than ``_CJK_THRESHOLD`` of all alphabetic
    characters the text is treated as Chinese.

    Parameters
    ----------
    text:
        Raw input string (may be any length; empty string returns ``'zh'``
        as a safe default because the original package only handled Chinese).

    Returns
    -------
    ``'zh'`` or ``'en'``
    """
    if not text:
        return "zh"

    cjk_count = 0
    alpha_count = 0

    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs (U+4E00–U+9FFF) + common extensions
        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
        ):
            cjk_count += 1
            alpha_count += 1
        elif ch.isalpha():
            alpha_count += 1

    if alpha_count == 0:
        return "zh"

    return "zh" if (cjk_count / alpha_count) >= _CJK_THRESHOLD else "en"


def get_nlp_for_lang(lang: Lang) -> "Language":
    """Return a cached spaCy pipeline for *lang* (``'zh'`` or ``'en'``).

    The pipeline is loaded at most once and then reused.  Thread-safe via
    double-checked locking.

    Raises
    ------
    RuntimeError
        When the required spaCy model has not been downloaded, with a clear
        instruction on how to fix it.
    """
    if lang in _cache:
        return _cache[lang]

    with _lock:
        if lang in _cache:
            return _cache[lang]

        model_name = ZH_MODEL if lang == "zh" else EN_MODEL
        try:
            import spacy  # lazy import – spacy not required at module load time

            _cache[lang] = spacy.load(model_name)
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is not installed. Install it with:\n\n"
                "    pip install spacy\n"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed.\n"
                "Install it with:\n\n"
                f"    python -m spacy download {model_name}\n"
            ) from exc

    return _cache[lang]


def get_nlp(text: str = "") -> "Language":
    """Return the correct spaCy pipeline for *text*, auto-detecting language.

    This is the primary entry point for callers that already have the raw
    text available.  Pass an empty string to get the Chinese pipeline (the
    original backwards-compatible default).

    Parameters
    ----------
    text:
        The article text used for language detection.  If omitted the
        Chinese pipeline is returned.

    Returns
    -------
    A cached ``spacy.Language`` instance.
    """
    lang = detect_language(text)
    return get_nlp_for_lang(lang)

