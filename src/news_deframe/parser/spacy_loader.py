"""Lazy-loading helper for the zh_core_web_md spaCy model.

Import this module does *not* load the model; call ``get_nlp()`` explicitly.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from spacy.language import Language

_lock = threading.Lock()
_nlp: Optional[Language] = None

MODEL_NAME = "zh_core_web_md"


def get_nlp() -> "Language":
    """Return a cached ``zh_core_web_md`` pipeline instance.

    Raises
    ------
    RuntimeError
        When the spaCy model is not installed.  Users should run::

            python -m spacy download zh_core_web_md
    """
    global _nlp
    if _nlp is not None:
        return _nlp

    with _lock:
        # Double-checked locking
        if _nlp is not None:
            return _nlp
        try:
            import spacy  # lazy import – spacy not required at module load time
            _nlp = spacy.load(MODEL_NAME)
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is not installed. Install it with:\n\n"
                "    pip install spacy\n"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{MODEL_NAME}' is not installed.\n"
                "Install it with:\n\n"
                f"    python -m spacy download {MODEL_NAME}\n"
            ) from exc
    return _nlp

