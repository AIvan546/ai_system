"""Трейсинг агентов (нод графа), LLM и тулов.

- LangSmith: задайте в окружении LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY,
  при необходимости LANGCHAIN_PROJECT (иначе подставится значение по умолчанию).
- Локальные логи: NUTRI_TRACE_LOG=1 или NUTRI_TRACE_LOG_LEVEL=INFO|DEBUG.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("nutri.trace")

F = TypeVar("F", bound=Callable[..., Any])

try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:  # pragma: no cover
    _langsmith_traceable = None


def init_tracing() -> None:
    """Инициализация: LangSmith из env, опционально — логирование в консоль."""
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes"):
        os.environ.setdefault("LANGCHAIN_PROJECT", "nutri-ai-meal-planner")
        logger.info(
            "LangSmith: LANGCHAIN_TRACING_V2 включён, проект=%s",
            os.environ.get("LANGCHAIN_PROJECT"),
        )

    log_level = os.environ.get("NUTRI_TRACE_LOG_LEVEL", "").upper()
    if log_level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    elif os.environ.get("NUTRI_TRACE_LOG", "").lower() in ("1", "true", "yes"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )


def _fallback_trace(name: str, kind: str) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            logger.info("%s.start %s", kind, name)
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                logger.info(
                    "%s.end %s duration_ms=%.1f",
                    kind,
                    name,
                    (time.perf_counter() - t0) * 1000,
                )
                return out
            except Exception:
                logger.exception("%s.error %s", kind, name)
                raise

        return wrapped  # type: ignore[return-value]

    return deco


def trace_agent(name: Optional[str] = None) -> Callable[[F], F]:
    """Нода графа / агент: в LangSmith — отдельный run, иначе — логи."""
    run_name = name

    def deco(fn: F) -> F:
        label = run_name or fn.__name__
        if _langsmith_traceable is not None:
            return _langsmith_traceable(name=f"agent:{label}")(fn)
        return _fallback_trace(label, "agent")(fn)

    return deco


def trace_llm(name: Optional[str] = None) -> Callable[[F], F]:
    """Вызов LLM (дополнительно к трейсу провайдера, если LangSmith включён)."""
    run_name = name

    def deco(fn: F) -> F:
        label = run_name or fn.__name__
        if _langsmith_traceable is not None:
            return _langsmith_traceable(name=f"llm:{label}")(fn)
        return _fallback_trace(label, "llm")(fn)

    return deco


def trace_tool(name: Optional[str] = None) -> Callable[[F], F]:
    """Чистая Python-функция тула: в LangSmith — как дочерний run."""
    run_name = name

    def deco(fn: F) -> F:
        label = run_name or fn.__name__
        if _langsmith_traceable is not None:
            return _langsmith_traceable(name=f"tool:{label}")(fn)
        return _fallback_trace(label, "tool")(fn)

    return deco
