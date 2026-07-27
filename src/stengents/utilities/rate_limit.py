"""Provider-neutral bounded rate-limit execution for model invocations."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RateLimitPolicy:
    on_rate_limit: str = "wait"
    max_attempts: int = 1
    max_cumulative_wait_seconds: int = 30
    paid_fallback: bool = False

    def __post_init__(self) -> None:
        if self.on_rate_limit not in {"wait", "fail"}:
            raise ValueError("invalid_rate_limit_policy")
        if self.max_attempts < 0 or self.max_cumulative_wait_seconds < 0 or self.paid_fallback:
            raise ValueError("invalid_rate_limit_policy")

    def as_record(self) -> dict[str, object]:
        return asdict(self)


class RateLimitExhausted(RuntimeError):
    def __init__(self, report: dict[str, object]) -> None:
        super().__init__("rate_limit_exhausted")
        self.report = report


def _classification(error: Exception) -> str | None:
    if getattr(error, "status_code", None) != 429 and type(error).__name__ != "RateLimitError":
        return None
    text = repr(getattr(error, "body", error))
    if "PerDay" in text:
        return "per-day"
    if "PerMinute" in text:
        return "per-minute"
    return "unknown"


def _retry_delay(error: Exception) -> int:
    match = re.search(r"(?:retryDelay|retry_delay)[^0-9]*(\d+)", repr(getattr(error, "body", error)))
    return int(match.group(1)) if match else 1


def execute_with_rate_limit(
    action: Callable[[], T],
    *,
    policy: RateLimitPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, dict[str, object]]:
    """Execute once or retry bounded, observable transient rate limits."""
    attempts = 0
    retries = 0
    cumulative_wait = 0
    classification: str | None = None
    while True:
        attempts += 1
        try:
            result = action()
            return result, {
                "policy": policy.as_record(),
                "classification": classification,
                "attempts": attempts,
                "retries": retries,
                "cumulative_wait_seconds": cumulative_wait,
            }
        except Exception as error:
            classification = _classification(error)
            if classification is None:
                raise
            delay = _retry_delay(error)
            can_wait = (
                policy.on_rate_limit == "wait"
                and classification != "per-day"
                and attempts < policy.max_attempts
                and cumulative_wait + delay <= policy.max_cumulative_wait_seconds
            )
            if not can_wait:
                raise RateLimitExhausted(
                    {
                        "policy": policy.as_record(),
                        "classification": classification,
                        "attempts": attempts,
                        "retries": retries,
                        "cumulative_wait_seconds": cumulative_wait,
                    }
                ) from error
            sleep(delay)
            retries += 1
            cumulative_wait += delay
