from __future__ import annotations

import json
import re

from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout as LiteLLMTimeout,
)

from .classifier import (
    is_content_policy_violation,
    is_context_window_exceeded,
    looks_like_auth_error,
    looks_like_malformed_conversation_history_error,
)
from .types import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMContentPolicyViolationError,
    LLMContextWindowExceedError,
    LLMMalformedConversationHistoryError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)


def _format_openai_subscription_usage_limit(exception: Exception) -> str | None:
    """Return a concise user-facing message for ChatGPT plan exhaustion.

    OpenAI's Codex subscription endpoint reports plan exhaustion as a 429
    `usage_limit_reached` payload. LiteLLM wraps that payload in a
    RateLimitError string, which is otherwise noisy and easy to mistake for an
    API-key or OAuth failure.
    """

    raw = str(exception)
    if "usage_limit_reached" not in raw:
        return None

    match = re.search(r'\{\s*"error"\s*:', raw)
    if match is None:
        return None

    try:
        payload, _ = json.JSONDecoder().raw_decode(raw[match.start() :])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict) or error.get("type") != "usage_limit_reached":
        return None

    plan_type = error.get("plan_type")
    if isinstance(plan_type, str) and plan_type:
        heading = f"ChatGPT {plan_type.capitalize()} usage limit reached."
    else:
        heading = "ChatGPT usage limit reached."

    details: list[str] = [heading]

    resets_in = error.get("resets_in_seconds")
    if isinstance(resets_in, (int, float)) and resets_in >= 0:
        total_minutes = max(1, int((float(resets_in) + 59) // 60))
        hours, minutes = divmod(total_minutes, 60)
        if hours and minutes:
            duration = f"{hours}h {minutes}m"
        elif hours:
            duration = f"{hours}h"
        else:
            duration = f"{minutes}m"
        details.append(f"Included Work/Codex usage resets in about {duration}.")

    window_minutes = error.get("limit_window_minutes")
    if isinstance(window_minutes, (int, float)) and window_minutes > 0:
        details.append(f"Current usage window: {int(window_minutes)} minutes.")

    details.append(
        "This is a subscription usage limit, not an authentication failure; "
        "reauthorization is not required."
    )
    details.append(
        "Check ChatGPT Settings > Usage for the current reset time, credits, "
        "or reset options available to this account."
    )
    return " ".join(details)


def map_provider_exception(exception: Exception) -> Exception:
    """
    Map provider/LiteLLM exceptions to SDK-typed exceptions.

    Returns original exception if no mapping applies.
    """
    # Context window exceeded first (highest priority among normal retries)
    if is_context_window_exceeded(exception):
        return LLMContextWindowExceedError(str(exception))

    # Malformed prompt history is distinct from context-window exhaustion even
    # though the recovery path still uses condensation.
    if looks_like_malformed_conversation_history_error(exception):
        return LLMMalformedConversationHistoryError(str(exception))

    # Auth-like errors often appear as BadRequest/OpenAIError with specific text
    if looks_like_auth_error(exception):
        return LLMAuthenticationError(str(exception))

    if isinstance(exception, RateLimitError):
        friendly_usage_limit = _format_openai_subscription_usage_limit(exception)
        return LLMRateLimitError(friendly_usage_limit or str(exception))

    if isinstance(exception, LiteLLMTimeout):
        return LLMTimeoutError(str(exception))

    # Connectivity and service-side availability issues → service unavailable
    if isinstance(
        exception, (APIConnectionError, ServiceUnavailableError, InternalServerError)
    ):
        return LLMServiceUnavailableError(str(exception))

    # Content-policy blocks are deterministic 4xx; distinguish them from generic
    # bad requests so the agent can recover softly instead of hard-erroring.
    if is_content_policy_violation(exception):
        return LLMContentPolicyViolationError(str(exception))

    # Generic client-side 4xx errors
    if isinstance(exception, BadRequestError):
        return LLMBadRequestError(str(exception))

    # Unknown: let caller re-raise original
    return exception
