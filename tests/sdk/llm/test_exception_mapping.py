import httpx
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMContentPolicyViolationError,
    LLMMalformedConversationHistoryError,
    LLMRateLimitError,
    map_provider_exception,
)


MODEL = "test-model"
PROVIDER = "test-provider"


def test_map_auth_error_from_bad_request():
    e = BadRequestError("Invalid API key provided", MODEL, PROVIDER)
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMAuthenticationError)


def test_map_auth_error_from_openai_error():
    # OpenAIError has odd behavior; create a BadRequestError that wraps an
    # auth-like message instead, as providers commonly route auth issues
    # through BadRequestError in LiteLLM
    e = BadRequestError("status 401 Unauthorized: missing API key", MODEL, PROVIDER)
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMAuthenticationError)


def test_map_typed_authentication_error_without_pattern_match():
    # Typed 401 from litellm whose message text doesn't contain any of the
    # auth heuristic patterns — should still map via the isinstance check.
    e = AuthenticationError("Bearer token expired", PROVIDER, MODEL)
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMAuthenticationError)


def test_map_typed_permission_denied_error():
    response = httpx.Response(
        status_code=403,
        request=httpx.Request("POST", "https://example.test"),
    )
    e = PermissionDeniedError("Region not allowed", PROVIDER, MODEL, response)
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMAuthenticationError)


def test_map_malformed_tool_history_bad_request():
    e = BadRequestError(
        (
            'AnthropicException - {"type":"error","error":{"type":'
            '"invalid_request_error","message":"messages.134: `tool_use` '
            "ids were found without `tool_result` blocks immediately after: "
            "toolu_01Aye4s5HrR2uXwXFYgtQi4H. Each `tool_use` block must have "
            'a corresponding `tool_result` block in the next message."}}'
        ),
        MODEL,
        PROVIDER,
    )
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMMalformedConversationHistoryError)


def test_map_openai_tool_argument_parse_internal_server_error():
    e = InternalServerError(
        (
            "OpenAIException - Failed to parse tool call arguments as JSON: "
            "invalid string: missing closing quote"
        ),
        PROVIDER,
        MODEL,
    )
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMMalformedConversationHistoryError)


def test_map_typed_content_policy_violation():
    e = ContentPolicyViolationError(
        "Output blocked by content filtering policy", MODEL, PROVIDER
    )
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMContentPolicyViolationError)
    # Subtype of LLMBadRequestError keeps back-compat for existing handlers.
    assert isinstance(mapped, LLMBadRequestError)


def test_map_content_policy_violation_text_fallback():
    # Proxied/aliased providers may flatten the typed class to a plain 400.
    e = BadRequestError(
        "AnthropicException - Output blocked by content filtering policy",
        MODEL,
        PROVIDER,
    )
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMContentPolicyViolationError)


def test_map_generic_bad_request():
    e = BadRequestError("Some client-side error not related to auth", MODEL, PROVIDER)
    mapped = map_provider_exception(e)
    assert isinstance(mapped, LLMBadRequestError)
    # A generic 400 must NOT be misclassified as content-policy.
    assert not isinstance(mapped, LLMContentPolicyViolationError)


def test_map_openai_subscription_usage_limit_to_friendly_message():
    e = RateLimitError(
        message=(
            'OpenAIException - {"error":{"type":"usage_limit_reached",'
            '"message":"The usage limit has been reached","plan_type":"plus",'
            '"resets_at":1794498018,"eligible_promo":null,'
            '"limit_window_minutes":300,"resets_in_seconds":10446}}'
        ),
        llm_provider="openai",
        model="gpt-6.1-sol",
    )

    mapped = map_provider_exception(e)

    assert isinstance(mapped, LLMRateLimitError)
    message = str(mapped)
    assert "ChatGPT Plus usage limit reached." in message
    assert "resets in about 2h 55m" in message
    assert "Current usage window: 300 minutes." in message
    assert "not an authentication failure" in message
    assert "reauthorization is not required" in message
    assert "Settings > Usage" in message
    assert "OpenAIException" not in message


def test_map_generic_rate_limit_keeps_provider_message():
    e = RateLimitError(
        message="ordinary provider rate limit",
        llm_provider="openai",
        model="gpt-4o",
    )

    mapped = map_provider_exception(e)

    assert isinstance(mapped, LLMRateLimitError)
    assert "ordinary provider rate limit" in str(mapped)


def test_passthrough_unknown_exception():
    class MyCustom(Exception):
        pass

    e = MyCustom("random")
    mapped = map_provider_exception(e)
    assert mapped is e
