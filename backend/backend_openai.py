"""Backend for OpenAI Responses API
(compatible with OpenAI official API, partial LiteLLM support, Azure, and OpenAI-compatible endpoints).

Key design decisions:
- Uses modern Responses API (`client.responses.create`)
- Uses `response_format` with JSON Schema for structured output
- Falls back to tool calling if response_format is rejected
- Falls back to plain text + JSON extraction as last resort
- Azure-compatible: automatically detects and drops unsupported params
- API key rotation via round-robin pool for load balancing
"""

import json
import logging
import os
import threading
import itertools
import time
import re
from typing import Any

import openai
from funcy import notnone, select_values

from .backend_utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)

from leaf.utils.config import Config

logger = logging.getLogger("leaf")

# ---------------------------------------------------------
# Concurrency control & API key rotation pool
# ---------------------------------------------------------

MAX_CONCURRENT_REQUESTS = 5
_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

_client_iterator = None
_client_pool_lock = threading.Lock()

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

AZURE_UNSUPPORTED_PARAMS = {
    "tool_choice",
}

# ---------------------------------------------------------
# Azure helpers
# ---------------------------------------------------------


def _is_azure_model(model: str | None, base_url: str | None = None) -> bool:
    """Detect if request targets Azure."""
    if base_url and "azure" in base_url.lower():
        return True

    api_type = os.getenv("OPENAI_API_TYPE", "").lower()
    if api_type == "azure":
        return True

    if model and model.startswith("azure/"):
        return True

    return False


def _sanitize_kwargs_for_azure(
    kwargs: dict,
    model: str | None = None,
) -> dict:
    """Remove params unsupported by Azure."""
    sanitized = dict(kwargs)

    base_url = os.getenv("OPENAI_BASE_URL", "")

    if _is_azure_model(model, base_url):
        for param in AZURE_UNSUPPORTED_PARAMS:
            if param in sanitized:
                logger.warning(
                    f"Dropping unsupported param '{param}' for Azure."
                )
                del sanitized[param]

    return sanitized


def _should_strip_tool_params(err_msg: str) -> bool:
    """Check if error suggests tool params unsupported."""
    indicators = [
        "tool_choice",
        "does not support parameters",
        "unsupportedparams",
        "unsupported params",
        "drop_params",
        "unknown parameter",
        "tools",
        "functions",
    ]

    err_lower = err_msg.lower()

    return any(ind in err_lower for ind in indicators)


# ---------------------------------------------------------
# Message conversion
# ---------------------------------------------------------


def messages_to_responses_input(messages):
    """Convert chat-completions style messages into Responses API input."""
    result = []

    for msg in messages:
        content = msg.get("content", "")

        if isinstance(content, list):
            converted_content = content
        else:
            converted_content = [
                {
                    "type": "input_text",
                    "text": str(content),
                }
            ]

        result.append(
            {
                "role": msg["role"],
                "content": converted_content,
            }
        )

    return result


# ---------------------------------------------------------
# Client setup
# ---------------------------------------------------------


def _setup_openai_client(cfg: Config | None = None):
    """Initialize client pool."""
    global _client_iterator

    if _client_iterator is not None:
        return

    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://www.litellm.org",
    )

    api_keys_env = os.getenv("LEAF_API_KEYS", os.getenv("LEAF_API_KEY", ""))

    if not api_keys_env:
        raise ValueError(
            "No API keys found. Set OPENAI_API_KEYS or OPENAI_API_KEY."
        )

    api_keys = [
        k.strip()
        for k in api_keys_env.split(",")
        if k.strip()
    ]

    if not api_keys:
        raise ValueError("No valid API keys after parsing.")

    logger.info(
        f"Initializing OpenAI client pool with {len(api_keys)} keys."
    )

    client_pool = []

    for key in api_keys:
        client = openai.OpenAI(
            api_key=key,
            base_url=base_url,
            max_retries=0,
        )

        client_pool.append(client)

    _client_iterator = itertools.cycle(client_pool)


def get_next_client() -> openai.OpenAI:
    """Thread-safe round robin client selection."""
    _setup_openai_client()

    with _client_pool_lock:
        return next(_client_iterator)


# ---------------------------------------------------------
# Responses API wrapper
# ---------------------------------------------------------


def create_completion_with_rotation(**kwargs):
    """Unified Responses API call."""
    client = get_next_client()

    return client.responses.create(**kwargs)


# ---------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------


def safe_json_loads(s: str) -> dict:
    """Parse JSON safely."""
    try:
        return json.loads(s)

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(
            f"JSON parse failed: {e}. Returning raw_output fallback."
        )

        return {
            "raw_output": str(s),
        }


def fix_json_string(s: str) -> str:
    """Fix common malformed JSON."""
    if not s:
        return s

    s = s.replace("\\'", "'")

    s = re.sub(
        r':\s*None\s*([,}])',
        r': null\1',
        s,
    )

    return s


def extract_json_from_content(content: str) -> str:
    """Extract JSON from markdown or raw text."""
    if not content:
        return ""

    backticks = chr(96) * 3

    pattern = rf"{backticks}(?:json)?\s*\n?(.*?){backticks}"

    match = re.search(
        pattern,
        content,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:
        return content[start : end + 1].strip()

    return content.strip()


# ---------------------------------------------------------
# Responses API parsing helpers
# ---------------------------------------------------------


def extract_response_text(response) -> str:
    """Extract plain text from Responses API object."""

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    texts = []

    for item in getattr(response, "output", []):

        if getattr(item, "type", None) != "message":
            continue

        for c in getattr(item, "content", []):

            ctype = getattr(c, "type", None)

            if ctype in ["output_text", "text"]:
                text = getattr(c, "text", "")

                if text:
                    texts.append(text)

    return "\n".join(texts)


def extract_tool_calls(response):
    """Extract function calls from Responses API."""
    tool_calls = []

    for item in getattr(response, "output", []):

        if getattr(item, "type", None) in [
            "function_call",
            "tool_call",
        ]:
            tool_calls.append(item)

    return tool_calls


# ---------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------


def _build_response_format(func_spec: FunctionSpec) -> dict:
    """Build response_format JSON Schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": func_spec.name,
            "description": func_spec.description,
            "schema": func_spec.json_schema,
            "strict": True,
        },
    }


def _validate_and_fix_output(
    output: dict,
    func_spec: FunctionSpec,
) -> dict:
    """Validate structured output."""
    schema = func_spec.json_schema

    required_fields = schema.get("required", [])
    properties = schema.get("properties", {})

    fixed = dict(output)

    for field in required_fields:

        if field not in fixed:

            prop = properties.get(field, {})
            field_type = prop.get("type", "string")

            if field_type == "boolean":
                fixed[field] = False

            elif field_type == "number":
                fixed[field] = None

            elif field_type == "string":
                fixed[field] = ""

            elif field_type == "array":
                fixed[field] = []

            elif field_type == "object":
                fixed[field] = {}

            else:
                fixed[field] = None

            logger.warning(
                f"Missing required field '{field}'. "
                f"Using default: {fixed[field]}"
            )

    return fixed


# ---------------------------------------------------------
# Main query API
# ---------------------------------------------------------


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:

    """Unified Responses API query."""

    _setup_openai_client(cfg)

    filtered_kwargs = select_values(notnone, model_kwargs)

    messages = opt_messages_to_list(
        system_message,
        user_message,
    )

    # ---------------------------------------------------------
    # Structured output hint
    # ---------------------------------------------------------

    use_response_format = False
    use_tool_calling = False

    if func_spec is not None:

        filtered_kwargs["response_format"] = (
            _build_response_format(func_spec)
        )

        use_response_format = True

        schema_hint = (
            "\n\nIMPORTANT: "
            "Return ONLY valid JSON matching this schema:\n"
            f"{json.dumps(func_spec.json_schema, indent=2, ensure_ascii=False)}"
        )

        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += schema_hint

    responses_input = messages_to_responses_input(messages)

    # ---------------------------------------------------------
    # Request execution
    # ---------------------------------------------------------

    t0 = time.time()

    completion = None

    model_name = filtered_kwargs.get("model")

    try:

        with _concurrency_semaphore:

            completion = backoff_create(
                create_completion_with_rotation,
                OPENAI_TIMEOUT_EXCEPTIONS,
                input=responses_input,
                **filtered_kwargs,
            )

    except openai.BadRequestError as e:

        err_msg = str(e).lower()

        # ---------------------------------------------------------
        # response_format -> tools fallback
        # ---------------------------------------------------------

        if (
            use_response_format
            and any(
                k in err_msg
                for k in [
                    "response_format",
                    "json_schema",
                    "unsupported",
                    "invalid",
                    "rejected",
                ]
            )
        ):

            logger.warning(
                "response_format rejected. "
                "Falling back to tool calling."
            )

            filtered_kwargs.pop("response_format", None)

            use_response_format = False

            filtered_kwargs["tools"] = [
                func_spec.as_openai_tool_dict
            ]

            filtered_kwargs["tool_choice"] = (
                func_spec.openai_tool_choice_dict
            )

            filtered_kwargs = _sanitize_kwargs_for_azure(
                filtered_kwargs,
                model_name,
            )

            use_tool_calling = "tools" in filtered_kwargs

            try:

                with _concurrency_semaphore:

                    completion = backoff_create(
                        create_completion_with_rotation,
                        OPENAI_TIMEOUT_EXCEPTIONS,
                        input=responses_input,
                        **filtered_kwargs,
                    )

            except openai.BadRequestError as e2:

                err_msg2 = str(e2).lower()

                # ---------------------------------------------------------
                # tools -> plain text fallback
                # ---------------------------------------------------------

                if _should_strip_tool_params(err_msg2):

                    logger.warning(
                        "Tool calling rejected. "
                        "Falling back to plain text."
                    )

                    filtered_kwargs.pop("tools", None)
                    filtered_kwargs.pop("tool_choice", None)

                    use_tool_calling = False

                    with _concurrency_semaphore:

                        completion = backoff_create(
                            create_completion_with_rotation,
                            OPENAI_TIMEOUT_EXCEPTIONS,
                            input=responses_input,
                            **filtered_kwargs,
                        )

                else:
                    raise

        else:
            raise

    if completion is None:
        raise RuntimeError(
            "API call returned None."
        )

    req_time = time.time() - t0

    # ---------------------------------------------------------
    # Output parsing
    # ---------------------------------------------------------

    if func_spec is None:

        output = extract_response_text(completion)

    elif use_response_format:

        content = extract_response_text(completion)

        if content:

            fixed_content = fix_json_string(content)

            output = safe_json_loads(fixed_content)

            if (
                isinstance(output, dict)
                and "raw_output" not in output
            ):
                output = _validate_and_fix_output(
                    output,
                    func_spec,
                )

        else:

            output = {
                "raw_output": "",
                "error": "Empty response",
            }

    elif use_tool_calling:

        tool_calls = extract_tool_calls(completion)

        if tool_calls:

            try:

                tool_call = tool_calls[0]

                raw_args = getattr(tool_call, "arguments", None)

                if raw_args is None:
                    raw_args = getattr(
                        getattr(tool_call, "function", None),
                        "arguments",
                        "{}",
                    )

                output = safe_json_loads(
                    fix_json_string(raw_args)
                )

                if (
                    isinstance(output, dict)
                    and "raw_output" not in output
                ):
                    output = _validate_and_fix_output(
                        output,
                        func_spec,
                    )

            except Exception as e:

                logger.error(
                    f"Tool parsing failed: {e}"
                )

                output = {
                    "raw_output": str(tool_calls[0]),
                }

        else:

            content = extract_response_text(completion)

            raw_args = extract_json_from_content(content)

            output = safe_json_loads(
                fix_json_string(raw_args)
            )

            if (
                isinstance(output, dict)
                and "raw_output" not in output
            ):
                output = _validate_and_fix_output(
                    output,
                    func_spec,
                )

    else:

        content = extract_response_text(completion)

        raw_args = extract_json_from_content(content)

        output = safe_json_loads(
            fix_json_string(raw_args)
        )

        if (
            isinstance(output, dict)
            and "raw_output" not in output
        ):
            output = _validate_and_fix_output(
                output,
                func_spec,
            )

    # ---------------------------------------------------------
    # Usage stats
    # ---------------------------------------------------------

    usage = getattr(completion, "usage", None)

    in_tokens = (
        getattr(usage, "input_tokens", 0)
        if usage
        else 0
    )

    out_tokens = (
        getattr(usage, "output_tokens", 0)
        if usage
        else 0
    )

    info = {
        "id": getattr(completion, "id", None),
        "model": getattr(completion, "model", "unknown"),
        "created": getattr(completion, "created", None),
    }

    if func_spec is not None:
        logger.info(
            f"Structured output keys: "
            f"{list(output.keys()) if isinstance(output, dict) else 'N/A'}"
        )

    return (
        output,
        req_time,
        in_tokens,
        out_tokens,
        info,
    )