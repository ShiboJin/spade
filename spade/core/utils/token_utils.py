"""Token accounting helpers."""

from typing import Any


def get_observation_delta(tokenizer, messages, prefix_tokens, chat_template_kwargs=None):
    """Append a rendered observation without rewriting sampled model tokens.

    Compare against the *actual* sampled prefix, including EOS and thinking
    tokens, rather than a re-rendered assistant turn that may add whitespace.
    Fail if the template would remove/rewrite previous reasoning.
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    prefix = tokenizer.decode(prefix_tokens, skip_special_tokens=False,
                              clean_up_tokenization_spaces=False)
    if not rendered.startswith(prefix):
        raise ValueError("Chat template rewrites the sampled prefix; check thinking preservation")
    delta = tokenizer.encode(rendered[len(prefix):], add_special_tokens=False)
    return delta, [0] * len(delta)


def get_token_delta(
    tokenizer: Any,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Return tokens and loss mask contributed by the newest message."""
    if not messages:
        return [], []
    is_assistant = messages[-1]["role"] == "assistant"
    extra_kwargs = {}
    if tools:
        extra_kwargs["tools"] = tools
    if chat_template_kwargs:
        extra_kwargs.update(chat_template_kwargs)

    current = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not is_assistant,
        **extra_kwargs,
    )
    previous = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=is_assistant,
        **extra_kwargs,
    )
    tokens = tokenizer.encode(current[len(previous) :], add_special_tokens=False)
    return tokens, [int(is_assistant)] * len(tokens)
