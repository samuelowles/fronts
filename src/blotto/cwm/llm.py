"""Provider abstraction for the synthesis loop.

The loop in ``blotto.cwm.synth`` needs exactly one thing from a language
model: turn (system, user) into text. Keeping that behind a Protocol means
the entire synthesis pipeline -- prompt building, code extraction, sandbox
loading, test evaluation, refinement -- runs with zero network and zero
provider SDKs installed, against fixtures recorded from real sessions.

Why fixtures rather than mocks: a mock asserts what the caller already
believes about the shape of a response, while a recorded fixture carries the
messiness of a real one -- prose around the code, multiple fenced blocks, an
apology before the fix. Tests must exercise extraction against that mess, so
the fixtures are committed and the client that produced them ships alongside
the client that replays them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

__all__ = [
    "LLMClient",
    "FixtureMiss",
    "prompt_key",
    "RecordedClient",
    "RecordingClient",
    "AnthropicClient",
    "OpenAIClient",
    "extract_code",
]


class LLMClient(Protocol):
    """Anything that can complete a (system, user) prompt pair."""

    def complete(
        self,
        system: str,
        user: str,
        stop: list[str] | None = None,
    ) -> str: ...


def prompt_key(system: str, user: str) -> str:
    """Return the fixture key for a prompt pair.

    A NUL separator so that ``("ab", "c")`` and ``("a", "bc")`` hash
    differently; without it a key collision would silently replay the wrong
    response and the failure would surface as a mysterious synthesis error
    rather than a fixture problem.
    """
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()


class FixtureMiss(Exception):
    """A prompt reached ``RecordedClient`` with no recorded response.

    Carries the missing key so a failing test names exactly what to
    re-record, and the fix is a mechanical re-run of ``RecordingClient``
    rather than an investigation.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(
            f"no fixture recorded for prompt key {key!r}; re-record it by "
            "wrapping the real client in RecordingClient and re-running the "
            "session that produced this prompt"
        )


@dataclass
class RecordedClient:
    """Replay responses from a JSONL fixture. The default client for tests.

    Each fixture line is ``{"key": <sha256 of system+user>, "response": str}``
    -- exactly what ``RecordingClient`` writes. Unknown keys raise
    ``FixtureMiss`` rather than returning something plausible, because a
    synthesised-response fallback here would let the test suite drift away
    from what the provider actually returns while still passing.
    """

    path: Path

    def __post_init__(self) -> None:
        self._responses: dict[str, str] = {}
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._responses[entry["key"]] = entry["response"]

    def complete(
        self,
        system: str,
        user: str,
        stop: list[str] | None = None,
    ) -> str:
        key = prompt_key(system, user)
        try:
            return self._responses[key]
        except KeyError as exc:
            raise FixtureMiss(key) from exc


@dataclass
class RecordingClient:
    """Wrap a real client and append every exchange to a fixture file.

    This is how fixtures get made: run the synthesis loop once against a live
    provider with this wrapper, and the result is a JSONL file ``RecordedClient``
    can replay forever after. Appends rather than overwrites, so several
    sessions can contribute to one fixture.
    """

    inner: LLMClient
    path: Path

    def complete(
        self,
        system: str,
        user: str,
        stop: list[str] | None = None,
    ) -> str:
        response = self.inner.complete(system, user, stop)
        record = {"key": prompt_key(system, user), "response": response}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return response


@dataclass
class AnthropicClient:
    """Anthropic SDK client. The SDK is imported inside ``complete`` so that
    importing this module -- and everything downstream of it -- succeeds on a
    machine with no network and no extras installed."""

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8192
    _client: object | None = field(default=None, repr=False, compare=False)

    def complete(
        self,
        system: str,
        user: str,
        stop: list[str] | None = None,
    ) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only with SDK absent
            raise ImportError(
                'the anthropic SDK is not installed; run pip install "blotto[llm]"'
            ) from exc
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            stop_sequences=stop or [],
        )
        return "".join(block.text for block in message.content if block.type == "text")


@dataclass
class OpenAIClient:
    """OpenAI SDK client, lazily imported for the same reason as above."""

    model: str = "gpt-4.1"
    max_tokens: int = 8192
    _client: object | None = field(default=None, repr=False, compare=False)

    def complete(
        self,
        system: str,
        user: str,
        stop: list[str] | None = None,
    ) -> str:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - exercised only with SDK absent
            raise ImportError(
                'the openai SDK is not installed; run pip install "blotto[llm]"'
            ) from exc
        if self._client is None:
            self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stop=stop,
        )
        return completion.choices[0].message.content or ""


_FENCE = re.compile(r"```[ \t]*(?:\w+)?[ \t]*\r?\n(.*?)```", re.DOTALL)
"""Matches a fenced block with or without a language tag. The optional tag
group is deliberately not captured: ``extract_code`` wants the code, and a
leading ``python`` marker is formatting, not content."""


def extract_code(response: str) -> str:
    """Pull Python out of a model response.

    Models wrap code in fences, sometimes after a paragraph of apology, and
    occasionally emit several blocks (one broken attempt, then the fix).
    Taking the LONGEST block implements that last case: the fix is longer
    than the snippet it patches. With no fence at all -- rare but it happens,
    especially on refinement passes -- the whole response is the code, and
    the sandbox will reject it if it is not.
    """
    blocks = [match.group(1) for match in _FENCE.finditer(response)]
    if not blocks:
        return response.strip()
    return max(blocks, key=len).strip()
