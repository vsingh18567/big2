from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, DefaultHttpxClient, OpenAI, RateLimitError

from big2.play.logging_utils import log_timestamp
from big2.play.types import PlayAction, PlayTurn
from big2.simulator.cards import (
    FLUSH,
    FOUR_KIND,
    FULLHOUSE,
    PAIR,
    PASS,
    SINGLE,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPLE,
    Combo,
    card_name,
    sort_cards,
)

COMBO_TYPE_NAMES = {
    PASS: "pass",
    SINGLE: "single",
    PAIR: "pair",
    TRIPLE: "triple",
    STRAIGHT: "straight",
    FLUSH: "flush",
    FULLHOUSE: "full house",
    FOUR_KIND: "four of a kind",
    STRAIGHT_FLUSH: "straight flush",
}

HTTPX_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=60,
    keepalive_expiry=100.0,
)


def _format_cards(cards: list[int] | tuple[int, ...]) -> str:
    return " ".join(card_name(card) for card in sort_cards(list(cards)))


def _format_combo(combo: Combo) -> str:
    if combo.type == PASS:
        return "pass"
    combo_name = COMBO_TYPE_NAMES.get(combo.type, f"type {combo.type}")
    return f"{combo_name}: {_format_cards(combo.cards)}"


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("LLM response did not contain any choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("LLM response choice did not contain a message")
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            item_text = getattr(item, "text", None)
            if isinstance(item_text, str):
                text_parts.append(item_text)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        extracted = "\n".join(text_parts).strip()
        if extracted:
            return extracted

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        raise ValueError(f"LLM response was a refusal: {refusal.strip()}")

    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        reasoning = reasoning.strip()
        json_match = re.search(r"\{.*\}", reasoning, re.DOTALL)
        if json_match:
            return json_match.group(0)
        raise ValueError(f"LLM response had reasoning but no action JSON: {reasoning}")

    raise ValueError(f"LLM response content was empty or in an unsupported format: {message!r}")


def _is_empty_response_payload(response: Any) -> bool:
    choices = getattr(response, "choices", None)
    if not choices:
        return False
    message = getattr(choices[0], "message", None)
    if message is None:
        return False
    return (
        getattr(message, "content", None) is None
        and getattr(message, "refusal", None) is None
        and getattr(message, "tool_calls", None) in (None, [])
        and getattr(message, "reasoning", None) is None
    )


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _build_response_format(legal_action_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "big2_action_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "One legal action ID from the provided list.",
                        "enum": legal_action_ids,
                    }
                },
                "required": ["action_id"],
                "additionalProperties": False,
            },
        },
    }


def _parse_structured_action_response(response_text: str, legal_actions: tuple[Combo, ...]) -> Combo:
    normalized = response_text.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].startswith("```") and lines[-1].strip() == "```":
            normalized = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {response_text!r}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"LLM response JSON must be an object: {response_text!r}")

    action_id = payload.get("action_id")
    if not isinstance(action_id, str):
        raise ValueError(f"LLM response JSON missing string action_id: {response_text!r}")

    action_map = {f"A{index}": action for index, action in enumerate(legal_actions, start=1)}
    try:
        return action_map[action_id]
    except KeyError as exc:
        raise ValueError(f"LLM response returned illegal action_id {action_id!r}") from exc


def _status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _is_retryable_openrouter_routing_404(exc: Exception) -> bool:
    if _status_code_from_exception(exc) != 404:
        return False
    error_message = str(exc)
    return (
        "No endpoints found that can handle the requested parameters" in error_message
        or "provider routing" in error_message.lower()
    )


def _relax_provider_require_parameters(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    extra_body = request_kwargs.get("extra_body")
    if not isinstance(extra_body, dict):
        return request_kwargs
    provider = extra_body.get("provider")
    if not isinstance(provider, dict) or "require_parameters" not in provider:
        return request_kwargs

    relaxed_request_kwargs = dict(request_kwargs)
    relaxed_extra_body = dict(extra_body)
    relaxed_provider = dict(provider)
    relaxed_provider.pop("require_parameters", None)
    if relaxed_provider:
        relaxed_extra_body["provider"] = relaxed_provider
    else:
        relaxed_extra_body.pop("provider", None)
    relaxed_request_kwargs["extra_body"] = relaxed_extra_body
    return relaxed_request_kwargs


def _is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (json.JSONDecodeError, RateLimitError, APIConnectionError, APITimeoutError, httpx.TimeoutException, httpx.ConnectError)):
        return True
    if _is_retryable_openrouter_routing_404(exc):
        return True
    return _status_code_from_exception(exc) in {408, 429, 500, 502, 503, 504}


@dataclass
class LLMPlayAgent:
    model: str = "minimax/minimax-m2"
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    reasoning_enabled: bool = True
    reasoning_effort: str = "low"
    reasoning_max_tokens: int | None = None
    timeout_ms: int = 60000
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    empty_response_retries: int = 1
    debug: bool = True
    client: Any | None = None
    session_id: str | None = None
    move_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        resolved_api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if resolved_api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=resolved_api_key,
                http_client=DefaultHttpxClient(
                    limits=HTTPX_LIMITS,
                ),
            )

    def choose_action(self, turn: PlayTurn) -> PlayAction:
        if len(turn.legal_actions) == 1 and turn.legal_actions[0].type == PASS:
            return PlayAction(
                action=turn.legal_actions[0],
                metadata={
                    "llm_skipped": True,
                    "llm_skip_reason": "forced_pass",
                    "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "llm_elapsed_ms": 0,
                },
            )
        prompt = self._build_user_prompt(turn)
        legal_action_ids = [self._action_id(index) for index in range(1, len(turn.legal_actions) + 1)]

        if self.client is None:
            raise RuntimeError(f"OPENROUTER_API_KEY is not set for LLM agent model {self.model}")

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "timeout": self.timeout_ms / 1000,
            "response_format": _build_response_format(legal_action_ids),
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "extra_body": {
                "provider": {"require_parameters": True},
            },
        }
        if self.reasoning_enabled:
            request_kwargs["extra_body"]["reasoning"] = {
                "enabled": True,
                "effort": self.reasoning_effort,
                "exclude": True,
            }
            if self.reasoning_max_tokens is not None:
                request_kwargs["extra_body"]["reasoning"]["max_tokens"] = self.reasoning_max_tokens
        if self.session_id is not None:
            request_kwargs["extra_body"]["session_id"] = self.session_id
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        if self.debug:
            print(
                f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} request_start legal_action_ids={legal_action_ids}",
                file=sys.stderr,
                flush=True,
            )
            if self.session_id is not None:
                print(
                    f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} session_id={self.session_id}",
                    file=sys.stderr,
                    flush=True,
                )

        request_started_at = time.perf_counter()
        last_error: Exception | None = None
        relaxed_provider_requirements = False
        for attempt in range(self.max_retries + 1):
            try:
                if hasattr(self.client.chat, "completions") and hasattr(self.client.chat.completions, "create"):
                    response = self.client.chat.completions.create(**request_kwargs)
                else:
                    response = self.client.chat.send(**request_kwargs)
                break
            except Exception as exc:
                last_error = exc
                if _is_retryable_openrouter_routing_404(exc) and not relaxed_provider_requirements:
                    relaxed_request_kwargs = _relax_provider_require_parameters(request_kwargs)
                    if relaxed_request_kwargs is not request_kwargs:
                        request_kwargs = relaxed_request_kwargs
                        relaxed_provider_requirements = True
                        if self.debug:
                            print(
                                f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} "
                                "routing_404_relax_require_parameters",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                if attempt >= self.max_retries or not _is_retryable_llm_error(exc):
                    raise
                sleep_seconds = self.retry_backoff_seconds * (2**attempt)
                if self.debug:
                    print(
                        f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} "
                        f"retryable_error attempt={attempt + 1} sleep_seconds={sleep_seconds:.2f} error={exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(sleep_seconds)
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("LLM request failed without returning a response or exception")
        elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
        if self.debug:
            print(
                f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} response_received elapsed_ms={elapsed_ms}",
                file=sys.stderr,
                flush=True,
            )
        extract_attempt = 0
        while True:
            try:
                response_text = _extract_response_text(response)
                break
            except Exception as exc:
                should_retry_empty_response = (
                    extract_attempt < self.empty_response_retries and _is_empty_response_payload(response)
                )
                if should_retry_empty_response:
                    extract_attempt += 1
                    if self.debug:
                        print(
                            f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} "
                            f"empty_response_retry attempt={extract_attempt}",
                            file=sys.stderr,
                            flush=True,
                        )
                    request_started_at = time.perf_counter()
                    if hasattr(self.client.chat, "completions") and hasattr(self.client.chat.completions, "create"):
                        response = self.client.chat.completions.create(**request_kwargs)
                    else:
                        response = self.client.chat.send(**request_kwargs)
                    elapsed_ms += int((time.perf_counter() - request_started_at) * 1000)
                    if self.debug:
                        print(
                            f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} "
                            f"response_received elapsed_ms={elapsed_ms}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue
                if self.debug:
                    print(
                        f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} failed_to_extract_response error={exc}",
                        file=sys.stderr,
                    )
                    print(
                        f"{log_timestamp()} [LLM DEBUG] legal_action_ids={legal_action_ids} legal_actions={[self._format_legal_actions(turn.legal_actions)]}",
                        file=sys.stderr,
                    )
                    print(f"{log_timestamp()} [LLM DEBUG] raw_response={response!r}", file=sys.stderr)
                raise
        usage = _extract_usage(response)
        try:
            action = _parse_structured_action_response(response_text, turn.legal_actions)
        except Exception as exc:
            if self.debug:
                print(
                    f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} failed_to_parse_json error={exc}",
                    file=sys.stderr,
                )
                print(f"{log_timestamp()} [LLM DEBUG] usage={usage}", file=sys.stderr)
                print(
                    f"{log_timestamp()} [LLM DEBUG] legal_action_ids={legal_action_ids} legal_actions={[self._format_legal_actions(turn.legal_actions)]}",
                    file=sys.stderr,
                )
            raise
        print(
            f"{log_timestamp()} [LLM DEBUG] model={self.model} actor={turn.actor_id} action={_format_combo(action)} usage={usage}",
            file=sys.stderr,
        )
        return PlayAction(
            action=action,
            metadata={
                "llm_prompt": prompt,
                "llm_response": response_text,
                "llm_usage": usage,
                "llm_elapsed_ms": elapsed_ms,
            },
        )

    def observe_action(self, actor_id: int, action: Combo) -> None:
        self.move_history.append(f"Player {actor_id}: {_format_combo(action)}")

    def on_game_end(self, winner: int | None) -> None:
        self.move_history.clear()
        self.session_id = None

    def set_game_session_id(self, session_id: str) -> None:
        self.session_id = session_id

    def _system_prompt(self) -> str:
        return (
            "Play Big 2. Choose exactly one listed legal action. "
            "Goal: empty your hand first. "
            "Basic strategy: shed weak or awkward cards efficiently, keep strong control cards or combinations unless they gain tempo, "
            "and respect opponents with few cards left. "
            'Return ONLY JSON matching the schema, like {"action_id":"A1"}. '
            "No extra text."
        )

    def _build_user_prompt(self, turn: PlayTurn) -> str:
        return self._user_prompt_prefix() + self._build_turn_state_block(turn)

    def _user_prompt_prefix(self) -> str:
        return (
            'Choose one legal action from Actions. Legend: H=history, Opp=P#:cards_left. '
            'Output only {"action_id":"A#"}.'
            "\n\n"
        )

    def _build_turn_state_block(self, turn: PlayTurn) -> str:
        return (
            f"P:{turn.actor_id}\n"
            f"H:\n{self._format_history()}\n"
            f"Hand: {_format_cards(turn.hand)}\n"
            f"Trick: {self._format_current_trick(turn)}\n"
            f"Passes: {turn.passes_in_row}\n"
            f"Opp: {self._format_opponents(turn)}\n"
            f"Actions:\n{self._format_legal_actions(turn.legal_actions)}"
        )

    def _format_history(self) -> str:
        return "\n".join(
            f"{index}. {self._compact_history_entry(entry)}" for index, entry in enumerate(self.move_history, start=1)
        ) or "-"

    def _compact_history_entry(self, entry: str) -> str:
        compact = entry.replace("Player ", "P")
        compact = compact.replace(": single: ", " ")
        compact = compact.replace(": pair: ", " pair ")
        compact = compact.replace(": triple: ", " tri ")
        compact = compact.replace(": straight flush: ", " sf ")
        compact = compact.replace(": four of a kind: ", " 4k ")
        compact = compact.replace(": full house: ", " fh ")
        compact = compact.replace(": straight: ", " str ")
        compact = compact.replace(": flush: ", " fl ")
        compact = compact.replace(": pass", " pass")
        return compact

    def _format_current_trick(self, turn: PlayTurn) -> str:
        return "None" if turn.current_trick is None else _format_combo(turn.current_trick)

    def _format_opponents(self, turn: PlayTurn) -> str:
        return " ".join(
            f"P{opponent.player_id}:{opponent.cards_left}"
            for opponent in turn.opponents
        )

    def _format_legal_actions(self, legal_actions: tuple[Combo, ...]) -> str:
        return "\n".join(
            f"{self._action_id(index)} {self._format_compact_action(action)}"
            for index, action in enumerate(legal_actions, start=1)
        )

    def _format_compact_action(self, action: Combo) -> str:
        if action.type == PASS:
            return "pass"
        combo_name = COMBO_TYPE_NAMES.get(action.type, f"t{action.type}")
        short_name = {
            "single": "",
            "pair": "pair ",
            "triple": "tri ",
            "straight": "str ",
            "flush": "fl ",
            "full house": "fh ",
            "four of a kind": "4k ",
            "straight flush": "sf ",
        }.get(combo_name, f"{combo_name} ")
        return f"{short_name}{_format_cards(action.cards)}".strip()

    @staticmethod
    def _action_id(index: int) -> str:
        return f"A{index}"


@dataclass
class SmartLLMAgent(LLMPlayAgent):
    def _system_prompt(self) -> str:
        return (
            "Play strong Big 2. Choose exactly one listed legal action. "
            "Goal: empty your hand first. "
            "Prefer efficient shedding, preserving strong control for high-value spots, and blocking opponents who are close to going out. "
            'Return ONLY JSON matching the schema, like {"action_id":"A1"}. '
            "No extra text."
        )

    def _build_user_prompt(self, turn: PlayTurn) -> str:
        return self._smart_user_prompt_prefix() + self._build_turn_state_block(turn)

    def _smart_user_prompt_prefix(self) -> str:
        return (
            'Choose one legal action. Actions are exhaustive. Legend: H=history, Opp=P#:cards_left. '
            'Output only {"action_id":"A#"}. Prefer efficient, strong play.'
            "\n\n"
        )
