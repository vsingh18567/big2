import json
from types import SimpleNamespace

import numpy as np
import pytest

from big2.play.agents import GreedyAgent, LLMPlayAgent, SmartLLMAgent, SmartAgent
from big2.play.types import OpponentView, PlayTurn
from big2.simulator.cards import FOUR_KIND, PAIR, PASS, SINGLE, Combo


def make_turn(
    *,
    hand: tuple[int, ...],
    legal_actions: tuple[Combo, ...],
    current_trick: Combo | None = None,
) -> PlayTurn:
    return PlayTurn(
        actor_id=0,
        hand=hand,
        legal_actions=legal_actions,
        current_trick=current_trick,
        passes_in_row=0,
        opponents=(
            OpponentView(player_id=1, cards_left=13, is_current=False),
            OpponentView(player_id=2, cards_left=13, is_current=False),
            OpponentView(player_id=3, cards_left=13, is_current=False),
        ),
        played_cards_by_player=((), (), (), ()),
        state_vector=np.zeros(1, dtype=np.int32),
    )


def test_greedy_agent_chooses_lowest_non_pass_combo() -> None:
    agent = GreedyAgent()
    low_single = Combo(SINGLE, [0], (0, 0))
    high_single = Combo(SINGLE, [4], (1, 0))
    turn = make_turn(hand=(0, 4), legal_actions=(Combo(PASS, [], ()), high_single, low_single))

    choice = agent.choose_action(turn)

    assert choice.action == low_single


def test_smart_agent_passes_on_unbeatable_four_of_a_kind() -> None:
    agent = SmartAgent()
    hand = (48, 49, 0)
    pass_action = Combo(PASS, [], ())
    power_pair = Combo(PAIR, [48, 49], (12, 1))
    current_trick = Combo(FOUR_KIND, [4, 5, 6, 7, 8], (1,))
    turn = make_turn(hand=hand, legal_actions=(pass_action, power_pair), current_trick=current_trick)

    choice = agent.choose_action(turn)

    assert choice.action == pass_action


def test_llm_agent_uses_action_id_response_and_includes_history_in_prompt() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    high_single = Combo(SINGLE, [4], (1, 0))
    turn = make_turn(hand=(0, 4), legal_actions=(Combo(PASS, [], ()), low_single, high_single))

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A2"}'))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3, total_tokens=14),
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(send=lambda **_: fake_response))
    agent = LLMPlayAgent(client=fake_client)
    agent.observe_action(1, high_single)

    choice = agent.choose_action(turn)

    assert choice.action == low_single
    assert "H:" in choice.metadata["llm_prompt"]
    assert "P1 4♦" in choice.metadata["llm_prompt"]
    assert "Hand:" in choice.metadata["llm_prompt"]
    assert choice.metadata["llm_usage"] == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}


def test_llm_agent_sends_session_id_in_extra_body() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))
    captured: dict[str, object] = {}

    def fake_send(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A1"}'))]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fake_send))
    agent = LLMPlayAgent(client=fake_client, debug=False)
    agent.set_game_session_id("game-123")

    choice = agent.choose_action(turn)

    assert choice.action == low_single
    assert captured["extra_body"]["session_id"] == "game-123"
    assert "temperature" not in captured


def test_llm_agent_omits_reasoning_block_when_disabled() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))
    captured: dict[str, object] = {}

    def fake_send(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A1"}'))]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fake_send))
    agent = LLMPlayAgent(client=fake_client, debug=False, reasoning_enabled=False)

    choice = agent.choose_action(turn)

    assert choice.action == low_single
    assert captured["extra_body"] == {"provider": {"require_parameters": True}}


def test_llm_agent_accepts_pass_response() -> None:
    pass_action = Combo(PASS, [], ())
    power_pair = Combo(PAIR, [48, 49], (12, 1))
    turn = make_turn(hand=(48, 49), legal_actions=(pass_action, power_pair))

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A1"}'))]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(send=lambda **_: fake_response))
    agent = LLMPlayAgent(client=fake_client)

    choice = agent.choose_action(turn)

    assert choice.action == pass_action


def test_llm_agent_skips_call_when_pass_is_only_legal_action() -> None:
    pass_action = Combo(PASS, [], ())
    turn = make_turn(hand=(48, 49), legal_actions=(pass_action,))

    def fail_send(**_kwargs):
        raise AssertionError("LLM client should not be called on forced pass")

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fail_send))
    agent = LLMPlayAgent(client=fake_client)

    choice = agent.choose_action(turn)

    assert choice.action == pass_action
    assert choice.metadata["llm_skipped"] is True
    assert choice.metadata["llm_skip_reason"] == "forced_pass"
    assert choice.metadata["llm_usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert choice.metadata["llm_elapsed_ms"] == 0


def test_llm_agent_retries_rate_limit_errors(monkeypatch) -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))
    calls = {"count": 0}
    sleeps: list[float] = []

    class RetryableError(Exception):
        status_code = 429

    def fake_send(**_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise RetryableError("rate limited")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A1"}'))]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fake_send))
    monkeypatch.setattr("big2.play.agents.llm_agent.time.sleep", lambda seconds: sleeps.append(seconds))
    agent = LLMPlayAgent(client=fake_client, debug=False, max_retries=3, retry_backoff_seconds=0.5)

    choice = agent.choose_action(turn)

    assert choice.action == low_single
    assert calls["count"] == 3
    assert sleeps == [0.5, 1.0]


def test_llm_agent_retries_response_json_decode_errors(monkeypatch) -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_send(**_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise json.JSONDecodeError("Expecting value", "not-json", 0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A1"}'))]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fake_send))
    monkeypatch.setattr("big2.play.agents.llm_agent.time.sleep", lambda seconds: sleeps.append(seconds))
    agent = LLMPlayAgent(client=fake_client, debug=False, max_retries=3, retry_backoff_seconds=0.5)

    choice = agent.choose_action(turn)

    assert choice.action == low_single
    assert calls["count"] == 3
    assert sleeps == [0.5, 1.0]


def test_llm_agent_retries_openrouter_routing_404(monkeypatch) -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))
    calls = {"count": 0}
    request_payloads: list[dict[str, object]] = []
    sleeps: list[float] = []

    class OpenRouterRoutingError(Exception):
        status_code = 404

    def fake_send(**kwargs):
        calls["count"] += 1
        request_payloads.append(kwargs)
        if calls["count"] == 1:
            raise OpenRouterRoutingError(
                "Error code: 404 - {'error': {'message': 'No endpoints found that can handle "
                "the requested parameters. To learn more about provider routing, visit: "
                "https://openrouter.ai/docs/guides/routing/provider-selection', 'code': 404}}"
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A1"}'))]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fake_send))
    monkeypatch.setattr("big2.play.agents.llm_agent.time.sleep", lambda seconds: sleeps.append(seconds))
    agent = LLMPlayAgent(client=fake_client, debug=False, max_retries=3, retry_backoff_seconds=0.5)

    choice = agent.choose_action(turn)

    assert choice.action == low_single
    assert calls["count"] == 2
    assert request_payloads[0]["extra_body"] == {"provider": {"require_parameters": True}, "reasoning": {"enabled": True, "effort": "low", "exclude": True}}
    assert request_payloads[1]["extra_body"] == {"reasoning": {"enabled": True, "effort": "low", "exclude": True}}
    assert sleeps == []


def test_llm_agent_retries_once_on_empty_response_payload() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))
    calls = {"count": 0}

    def fake_send(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            refusal=None,
                            tool_calls=None,
                            reasoning=None,
                        )
                    )
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A1"}'))]
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fake_send))
    agent = LLMPlayAgent(client=fake_client, debug=False, empty_response_retries=1)

    choice = agent.choose_action(turn)

    assert choice.action == low_single
    assert calls["count"] == 2


def test_llm_agent_does_not_retry_non_retryable_errors(monkeypatch) -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))
    calls = {"count": 0}
    sleeps: list[float] = []

    class NonRetryableError(Exception):
        pass

    def fake_send(**_kwargs):
        calls["count"] += 1
        raise NonRetryableError("bad request")

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fake_send))
    monkeypatch.setattr("big2.play.agents.llm_agent.time.sleep", lambda seconds: sleeps.append(seconds))
    agent = LLMPlayAgent(client=fake_client, debug=False, max_retries=3, retry_backoff_seconds=0.5)

    with pytest.raises(NonRetryableError):
        agent.choose_action(turn)

    assert calls["count"] == 1
    assert sleeps == []


def test_llm_agent_does_not_retry_other_404_errors(monkeypatch) -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))
    calls = {"count": 0}
    sleeps: list[float] = []

    class OtherNotFoundError(Exception):
        status_code = 404

    def fake_send(**_kwargs):
        calls["count"] += 1
        raise OtherNotFoundError("model not found")

    fake_client = SimpleNamespace(chat=SimpleNamespace(send=fake_send))
    monkeypatch.setattr("big2.play.agents.llm_agent.time.sleep", lambda seconds: sleeps.append(seconds))
    agent = LLMPlayAgent(client=fake_client, debug=False, max_retries=3, retry_backoff_seconds=0.5)

    with pytest.raises(OtherNotFoundError):
        agent.choose_action(turn)

    assert calls["count"] == 1
    assert sleeps == []


def test_llm_agent_accepts_list_text_content_response() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    high_single = Combo(SINGLE, [4], (1, 0))
    turn = make_turn(hand=(0, 4), legal_actions=(Combo(PASS, [], ()), low_single, high_single))

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=[{"type": "output_text", "text": '{"action_id":"A3"}'}])
            )
        ]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(send=lambda **_: fake_response))
    agent = LLMPlayAgent(client=fake_client)

    choice = agent.choose_action(turn)

    assert choice.action == high_single


def test_llm_agent_accepts_fenced_json_response() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(low_single,))

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"action_id":"A1"}\n```'))]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(send=lambda **_: fake_response))
    agent = LLMPlayAgent(client=fake_client)

    choice = agent.choose_action(turn)

    assert choice.action == low_single


def test_llm_agent_raises_when_response_is_invalid() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    high_single = Combo(SINGLE, [4], (1, 0))
    turn = make_turn(hand=(0, 4), legal_actions=(Combo(PASS, [], ()), low_single, high_single))

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A99"}'))]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(send=lambda **_: fake_response))
    agent = LLMPlayAgent(client=fake_client)

    with pytest.raises(ValueError, match="illegal action_id"):
        agent.choose_action(turn)


def test_llm_agent_raises_with_refusal_details() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(Combo(PASS, [], ()), low_single))
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="I can't help with that"))]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(send=lambda **_: fake_response))
    agent = LLMPlayAgent(client=fake_client)

    with pytest.raises(ValueError, match="response was a refusal"):
        agent.choose_action(turn)


def test_llm_agent_raises_without_api_key() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    turn = make_turn(hand=(0,), legal_actions=(Combo(PASS, [], ()), low_single))
    agent = LLMPlayAgent(client=None, api_key=None)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent.choose_action(turn)


def test_smart_llm_agent_centers_prompt_on_legal_actions() -> None:
    low_single = Combo(SINGLE, [0], (0, 0))
    high_single = Combo(SINGLE, [4], (1, 0))
    turn = make_turn(hand=(0, 4), legal_actions=(Combo(PASS, [], ()), low_single, high_single))

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"action_id":"A3"}'))]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(send=lambda **_: fake_response))
    agent = SmartLLMAgent(client=fake_client)

    choice = agent.choose_action(turn)

    assert choice.action == high_single
    prompt = choice.metadata["llm_prompt"]
    assert "Actions are exhaustive" in prompt
    assert "Actions:\nA1 pass\nA2 3♦\nA3 4♦" in prompt
