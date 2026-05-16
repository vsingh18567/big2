from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from big2.play.agents import GreedyAgent, LLMPlayAgent, SmartAgent, TrainedPPOAgent
from big2.play.types import PlayAgent


class AgentType(Enum):
    GREEDY = "greedy"
    SMART = "smart"
    TRAINED_PPO = "trained_ppo"
    OPENAI_GPT_5_4 = "openai/gpt-5.4"
    OPENAI_GPT_5_4_MINI = "openai/gpt-5.4-mini"
    ANTHROPIC_CLAUDE_SONNET_4_6 = "anthropic/claude-sonnet-4.6"
    ANTHROPIC_CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4.5"
    MOONSHOTAI_KIMI_K2_5 = "moonshotai/kimi-k2.5"
    DEEPSEEK_DEEPSEEK_V3_2 = "deepseek/deepseek-v3.2"
    GEMMA_4 = "google/gemma-4-31b-it"
    GPT_OSS = 'openai/gpt-oss-120b'
    GEMINI_3_FLASH = 'google/gemini-3-flash-preview'

@dataclass(frozen=True)
class AgentConfig:
    agent_type: AgentType
    checkpoint_path: str | None = None
    device: str = "cpu"
    reasoning_enabled: bool = True


@dataclass(frozen=True)
class ArenaEntrantConfig(AgentConfig):
    id: str = ""


@dataclass(frozen=True)
class ArenaConfig:
    entrants: tuple[ArenaEntrantConfig, ...]


LLM_AGENT_TYPES = {
    AgentType.OPENAI_GPT_5_4,
    AgentType.OPENAI_GPT_5_4_MINI,
    AgentType.ANTHROPIC_CLAUDE_SONNET_4_6,
    AgentType.ANTHROPIC_CLAUDE_HAIKU_4_5,
    AgentType.MOONSHOTAI_KIMI_K2_5,
    AgentType.DEEPSEEK_DEEPSEEK_V3_2,
    AgentType.GEMMA_4,
    AgentType.GPT_OSS,
    AgentType.GEMINI_3_FLASH,
}


def parse_agent_type(value: str) -> AgentType:
    try:
        return AgentType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported agent type: {value}") from exc


def load_arena_config(path: str | Path) -> ArenaConfig:
    config_path = Path(path)
    with config_path.open() as handle:
        payload = json.load(handle)

    entrants_payload = payload.get("entrants")
    if not isinstance(entrants_payload, list) or not entrants_payload:
        raise ValueError("Arena config must contain a non-empty 'entrants' list")

    entrants: list[ArenaEntrantConfig] = []
    seen_ids: set[str] = set()
    for raw_entrant in entrants_payload:
        if not isinstance(raw_entrant, dict):
            raise ValueError("Each entrant must be an object")
        entrant_id = raw_entrant.get("id")
        if not isinstance(entrant_id, str) or not entrant_id:
            raise ValueError("Each entrant must define a non-empty 'id'")
        if entrant_id in seen_ids:
            raise ValueError(f"Duplicate entrant id: {entrant_id}")
        seen_ids.add(entrant_id)
        entrants.append(
            ArenaEntrantConfig(
                id=entrant_id,
                agent_type=parse_agent_type(raw_entrant["agent_type"]),
                checkpoint_path=raw_entrant.get("checkpoint_path"),
                device=raw_entrant.get("device", "cpu"),
                reasoning_enabled=raw_entrant.get("reasoning_enabled", True),
            )
        )

    return ArenaConfig(entrants=tuple(entrants))


def load(config: AgentConfig) -> PlayAgent:
    if config.agent_type == AgentType.GREEDY:
        return GreedyAgent()
    if config.agent_type == AgentType.SMART:
        return SmartAgent()
    if config.agent_type == AgentType.TRAINED_PPO:
        if config.checkpoint_path is None:
            raise ValueError("TRAINED_PPO requires checkpoint_path")
        return TrainedPPOAgent.from_checkpoint(config.checkpoint_path, device=config.device)
    if config.agent_type in LLM_AGENT_TYPES:
        return LLMPlayAgent(
            model=config.agent_type.value,
            reasoning_enabled=config.reasoning_enabled,
        )
    raise ValueError(f"Unsupported agent type: {config.agent_type}")
