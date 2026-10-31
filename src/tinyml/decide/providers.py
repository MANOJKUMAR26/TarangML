"""Assistants that can answer a fixed-choice question when the rules are unsure.

Jev is a placeholder until it is wired in (Phase 5). For now the assistant is
Claude, used through tools the team already pays for: Claude Code in headless
mode on this machine, or Claude chat via copy and paste.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from tinyml.config import AssistantSettings


class ProviderUnavailable(RuntimeError):
    pass


@dataclass
class ProviderReply:
    choice: str | None
    confidence: float
    follow_up: str | None
    version: str
    raw: str


def build_prompt(
    decision_type: str,
    question: str,
    options: list[str],
    summary: dict,
    use_case: str,
    examples: list[dict],
) -> str:
    lines = [
        "You are helping a no-code TinyML tool make one fixed-choice decision.",
        f"Decision type: {decision_type}",
        f"Question: {question}",
        f"User's use case: {use_case or '(not given)'}",
        "Facts the tool computed (summaries only, no raw data):",
        json.dumps(summary, indent=2, default=str),
        f"Options (pick exactly one): {json.dumps(options)}",
    ]
    if examples:
        lines.append("Past decisions of this type that users corrected (most similar first):")
        for ex in examples:
            lines.append(f"- facts={json.dumps(ex['summary'], default=str)} -> user chose {ex['final_choice']!r}")
    lines += [
        "Reply with only a JSON object, no prose:",
        '{"choice": "<one option>", "confidence": <0..1>, "follow_up": "<one short plain-English question '
        'for the user if you are unsure, else null>"}',
    ]
    return "\n".join(lines)


def parse_reply(text: str, options: list[str]) -> tuple[str | None, float, str | None]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, 0.0, None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 0.0, None
    choice = data.get("choice")
    if choice not in options:
        choice = None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    follow_up = data.get("follow_up") or None
    return choice, max(0.0, min(1.0, confidence)), follow_up


class JevProvider:
    """Reserved slot for Jev. Not configured in the open-source build."""

    name = "jev"

    def available(self) -> bool:
        return False

    def ask(self, prompt: str, options: list[str]) -> ProviderReply:
        raise ProviderUnavailable("Jev is not configured")


@lru_cache(maxsize=4)
def _claude_version(command: str) -> str:
    try:
        out = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


class ClaudeCodeProvider:
    """Runs `claude -p` (Claude Code headless mode) with the user's own login."""

    name = "claude_code"

    def __init__(self, settings: AssistantSettings):
        self.settings = settings

    def available(self) -> bool:
        return shutil.which(self.settings.claude_command) is not None

    def version(self) -> str:
        version = _claude_version(self.settings.claude_command)
        return f"{version} model={self.settings.claude_model or 'default'}"

    def run(self, prompt: str, cwd: str | None = None, tools: str | None = None) -> str:
        if not self.available():
            raise ProviderUnavailable(f"'{self.settings.claude_command}' is not on PATH")
        cmd = [self.settings.claude_command, "-p", prompt, "--output-format", "json"]
        if self.settings.claude_model:
            cmd += ["--model", self.settings.claude_model]
        if tools:
            cmd += ["--allowedTools", tools]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.settings.timeout_s, cwd=cwd
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderUnavailable("Claude Code timed out") from exc
        if out.returncode != 0:
            raise ProviderUnavailable(f"Claude Code failed: {out.stderr.strip()[:300]}")
        try:
            payload = json.loads(out.stdout)
            return str(payload.get("result", ""))
        except json.JSONDecodeError:
            return out.stdout

    def ask(self, prompt: str, options: list[str]) -> ProviderReply:
        text = self.run(prompt)
        choice, confidence, follow_up = parse_reply(text, options)
        return ProviderReply(choice, confidence, follow_up, self.version(), text)


class ClaudeChatProvider:
    """Copy-and-paste with Claude chat. The UI or CLI shows the prompt; the user pastes the reply."""

    name = "claude_chat"

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return "claude-chat (manual)"
