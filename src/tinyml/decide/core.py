"""decide(question, options): the one door every judgement call goes through.

Escalation order (PRD section 5):
  1. rules and recipe defaults (plus rules learned from repeated user corrections)
  2. Jev (reserved slot, off in this build)
  3. the assistant (Claude Code headless, or Claude chat by copy-paste), only below the cut-off
  4. the user
Every decision is logged, without raw sensor data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tinyml.config import Settings, load_settings
from tinyml.decide.providers import (
    ClaudeChatProvider,
    ClaudeCodeProvider,
    JevProvider,
    ProviderUnavailable,
    build_prompt,
    parse_reply,
)
from tinyml.store.db import Store

LEARN_AFTER = 3  # identical user corrections before a correction becomes a fixed rule


@dataclass
class Question:
    type: str
    key: str
    text: str
    options: list[str]
    summary: dict[str, Any] = field(default_factory=dict)
    use_case: str = ""


@dataclass
class Answer:
    choice: str
    confidence: float
    source: str
    needs_user: bool = False
    follow_up: str | None = None
    prompt: str | None = None
    decision_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# Interactive hook used by the CLI: receives (question, best guess, prompt-for-chat) and returns a choice.
AskUser = Callable[[Question, "Answer", str | None], str | None]


class Decider:
    def __init__(
        self,
        store: Store,
        project_id: str | None,
        settings: Settings | None = None,
        ask_user: AskUser | None = None,
    ):
        self.store = store
        self.project_id = project_id
        self.settings = settings or load_settings()
        self.ask_user = ask_user
        self.jev = JevProvider()

    def cutoff(self, decision_type: str) -> float:
        return self.settings.assistant.cutoffs.get(decision_type, 0.7)

    def decide(self, q: Question, guess: tuple[str, float] | None) -> Answer:
        cutoff = self.cutoff(q.type)
        learned = self.store.learned_rule(q.type, q.key)
        if learned and learned["choice"] in q.options:
            return self._log(q, Answer(learned["choice"], 0.95, "learned_rule"), final=True)

        best = Answer(guess[0], guess[1], "rules") if guess else Answer(q.options[0], 0.0, "default")
        if best.confidence >= cutoff:
            return self._log(q, best, final=True)

        if self.jev.available():
            try:
                reply = self.jev.ask("", q.options)
                if reply.choice and reply.confidence >= cutoff:
                    return self._log(q, Answer(reply.choice, reply.confidence, "jev"), version=reply.version, final=True)
            except ProviderUnavailable:
                pass

        provider = self.settings.assistant.provider
        prompt: str | None = None
        version: str | None = None
        if provider in ("claude_code", "claude_chat"):
            examples = self._similar_examples(q)
            prompt = build_prompt(q.type, q.text, q.options, q.summary, q.use_case, examples)
        if provider == "claude_code":
            claude = ClaudeCodeProvider(self.settings.assistant)
            try:
                reply = claude.ask(prompt or "", q.options)
                version = reply.version
                if reply.choice:
                    candidate = Answer(reply.choice, reply.confidence, "claude_code", follow_up=reply.follow_up)
                    if reply.confidence >= cutoff:
                        return self._log(q, candidate, version=version, final=True)
                    if reply.confidence > best.confidence:
                        best = candidate
                elif reply.follow_up:
                    best.follow_up = reply.follow_up
            except ProviderUnavailable as exc:
                version = f"unavailable: {exc}"
        elif provider == "claude_chat":
            version = ClaudeChatProvider().version()

        if self.ask_user:
            chat_prompt = prompt if provider == "claude_chat" else None
            reply_text = self.ask_user(q, best, chat_prompt)
            if reply_text:
                if provider == "claude_chat" and reply_text.strip().startswith("{"):
                    choice, confidence, follow_up = parse_reply(reply_text, q.options)
                    if choice:
                        ans = Answer(choice, confidence, "claude_chat", follow_up=follow_up)
                        return self._log(q, ans, version=version, final=True)
                if reply_text in q.options:
                    ans = Answer(reply_text, 1.0, "user")
                    logged = self._log(q, best, version=version, final=False)
                    self.record_user_choice(logged.decision_id, reply_text)  # type: ignore[arg-type]
                    ans.decision_id = logged.decision_id
                    return ans

        best.needs_user = True
        best.prompt = prompt if provider == "claude_chat" else None
        return self._log(q, best, version=version, final=False)

    def record_user_choice(self, decision_id: int, choice: str) -> None:
        """The user confirmed or overrode a decision. Repeated identical corrections become rules."""
        row = self.store.decision(decision_id)
        if row is None:
            return
        self.store.finalize_decision(decision_id, choice, "user")
        if row["answer"] == choice:
            return
        same = [
            d for d in self.store.corrected_examples(row["decision_type"], limit=200)
            if d["decision_key"] == row["decision_key"] and d["final_choice"] == choice
        ]
        if len(same) >= LEARN_AFTER:
            self.store.upsert_learned_rule(row["decision_type"], row["decision_key"], choice, len(same))

    def _similar_examples(self, q: Question, limit: int = 3) -> list[dict[str, Any]]:
        words = set(q.key.lower().replace("_", " ").split())
        scored = []
        for ex in self.store.corrected_examples(q.type):
            other = set(str(ex["decision_key"]).lower().replace("_", " ").split())
            overlap = len(words & other) / max(1, len(words | other))
            scored.append((overlap, ex))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [ex for _, ex in scored[:limit]]

    def _log(self, q: Question, ans: Answer, version: str | None = None, final: bool = False) -> Answer:
        decision_id = self.store.log_decision({
            "project_id": self.project_id,
            "decision_type": q.type,
            "decision_key": q.key,
            "question": q.text,
            "summary": q.summary,
            "options": q.options,
            "answer": ans.choice,
            "confidence": ans.confidence,
            "source": ans.source,
            "provider_version": version,
            "follow_up": ans.follow_up,
            "final_choice": ans.choice if final else None,
            "final_source": ans.source if final else None,
        })
        ans.decision_id = decision_id
        return ans
