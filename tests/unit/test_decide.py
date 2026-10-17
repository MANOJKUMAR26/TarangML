import json
import os
import stat

from tinyml.config import AssistantSettings, Settings
from tinyml.decide import Decider, Question

Q = Question("column_role", "c4", "What is column c4?", ["time", "sensor_axis", "label", "recording_id", "ignore"],
             {"name": "c4", "min": 0, "max": 1})


def test_confident_rule_decides_and_is_logged(store):
    ans = Decider(store, None, Settings()).decide(Q, ("sensor_axis", 0.95))
    assert ans.choice == "sensor_axis" and ans.source == "rules" and not ans.needs_user
    assert store.decision(ans.decision_id)["final_choice"] == "sensor_axis"


def test_unsure_rule_asks_the_user_when_ai_is_off(store):
    ans = Decider(store, None, Settings()).decide(Q, ("label", 0.4))
    assert ans.needs_user and ans.choice == "label"
    assert store.decision(ans.decision_id)["final_choice"] is None


def test_claude_chat_mode_returns_a_prompt(store):
    settings = Settings(assistant=AssistantSettings(provider="claude_chat"))
    ans = Decider(store, None, settings).decide(Q, ("label", 0.4))
    assert ans.needs_user and "Reply with only a JSON object" in ans.prompt


def test_claude_code_provider_is_used_below_the_cutoff(store, tmp_path, monkeypatch):
    fake = tmp_path / "claude"
    reply = json.dumps({"result": json.dumps({"choice": "ignore", "confidence": 0.9, "follow_up": None})})
    fake.write_text(f"#!/bin/sh\nif [ \"$1\" = --version ]; then echo 9.9; exit 0; fi\necho '{reply}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    settings = Settings(assistant=AssistantSettings(provider="claude_code"))
    ans = Decider(store, None, settings).decide(Q, ("label", 0.4))
    assert ans.choice == "ignore" and ans.source == "claude_code"
    assert "9.9" in store.decision(ans.decision_id)["provider_version"]


def test_repeated_corrections_become_a_rule(store):
    decider = Decider(store, None, Settings())
    for _ in range(3):
        ans = decider.decide(Q, ("label", 0.4))
        decider.record_user_choice(ans.decision_id, "recording_id")
    ans = decider.decide(Q, ("label", 0.4))
    assert ans.source == "learned_rule" and ans.choice == "recording_id"


def test_cli_style_user_callback(store):
    decider = Decider(store, None, Settings(), ask_user=lambda q, best, prompt: "ignore")
    ans = decider.decide(Q, ("label", 0.4))
    assert ans.choice == "ignore" and ans.source == "user"
