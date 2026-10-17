"""Step 1 and 2: turn the user's paragraph into a ProblemSpec.

Rules read the text first. Anything they are unsure about goes through decide(),
which may ask the assistant for one plain follow-up question, and finally the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tinyml.contracts import ProblemSpec
from tinyml.decide import Answer, Decider, Question
from tinyml.recipes import Recipe, load_recipes

CONNECTIVITY_WORDS = {
    "wifi": [r"wi-?fi", r"wlan", r"router", r"mqtt over wi"],
    "ble": [r"bluetooth", r"\bble\b", r"phone app", r"smartphone"],
    "cellular": [r"cellular", r"\blte\b", r"\b4g\b", r"\b5g\b", r"nb-?iot", r"lte-?m", r"\bsim\b", r"gsm"],
    "none": [r"no (?:radio|connectivity|network|wireless)", r"offline", r"standalone", r"local (?:alarm|buzzer|led)",
             r"buzzer", r"\bled\b only", r"stores? (?:it )?(?:to|on) (?:an? )?sd"],
}
POWER_WORDS = {
    "battery": [r"batter", r"coin cell", r"\baa\b", r"solar", r"energy harvest", r"wireless sensor"],
    "plugged": [r"plugged", r"mains", r"wall power", r"usb[- ]powered", r"powered from", r"24 ?v", r"always powered"],
}
PRIORITY_WORDS = {
    "battery": [r"battery life", r"years? on", r"low power", r"months on", r"long battery"],
    "cost": [r"cheap", r"low[- ]cost", r"inexpensive", r"as little as possible", r"cost matters"],
    "speed": [r"real[- ]time", r"instant", r"fast", r"quick(?:ly)?", r"latency", r"immediately"],
}
VENDOR_WORDS = {
    "espressif": [r"esp32", r"espressif", r"esp-?idf"],
    "st": [r"stm32", r"\bst micro", r"stmicro"],
    "nordic": [r"\bnrf", r"nordic"],
    "raspberrypi": [r"\bpico\b", r"rp2040", r"rp2350", r"raspberry"],
}
ANOMALY_WORDS = [r"abnormal", r"anomal", r"unusual", r"fail", r"fault", r"broken", r"wrong", r"wear", r"strange",
                 r"something (?:is )?off", r"out of the ordinary", r"looks? odd"]
CLASSIFY_WORDS = [r"classif", r"which (?:activity|event|kind|type)", r"recogni[sz]e", r"tell (?:apart|which|what)",
                  r"categori", r"detect (?:when|if) .* (?:or|vs)", r"distinguish", r"whether .* or"]


def _hits(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def _best(text: str, table: dict[str, list[str]]) -> tuple[str | None, float]:
    scores = {key: _hits(text, pats) for key, pats in table.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return None, 0.0
    top, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, 0)
    if top[1] == second[1]:
        return top[0], 0.4
    return top[0], 0.9 if top[1] >= 1 and second[1] == 0 else 0.7


def guess_recipe(text: str, recipes: dict[str, Recipe]) -> tuple[str | None, float]:
    scores: dict[str, int] = {}
    for rid, recipe in recipes.items():
        scores[rid] = sum(1 for word in recipe.keywords if re.search(rf"\b{re.escape(word)}", text))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] == 0:
        return None, 0.0
    top_score = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if top_score == second:
        return ranked[0][0], 0.4
    confidence = 0.95 if top_score >= 2 and second == 0 else 0.8 if second == 0 else 0.6
    return ranked[0][0], confidence


def guess_budget(text: str) -> float | None:
    match = re.search(r"(?:\$|usd\s*|us\$)\s*(\d+(?:\.\d+)?)", text) or re.search(
        r"(\d+(?:\.\d+)?)\s*(?:dollars|usd|\$)", text
    )
    return float(match.group(1)) if match else None


def guess_latency_ms(text: str) -> float | None:
    match = re.search(r"within\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec|seconds?)\b", text)
    if not match:
        return None
    value = float(match.group(1))
    return value if match.group(2).startswith("m") else value * 1000.0


@dataclass
class IntakeResult:
    spec: ProblemSpec
    answers: dict[str, Answer]
    questions: list[dict] = field(default_factory=list)


def parse_use_case(text: str, decider: Decider, vendor: str | None = None) -> IntakeResult:
    lowered = text.lower()
    recipes = load_recipes()
    answers: dict[str, Answer] = {}

    rid, conf = guess_recipe(lowered, recipes)
    answers["recipe"] = decider.decide(
        Question("recipe", "use_case", "Which of these is closest to what you are building?", list(recipes),
                 {"keyword_hits": rid, "text_length": len(text)}, text),
        (rid, conf) if rid else None,
    )
    recipe = recipes[answers["recipe"].choice]

    anomaly_hits, classify_hits = _hits(lowered, ANOMALY_WORDS), _hits(lowered, CLASSIFY_WORDS)
    if anomaly_hits > classify_hits:
        task_guess = ("anomaly", 0.9)
    elif classify_hits > anomaly_hits:
        task_guess = ("classification", 0.9)
    else:
        task_guess = (recipe.default_task, 0.75)
    answers["task"] = decider.decide(
        Question("task", recipe.id, "Should the device flag anything unusual, or name which event happened?",
                 ["anomaly", "classification"], {"recipe": recipe.id, "anomaly_words": anomaly_hits,
                                                 "classify_words": classify_hits}, text),
        task_guess,
    )

    conn, conn_conf = _best(lowered, CONNECTIVITY_WORDS)
    answers["connectivity"] = decider.decide(
        Question("connectivity", recipe.id, "How will the device send its alerts: Wi-Fi, Bluetooth, cellular, or not at all?",
                 ["wifi", "ble", "cellular", "none", "any"], {"mentioned": conn}, text),
        (conn, conn_conf) if conn else ("any", 0.5),
    )

    power, power_conf = _best(lowered, POWER_WORDS)
    answers["power"] = decider.decide(
        Question("power", recipe.id, "Will it run on a battery or be plugged in?", ["battery", "plugged"],
                 {"mentioned": power}, text),
        (power, power_conf) if power else ("battery", 0.5),
    )

    prio, prio_conf = _best(lowered, PRIORITY_WORDS)
    if not prio:
        prio, prio_conf = ("battery", 0.65) if answers["power"].choice == "battery" else ("cost", 0.6)
    answers["priority"] = decider.decide(
        Question("priority", recipe.id, "What matters most: battery life, low cost, or speed?",
                 ["battery", "cost", "speed"], {"mentioned": prio, "power": answers["power"].choice}, text),
        (prio, prio_conf),
    )

    detected_vendor, _ = _best(lowered, VENDOR_WORDS)
    spec = ProblemSpec(
        recipe=recipe.id,
        task=answers["task"].choice,  # type: ignore[arg-type]
        vendor=vendor or detected_vendor or "any",
        connectivity=answers["connectivity"].choice,  # type: ignore[arg-type]
        power=answers["power"].choice,  # type: ignore[arg-type]
        budget_usd_per_unit=guess_budget(lowered),
        priority=answers["priority"].choice,  # type: ignore[arg-type]
        max_latency_ms=guess_latency_ms(lowered) or recipe.default_max_latency_ms,
        description=text.strip(),
        seed=decider.settings.seed,
    )
    questions = [
        {
            "field": name,
            "decision_id": ans.decision_id,
            "question": ans.follow_up or _question_text(name),
            "guess": ans.choice,
            "confidence": round(ans.confidence, 2),
            "options": _options(name, recipes),
            "prompt": ans.prompt,
        }
        for name, ans in answers.items()
        if ans.needs_user
    ]
    return IntakeResult(spec=spec, answers=answers, questions=questions[:1] if questions else [])


def _question_text(name: str) -> str:
    return {
        "recipe": "Which of these is closest to what you are building?",
        "task": "Should the device flag anything unusual, or name which event happened?",
        "connectivity": "How will the device send its alerts: Wi-Fi, Bluetooth, cellular, or not at all?",
        "power": "Will it run on a battery or be plugged in?",
        "priority": "What matters most: battery life, low cost, or speed?",
    }[name]


def _options(name: str, recipes: dict[str, Recipe]) -> list[str]:
    return {
        "recipe": list(recipes),
        "task": ["anomaly", "classification"],
        "connectivity": ["wifi", "ble", "cellular", "none", "any"],
        "power": ["battery", "plugged"],
        "priority": ["battery", "cost", "speed"],
    }[name]
