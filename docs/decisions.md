# Judgement calls: rules, assistant, user

Plain code produces every number. Judgement calls (which recipe fits, what a column is, how serious a data problem is, which label means normal) all go through one function, `Decider.decide(question, rule_guess)` in `src/tinyml/decide/core.py`.

The order is the PRD's. First, rules learned from repeated user corrections and the built-in rules answer. If their confidence is at or above the cut-off for that decision type (`settings.json`, `assistant.cutoffs`), that is the answer. Next is the Jev slot, which is empty in this build. Then comes the assistant, if one is enabled. Last, the user answers: in the web app as a highlighted question or dropdown, on the command line as a prompt.

## The assistant, for now: Claude

Until Jev is wired in, the assistant is Claude, used through tools the team already has. You choose it in the app under **Assistant**, or with `tinyml settings --assistant ...`.

- **Ask me** (`off`): no AI at all. Every unsure point becomes a question for you. The product is fully usable this way, as the PRD requires.
- **Claude Code** (`claude_code`): tinyml runs `claude -p "<prompt>" --output-format json` on your machine with your own login. The prompt contains only the column summary and the options, never raw rows. Claude answers with JSON (choice, confidence, an optional follow-up question). If its confidence is below the cut-off, you are asked. The Claude Code version is logged with every decision.
- **Claude chat** (`claude_chat`): tinyml shows the same prompt for you to paste into Claude chat. You paste the JSON answer back, or simply pick an option.

The same setting powers "Ask about this package" on the Build screen. With Claude Code, it unpacks the package and lets Claude read it (read-only tools); with Claude chat, you get a ready prompt that includes the report and header.

## The decision log

Every call is logged in the `decisions` table with the decision type, the summary that was sent, the options, the answer and confidence, the source (rules, learned rule, Claude Code, Claude chat or user), the assistant version, any follow-up question, your final choice, and whether the build that followed passed. The log is visible in the app under **Decisions** and with `tinyml decisions`.

The log is put to work in two ways today. After three identical corrections for the same decision type and key (for example, column `trip` corrected to `recording_id`), the correction becomes a fixed rule. And the most similar past corrections are added to the assistant prompt as examples. Tuning cut-offs per type, and training a small classifier to sit behind `decide()`, can be built on the same table.

## Adding Jev later

Implement `JevProvider.available()` and `ask()` in `decide/providers.py`. The escalation order already calls it between the rules and the assistant.
