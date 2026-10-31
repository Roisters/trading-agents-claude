---
description: Run the TradingAgents multi-agent analysis on a ticker, on your Claude plan
argument-hint: <TICKER> [YYYY-MM-DD] [analysts e.g. market,social,news,fundamentals]
allowed-tools: Bash(tradingagents-claude:*), Read(~/.tradingagents/**)
---

Run the TradingAgents pipeline for: $ARGUMENTS

1. Parse the arguments: ticker (required; if missing, ask for one and stop), date (YYYY-MM-DD, default: today; not in the future), analysts (comma-separated from `market`, `social`, `news`, `fundamentals`; default: all four; "sentiment" means `social`). "2 rounds" etc. sets the debate and risk rounds; otherwise 1.
2. Run this as one Bash command on its own, with no `cd`, no redirection and nothing chained before or after it, in the background with a 30-minute timeout. It takes about 10 to 20 minutes; the first run also installs the engine's dependencies:

```
tradingagents-claude --ticker <TICKER> --date <DATE> --analysts <ANALYSTS> --save --no-show
```

   For N rounds, put `TRADINGAGENTS_MAX_DEBATE_ROUNDS=N TRADINGAGENTS_MAX_RISK_ROUNDS=N` in front of it (this needs the user's approval once).

   If it says `uvx not found`, tell the user to install uv (https://docs.astral.sh/uv/) and stop. If a model call fails with an auth error, tell the user to run `claude` once and sign in.

3. Do not do any analysis yourself: the agents in the pipeline do it. When the run finishes, read the end of its output: after `===== REPORT FOLDER:` it prints the research manager's plan, the trader's proposal and the portfolio manager's decision. Give the user the final rating, a short summary of the research, trading and risk views from those, and the path to `complete_report.md` (or `complete_report.html`) in that folder for the full report. Say that it is research output, not financial advice. If the run fails, show the error.
