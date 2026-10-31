# TradingAgents for Claude Code

Adds `/tradingagents` to Claude Code. It runs the
[TradingAgents](https://github.com/TauricResearch/TradingAgents) multi-agent
stock analysis on a ticker, with every model call going through `claude -p`
on your Claude plan. No API key needed.

> Research output, not financial advice. The agents can be wrong; check
> anything before you act on it.

## What runs

1. **Analysts**: market (technicals), sentiment, news and fundamentals, plus a
   value analyst that reads SEC filings.
2. **Research team**: a bull and a bear researcher debate, a fact checker
   checks their claims, and a research manager weighs them.
3. **Trader**: turns the plan into a trade.
4. **Risk team**: aggressive, conservative and neutral analysts debate it.
5. **Portfolio manager**: gives the final rating (Buy, Overweight, Hold,
   Underweight or Sell).

## Needs

- Claude Code, signed in (run `claude` once)
- [uv](https://docs.astral.sh/uv/getting-started/installation/). It fetches
  Python 3.13 and the engine's dependencies (~250 MB) on the first run.
- macOS or Linux. Windows is untested.

## Install

```
/plugin marketplace add Roisters/trading-agents-claude
/plugin install tradingagents@tradingagents
```

## Use

```
/tradingagents NVDA
/tradingagents AAPL 2026-09-30
/tradingagents MSFT market,news
/tradingagents TSLA 2 rounds
```

- **Ticker**: any Yahoo Finance symbol (`NVDA`, `0700.HK`, `BTC-USD`).
- **Date**: `YYYY-MM-DD`, default today. Not in the future.
- **Analysts**: any of `market`, `social` (sentiment), `news`,
  `fundamentals`; default all four. The value analyst always runs.

A run takes about 10 to 20 minutes and makes a few dozen model calls, which
count toward your Claude plan's usage limits: a few runs back to back can use
up a session's allowance. Fewer analysts or a lighter model
(`TRADINGAGENTS_DEEP_THINK_LLM=claude-sonnet-5`) use less. Claude then
summarises the final rating and each team's view.

Reports are saved to `~/.tradingagents/logs/reports/<TICKER>_<timestamp>/`
(`complete_report.md` and `complete_report.html`, plus one file per agent).

## Settings (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `TRADINGAGENTS_QUICK_THINK_LLM` | `claude-sonnet-5` | Model for the analysts and debaters |
| `TRADINGAGENTS_DEEP_THINK_LLM` | `claude-opus-5-5` | Model for the managers |
| `TRADINGAGENTS_MAX_DEBATE_ROUNDS` | `1` | Bull/bear debate rounds |
| `TRADINGAGENTS_MAX_RISK_ROUNDS` | `1` | Risk team debate rounds |
| `TRADINGAGENTS_OUTPUT_LANGUAGE` | `English` | Language of the reports |
| `FRED_API_KEY` | unset | Optional. Adds US macro data ([free key](https://fred.stlouisfed.org/docs/api/api_key.html)) |
| `SEC_EDGAR_USER_AGENT` | built in | Optional. Your name and email, which SEC asks callers to send |
| `TRADINGAGENTS_SOURCE` | bundled `engine/` | Run a different copy of the engine |

Set them in your shell profile, or in the `env` block of Claude Code's
`settings.json`. If `ANTHROPIC_API_KEY` is set, the plugin ignores it, so runs
stay on your plan.

## Troubleshooting

- **`uvx not found`**: install uv, then restart Claude Code.
- **Auth errors**: run `claude` once and sign in.
- **`You've hit your session limit`**: the run stops at once; run it again
  after the reset time in the message.
- **A run stopped part way**: run the same command again; it starts over. To
  resume from the last finished step instead, set
  `TRADINGAGENTS_CHECKPOINT_ENABLED=true` before the run that might stop.

## Where the engine comes from

`engine/` is a modified copy of TradingAgents v0.7.0 by Tauric Research. It
adds the `claude_code` provider, the fact checker, the value analyst and
headless CLI flags. `bin/tradingagents-claude` runs it.

## Support

If this plugin is useful to you, you can support its development on
[Ko-fi](https://ko-fi.com/roisters). Donations are optional and don't buy
features, support priority or investment advice.

## License

Apache-2.0 (see `LICENSE`). `engine/` is also Apache-2.0 (see
`engine/LICENSE`), with the changes listed above.
