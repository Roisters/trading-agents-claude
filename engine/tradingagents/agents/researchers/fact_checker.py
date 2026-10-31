"""Fact Checker: checks the bull and bear researchers' claims before the Research Manager weighs them.

The researchers argue from the analyst reports without tools, so a figure one of
them gets wrong or makes up can win the debate. This agent re-reads the reports,
re-pulls the data where a claim needs it, and hands the Research Manager a
verdict on each claim.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts import fundamentals_analyst, market_analyst
from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction

# The tools this agent is offered; its tool node is built from the same tuple.
TOOLS = market_analyst.TOOLS + fundamentals_analyst.TOOLS

_REPORTS = (
    ("Market Analyst", "market_report"),
    ("Sentiment Analyst", "sentiment_report"),
    ("News Analyst", "news_report"),
    ("Fundamentals Analyst", "fundamentals_report"),
    ("Value Analyst", "valuation_report"),
)


def create_fact_checker(llm):
    def fact_checker_node(state):
        reports = "\n\n".join(f"### {name}\n{state[key]}" for name, key in _REPORTS if state.get(key))

        system_message = (
            "You are the Fact Checker for an investment debate. The bull and bear researchers below argued"
            " from the analyst reports. List every specific factual claim either side relied on: figures,"
            " growth rates, margins, prices, indicator readings, dates, events. Check each against the analyst"
            " reports; when a report does not settle it, call a tool to pull the data. Do not judge who argued"
            " better, and do not add arguments of your own."
            "\n\nWrite one line per claim: the side (Bull/Bear), the claim, the verdict (Verified / Wrong /"
            " Unsupported) and, for Wrong, the correct figure and its source. End with a one-line summary of"
            " which side's case rests on claims that did not check out, if either."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all tool-call date ranges. {instrument_context}\n"
                    "{system_message}\n\n## Analyst reports\n{reports}\n\n## Debate\n{debate}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        # Values go in as partials, so braces in the reports or debate are text, not template fields.
        prompt = prompt.partial(
            system_message=system_message,
            tool_names=", ".join(tool.name for tool in TOOLS),
            current_date=state["trade_date"],
            instrument_context=get_instrument_context_from_state(state),
            reports=reports or "(none)",
            debate=state["investment_debate_state"].get("history", ""),
        )

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "fact_check_report": report,
        }

    return fact_checker_node
