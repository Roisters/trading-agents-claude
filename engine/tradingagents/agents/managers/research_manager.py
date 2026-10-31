"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")
        fact_check = state.get("fact_check_report") or "(no fact check was made)"
        valuation = state.get("valuation_report") or "(no valuation was made)"

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

The debate always contains conflicting arguments; deciding which side is stronger is the job, so conflict alone is not a reason to Hold. Commit to the side with the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not manufacture a direction to appear decisive. Weigh the bull and bear cases on their merits, independent of which side spoke first or last.

---

**Debate History:**
{history}

**Buffett-style valuation (price-blind, then compared with the price):**
{valuation}

Weigh the valuation the way Buffett would: its computed intrinsic value and margin of safety are evidence from the filings; a price well above value argues against a Buy however strong the story, and a wide margin of safety supports one. Its letter grades are a model's low-trust guess: use them only to break a tie, never as the reason for a call. When it says the company is too hard to value, decide on the other evidence.

**Fact Check of the debate's claims:**
{fact_check}

An argument that rests on a claim the fact check found Wrong or Unsupported is weaker by that much; say so in the rationale when it decides the call.

## Output

Write these sections, in this order, starting with the recommendation on its own line:

- **Recommendation**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Rationale**: which arguments decided it
- **Strategic Actions**: concrete steps for the trader, sized against a standard allocation

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
