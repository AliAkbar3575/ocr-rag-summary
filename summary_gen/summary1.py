from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class SummaryState(TypedDict):
    input_text: str
    case_fact_summary: str
    approved: bool


class HITLSummaryEngine:

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        
        self.llm = ChatGroq(model=model_name, temperature=0.2)
        self.editing_llm = ChatGroq(model=model_name, temperature=0.1)

        self.checkpointer = MemorySaver()
        self.app = self._build_graph()


    def _build_graph(self):

        builder = StateGraph(SummaryState)

        builder.add_node("generate_summary", self._generate_summary_node)
        builder.add_node("human_review", self._human_review_node)

        builder.add_edge(START, "generate_summary")
        builder.add_edge("generate_summary", "human_review")

        return builder.compile(checkpointer=self.checkpointer)


    def _generate_summary_node(self, state: SummaryState):

        prompt = f"""
Generate a grounded CASE FACT SUMMARY.

Rules:
- Only use given text, and use all the pages.
- No hallucination
- Keep structured bullet points
- You must make summary for every different topics (eg. if the document has 3 distinct topics in 5 pages, provide 3 summaries. If the whole document is about one topic, make only 1 summary - These are only the examples. given text has multiple topics in multiple pages).
  

TEXT:
{state["input_text"]}
"""

        response = self.llm.invoke(prompt)

        return {
            "case_fact_summary": response.content,
            "approved": False
        }


    def _human_review_node(self, state: SummaryState):

        decision = interrupt({
            "case_fact_summary": state["case_fact_summary"],
            "instruction": """
        You can:
        1. approve → type "yes"
        2. give feedback → describe what to change

        Example:
        - "change year to 50"
        - "make it shorter"
        """
        })

        if isinstance(decision, str) and decision.lower() == "yes":
            return Command(
                goto=END,
                update={"approved": True}
            )

        feedback = f"USER WANTS CHANGE: {decision}"

        edit_prompt = f"""
        You are a HIGH-PRECISION TEXT PATCH SYSTEM.

        You MUST behave like a Git diff editor.

        TASK:
        Apply ONLY the user's requested changes.

        RULES:
        - Do NOT rewrite the document
        - Do NOT paraphrase unchanged text
        - Do NOT restructure sections
        - Keep ALL unchanged sentences EXACTLY identical
        - Only modify the minimum required words

        Think in terms of: PATCH, NOT REWRITE

        ORIGINAL:
        {state["case_fact_summary"]}

        USER FEEDBACK:
        {feedback}

        Return full updated document with minimal diff applied.
        """

        response = self.editing_llm.invoke(edit_prompt)

        return Command(
            goto="human_review",
            update={"case_fact_summary": response.content}
        )