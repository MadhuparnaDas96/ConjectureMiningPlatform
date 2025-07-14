from dotenv import load_dotenv
from pydantic import BaseModel 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool
from tools import definition_tool
from tools import find_mathlib_knn_tool
from tools import danalyze_csv_tool
from tools import rank_tool
from tools import fetch_definition_from_mathlib
from tools import (
    PatternDiscovery,
    MLIntegration,
    init_lean_server,
    check_command,
    load_file
)
import subprocess

import argparse
import logging
import networkx as nx 
import random

###from langchain_gemini import ChatGemini

load_dotenv()

class ConjecturegenerationResponse(BaseModel):
    topic: str  
    summary: str              # Topic of the conjecture (e.g., "group theory", "number theory")
    conjecture: str
    source: list[str]    # List of sources or references that inspired the conjecture
    explanation: str  
    tools_used : list[str]          # Source of the conjecture (e.g., "mathlib", "user input")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)

###llm2 = ChatGemini(model="gemini-1.5-pro", api_key="your_gemini_api_key")

parser = PydanticOutputParser(pydantic_object=ConjecturegenerationResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an advanced mathematical reasoning assistant. Your task is to generate insightful and original advanced mathematical conjectures and hypothesis based on a given topic.
            You must understand the core ideas in the topic.
            Avoid restating known theorems.
            Output a plausible, original conjecture in the style of mathematical research. Please include mathematical equations.
            Clearly explain the reasoning behind the conjecture.
            Use `find_related_mathlib` to fetch Mathlib definitions via Lean and identify k-nearest theorems.
            If CSV data is detected, call `analyze_csv_dataset`.
            Use `discover_motifs` to mine symbolic patterns and `rank_motifs` to evaluate them.
            Then synthesise an original and plausible mathematical conjecture or hypothesis based on all results.
            Warp this output in this formal and provide no other text\n{format_instructions}
            """,
        ),
("placeholder", "{chat_history}"),
("human", "{query}"),
("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


tools = [search_tool, definition_tool, danalyze_csv_tool, rank_tool, find_mathlib_knn_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("what can I help you with?")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print("Structured Response:", structured_response)
except Exception as e:
    print("Error parsing response:", e, "Raw Response:", raw_response)


def main():
    parser = argparse.ArgumentParser(
        description="Conjecture mining pipeline with Lean integration"
    )
    parser.add_argument(
        "--patterns", type=str, nargs="+",
        help="LaTeX expressions to mine patterns from"
    )
    parser.add_argument(
        "--query", type=str,
        help="Query conjecture for similarity ranking"
    )
    parser.add_argument(
        "--lean-version", type=str, default=None,
        help="Lean version to use (e.g. v4.18.0)"
    )
    parser.add_argument(
        "--verbose-lean", action="store_true",
        help="Show Lean build logs on startup"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize Lean server
    lean_server = init_lean_server(
        verbose=args.verbose_lean,
        lean_version=args.lean_version
    )
    logger.info("Lean server initialized.")

    # Sanity-check Lean
    msgs = check_command(lean_server, "#check Nat.add_comm")
    logger.info(f"Lean check: {msgs}")

    # Pattern mining
    if args.patterns:
        pd = PatternDiscovery(args.patterns)
        motifs = pd.mine_motifs(min_support=2)
        logger.info(f"Mined motifs: {motifs}")
        templates = pd.discover_templates()
        logger.info(f"Discovered templates: {templates}")

    # ML-based ranking
    if args.query and args.patterns:
        ml = MLIntegration()
        ranked = ml.rank_by_similarity(args.query, args.patterns)
        logger.info("Similarity ranking results:")
        for cand, score in ranked:
            logger.info(f"  {cand} -> {score:.4f}")

if __name__ == "__main__":
    main()


