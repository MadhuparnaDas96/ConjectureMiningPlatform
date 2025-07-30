from dotenv import load_dotenv
from pydantic import BaseModel 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import find_mathlib_knn_tool
from tools import fetch_definition_from_mathlib
from tools import find_related_mathlib
from typing import List, Dict, Any
from langchain.tools import Tool
import json


from tools import (
    search_tool,
    definition_tool,
    danalyze_csv_tool,
    rank_tool,
    arxiv_search_tool,
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


# Define batch pipeline function (formerly main())
def batch_pipeline(input_json: str) -> Dict[str, Any]:
    """
    Run pattern mining and similarity ranking as a tool.
    Expects a JSON string with keys:
      - patterns: List[str]
      - query: str (optional)
      - lean_version: str (optional)
      - verbose_lean: bool (optional)
    Returns a dict of results.
    """
    data = json.loads(input_json)
    patterns = data.get("patterns", [])
    query = data.get("query")
    lean_version = data.get("lean_version")
    verbose = data.get("verbose_lean", False)

    # Initialize Lean server and sanity check
    lean_server = init_lean_server(verbose=verbose, lean_version=lean_version)
    _ = check_command(lean_server, "#check Nat.add_comm")

    results: Dict[str, Any] = {}
    # Pattern mining
    if patterns:
        pd = PatternDiscovery(patterns)
        results["motifs"] = pd.mine_motifs(min_support=2)
        results["templates"] = pd.discover_templates()
    # ML-based ranking
    if patterns and query:
        ml = MLIntegration()
        results["ranking"] = ml.rank_by_similarity(query, patterns)
    return results

# Register batch pipeline as a tool
tools = [
    search_tool,
    definition_tool,
    danalyze_csv_tool,
    rank_tool,
    arxiv_search_tool,
    find_mathlib_knn_tool,
    Tool.from_function(
        batch_pipeline,
        name="batch_pipeline",
        description=(
            "Run pattern-mining and similarity-ranking pipeline. "
            "Input JSON with 'patterns':List[str], 'query':str, 'lean_version':str, 'verbose_lean':bool."
        )
    )
]

load_dotenv()

class ConjecturegenerationResponse(BaseModel):
    topic: str  
    summary: str              # Topic of the conjecture (e.g., "group theory", "number theory")
    conjecture: str
    source: list[str]    # List of sources or references that inspired the conjecture
    explanation: str  
    tools_used : list[str]          # Source of the conjecture (e.g., "mathlib", "user input")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.8)

###llm2 = ChatGemini(model="gemini-1.5-pro", api_key="your_gemini_api_key")

parser = PydanticOutputParser(pydantic_object=ConjecturegenerationResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert mathematical researcher assistant focused on generating novel, non-trivial, and insightful conjectures based on a given topic.
            You must understand the core ideas in the topic.
            Use critical mathematical thinking like a mathematician.
            Avoid restating known theorems but do not ignore them.
            Output a plausible, original novel and non-trivial conjecture in the style of mathematical research. Please include mathematical equations.
            Clearly explain the reasoning behind the conjecture.
            Retrieve and list recent relevant definitions, theorems, and conjectures from Mathlib using find_related_mathlib.
            Search for recent academic papers related to the topic (e.g., via arxiv_search) and summarise key results, highlighting any partial progress or open problems.
            If CSV data is detected, call `analyze_csv_dataset`.
            Use discover_motifs to find frequent symbolic patterns in the input expressions and rank_motifs to prioritise the most relevant patterns.
            Critically examine all gathered information to identify gaps, limitations, or unexplored directions in the current knowledge.
            Based on this analysis, propose a new conjecture that builds upon and refines existing knowledge, introduces novel structure or regularity, is plausible, non-trivial, and mathematically interesting for a mathematician, includes relevant mathematical expressions or equations.
            Clearly explain the rationale behind the conjecture in a concise, logical manner, detailing the thought process and how the conjecture relates to known results.
            Warp this output in this formal and provide no other text\n{format_instructions}
            """,
        ),
("placeholder", "{chat_history}"),
("human", "{query}"),
("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


tools = [
    search_tool,
    definition_tool,
    danalyze_csv_tool,
    rank_tool,
    arxiv_search_tool,
    find_mathlib_knn_tool,
    Tool.from_function(
        batch_pipeline,
        name="batch_pipeline",
        description=(
            "Run pattern-mining and similarity-ranking pipeline. "
            "Input JSON with 'patterns':List[str], 'query':str, 'lean_version':str, 'verbose_lean':bool."
        )
    )
]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
if __name__ == "__main__":
    query = input("what can I help you with? ")
    raw_response = agent_executor.invoke({"query": query})
    print(raw_response)
    # Optionally parse structured output
    try:
        structured = parser.parse(raw_response.get("output")[0]["text"])
        print("Structured Response:", structured)
    except Exception:
        pass




#query = input("what can I help you with?")
#raw_response = agent_executor.invoke({"query": query})
#print(raw_response)

#try:
 #   structured_response = parser.parse(raw_response.get("output")[0]["text"])
  #  print("Structured Response:", structured_response)
#except Exception as e:
 #   print("Error parsing response:", e, "Raw Response:", raw_response)




#if __name__ == "__main__":
 #   print(find_related_mathlib("Nat.add_comm"))

