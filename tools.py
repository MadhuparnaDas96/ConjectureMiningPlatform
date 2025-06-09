from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun 
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import subprocess
import mathlib 
import os
from MathLibrary import MathLibrary





search = DuckDuckGoSearchRun()
search_tool =Tool(
    name = "search",
    func = search.run,
    description = "search the web for relevant information"
)

def fetch_definition_from_mathlib(term: str) -> str:
    """
    Fetches the Lean definition of `term` from Mathlib via `#print`.
    This simple callback writes a Lean script, builds the Lake project,
    and returns the raw Lean output.
    """
    lean_code = f"""
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic

open Nat
open Rat

#print {term}
"""
    script_path = "query_definition.lean"
    with open(script_path, "w") as f:
        f.write(lean_code)


    result = subprocess.run(
        ["lake", "exec", "lean", "--run", script_path],
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.stdout.strip() if result.stdout else result.stderr.strip()

definition_tool = Tool.from_function(
    fetch_definition_from_mathlib,
    name="fetch_definition_from_mathlib",
    description="Fetches definitions from Mathlib via Lean/Lake."
)
