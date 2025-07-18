from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun 
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import subprocess
import json
import mathlib 
import os
from lean_interact import LeanREPLConfig, LeanServer, Command, FileCommand, ProofStep
from typing import List, Tuple, Dict
import sympy as sp
from collections import Counter
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sympy.parsing.latex import parse_latex
from difflib import get_close_matches
from functools import partial

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
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Ring.Basic

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
        timeout=3600
    )
    return result.stdout.strip() if result.stdout else result.stderr.strip()

definition_tool = Tool.from_function(
    fetch_definition_from_mathlib,
    name="fetch_definition_from_mathlib",
    description="Fetches definitions from Mathlib via Lean/Lake."
)

class PatternDiscovery:
    """
    Advanced pattern discovery using symbolic parsing and frequent motif mining.
    """
    def __init__(self, patterns: List[str] = None):
        self.raw_patterns = patterns or []
        self.parsed_expressions = [self._parse_expr(p) for p in self.raw_patterns]

    def _parse_expr(self, latex_str: str) -> sp.Expr:
        try:
            return parse_latex(latex_str)
        except Exception:
            return None

    def mine_motifs(self, min_support: int = 2) -> List[Tuple[str, int]]:
        motif_counter = Counter()
        for expr in self.parsed_expressions:
            if expr is None:
                continue
            for sub in sp.preorder_traversal(expr):
                motif_counter[str(sub)] += 1
        return [(motif, cnt) for motif, cnt in motif_counter.items() if cnt >= min_support]

    def discover_templates(self) -> List[str]:
        templates = set()
        for expr in self.parsed_expressions:
            if expr is None:
                continue
            s = str(expr)
            tpl = re.sub(r"\b\d+\b", "<NUM>", s)
            tpl = re.sub(r"\b[a-zA-Z]\b", "<VAR>", tpl)
            templates.add(tpl)
        return list(templates)
    


###pattern discovery tool

def discover_motifs(patterns: List[str]) -> List[Tuple[str,int]]:
    raw = patterns or []
    parsed = []
    for p in raw:
        try:
            parsed.append(parse_latex(p))
        except:
            continue
    counter = Counter()
    for expr in parsed:
        for sub in sp.preorder_traversal(expr):
            counter[str(sub)] += 1
    return [(m,c) for m,c in counter.items() if c>=2]

motif_tool = Tool.from_function(
    discover_motifs,
    name="discover_motifs",
    description="Extract frequent symbolic motifs from a list of LaTeX expressions."
)



class MLIntegration:
    """
    Embedding-based clustering and similarity scoring for conjecture candidates.
    """
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_tensor=True)

    def cluster_candidates(self, texts: List[str], threshold: float = 0.75) -> Dict[int, List[int]]:
        embeddings = self.embed_texts(texts)
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        clusters = {}
        visited = set()
        cid = 0
        for i in range(len(texts)):
            if i in visited:
                continue
            clusters[cid] = [i]
            visited.add(i)
            for j in range(i+1, len(texts)):
                if sim_matrix[i, j] >= threshold:
                    clusters[cid].append(j)
                    visited.add(j)
            cid += 1
        return clusters

    def rank_by_similarity(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        query_emb = self.model.encode(query, convert_to_tensor=True)
        cand_embs = self.embed_texts(candidates)
        scores = util.pytorch_cos_sim(query_emb, cand_embs)[0]
        sorted_idxs = np.argsort(-scores.cpu().numpy())
        return [(candidates[i], float(scores[i])) for i in sorted_idxs]
    
###ML integration tool
def rank_motifs(query: str, motifs: List[str]) -> List[Tuple[str,float]]:
    model = SentenceTransformer('all-mpnet-base-v2')
    q_emb = model.encode(query, convert_to_tensor=True)
    c_embs = model.encode(motifs, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(q_emb, c_embs)[0]
    idxs = np.argsort(-sims.cpu().numpy())
    return [(motifs[i], float(sims[i])) for i in idxs]

def rank_motifs_tool(input_str: str) -> list:
    try:
        input_json = json.loads(input_str)
    except Exception:
        return ["Error: input is not valid JSON"]

    query = input_json.get("query", "")
    motifs = input_json.get("motifs", [])
    if not query or not motifs:
        return []

    return rank_motifs(query, motifs)

rank_tool = Tool.from_function(
    rank_motifs_tool,
    name="rank_motifs",
    description="Rank symbolic motifs by similarity to a query. Input a JSON dict with 'query' string and 'motifs' list of strings."
)


    

#Lean helpers

def init_lean_server(verbose: bool = False, lean_version: str = None) -> LeanServer:
    """
    Configure and start a Lean 4 server via lean-interact.
    """
    config = LeanREPLConfig(verbose=verbose, lean_version=lean_version)
    server = LeanServer(config)
    return server

lean_server = init_lean_server()

def check_command(server: LeanServer, cmd: str) -> List[str]:
    """
    Send a single command to Lean and return its messages.
    """
    resp = server.run(Command(cmd=cmd))
    return resp.messages

def load_file(server: LeanServer, path: str) -> List[str]:
    """
    Load a .lean file into Lean and return its messages.
    """
    resp = server.run(FileCommand(path=path))
    return resp.messages    



def analyze_csv_dataset(csv_text: str) -> str:
    """
    Parses CSV input, performs symbolic pattern discovery on summary statistics,
    and returns a set of conjecture templates or patterns.
    """
    # Write CSV to temp file
    tmp = "input_data.csv"
    with open(tmp, "w") as f:
        f.write(csv_text)
    df = pd.read_csv(tmp)
    # Simple analysis: compute correlations and frequent values
    corr = df.corr().abs()
    # find top correlated pairs
    idx = np.triu_indices_from(corr, k=1)
    pairs = sorted(
        [((df.columns[i], df.columns[j]), corr.values[i, j]) for i, j in zip(*idx)],
        key=lambda x: -x[1]
    )[:5]
    patterns = []
    for (a, b), score in pairs:
        patterns.append(f"Correlation({a},{b})={score:.3f}")
    return "\n".join(patterns)

###CSV analysis tool
danalyze_csv_tool = Tool.from_function(
    analyze_csv_dataset,
    name="analyze_csv_dataset",
    description="Analyze a CSV dataset provided as text and extract patterns for conjecture generation."
)

# Fill this list with common Mathlib declarations available in your Lake environment
MATHLIB_NAMES = [
    "Nat.add_comm", "Nat.mul_comm", "List.map_nil", "Monoid.mul_one",
    # ... populate as needed or load dynamically via lean server
]

def _find_related_mathlib(
    term: str,
    lean_server: LeanServer,
    k: int = 3
) -> List[Tuple[str, str]]:
    """
    Find up to k nearest declarations in Mathlib by name similarity,
    then fetch their definitions.
    """
    matches = get_close_matches(term, MATHLIB_NAMES, n=k, cutoff=0.5)
    print(f"[debug] matches for {term!r} → {matches}", flush=True)

    results: List[Tuple[str, str]] = []
    for name in matches:
        definition = fetch_definition_from_mathlib(name)
        results.append((name, definition))
    return results

find_related_mathlib = partial(_find_related_mathlib, lean_server=lean_server, k=3)

find_mathlib_knn_tool = Tool.from_function(
    find_related_mathlib,
    name="find_related_mathlib",
    description="Given a term, find the 3 closest Mathlib names and return their definitions."
)

lean_server = init_lean_server()





