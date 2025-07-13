from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun 
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import subprocess
import mathlib 
import os
from lean_interact import LeanREPLConfig, LeanServer, Command, FileCommand, ProofStep
from typing import List, Tuple, Dict
import sympy as sp
from collections import Counter
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sympy.parsing.latex import parse_latex

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
    

#Lean helpers

def init_lean_server(verbose: bool = False, lean_version: str = None) -> LeanServer:
    """
    Configure and start a Lean 4 server via lean-interact.
    """
    config = LeanREPLConfig(verbose=verbose, lean_version=lean_version)
    server = LeanServer(config)
    return server

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


def filter_motifs_with_lean(motifs: List[str], lean_server: LeanServer) -> List[Tuple[str, str]]:
    """
    For each motif (string), check if Lean can type-check or simplify it.
    Returns a list of (motif, Lean message) pairs that passed.
    """
    valid = []
    for motif in motifs:
        try:
            cmd = f"#check {motif}"
            resp = lean_server.run(Command(cmd=cmd))
            if resp.messages:
                valid.append((motif, resp.messages[0]))  # keep 1st message (usually the type)
        except Exception as e:
            continue  # skip motifs that Lean rejects outright
    return valid