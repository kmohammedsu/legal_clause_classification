# Wrapper for numbered script: use 01_load_cuad_clauses (run order).
# Notebooks import from here so "from scripts.load_cuad_clauses import get_clauses_df, ..." works.
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent / "01_load_cuad_clauses.py"
_spec = importlib.util.spec_from_file_location("load_cuad_clauses", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_clauses_df = _mod.get_clauses_df
extract_clauses_from_cuadv1 = _mod.extract_clauses_from_cuadv1
