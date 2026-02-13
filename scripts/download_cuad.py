# Wrapper for numbered script: use 00_download_cuad (run order).
# Notebooks import from here so "from scripts.download_cuad import ensure_cuad_data" works.
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent / "00_download_cuad.py"
_spec = importlib.util.spec_from_file_location("download_cuad", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ensure_cuad_data = _mod.ensure_cuad_data
get_project_root = _mod.get_project_root
