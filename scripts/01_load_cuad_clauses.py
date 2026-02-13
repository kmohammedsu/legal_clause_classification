"""
01 – Load clauses from CUAD JSON (CUADv1.json or test.json).
Used by 02_eda and 03_data_preprocessing notebooks.

Usage (from project root, with data_path set):
    # In notebook: load via importlib from scripts/01_load_cuad_clauses.py
    clauses_df = get_clauses_df(data_path)
    test_df = extract_clauses_from_cuadv1(data_path / "test.json")
"""

import json
from pathlib import Path
from typing import Union

import pandas as pd


def extract_clauses_from_cuadv1(json_filepath: Union[Path, str]) -> pd.DataFrame:
    """
    Extract all clauses (answer spans) with their categories from a CUAD JSON file.
    Works with CUADv1.json or test.json.

    Returns a DataFrame with columns: contract_id, clause_text, category,
    answer_start, answer_end, text_length, word_count.
    """
    json_filepath = Path(json_filepath)
    clauses = []

    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "data" not in data:
        print("[WARNING]  No 'data' key found in JSON file")
        return pd.DataFrame()

    contracts = data["data"]
    print(f"Processing {len(contracts)} contracts...")

    for contract in contracts:
        contract_id = contract.get("title", "unknown")

        if "paragraphs" not in contract:
            continue

        for para in contract["paragraphs"]:
            context = para.get("context", "")

            if "qas" not in para:
                continue

            for qa in para["qas"]:
                question = qa.get("question", "")

                category = None
                if 'related to "' in question:
                    category = question.split('related to "')[1].split('"')[0]

                if not category:
                    continue

                answers = qa.get("answers", [])
                is_impossible = qa.get("is_impossible", False)

                if not is_impossible and answers:
                    for answer in answers:
                        answer_text = answer.get("text", "").strip()
                        answer_start = answer.get("answer_start", None)

                        if answer_text:
                            clauses.append(
                                {
                                    "contract_id": contract_id,
                                    "clause_text": answer_text,
                                    "category": category,
                                    "answer_start": answer_start,
                                    "answer_end": (
                                        answer_start + len(answer_text)
                                        if answer_start is not None
                                        else None
                                    ),
                                    "text_length": len(answer_text),
                                    "word_count": len(answer_text.split()),
                                }
                            )

    df = pd.DataFrame(clauses)

    if not df.empty:
        print(f"\n[OK] Extracted {len(df)} clauses from {len(contracts)} contracts")
        print(f"   Unique categories: {df['category'].nunique()}")
        print(f"   Unique contracts: {df['contract_id'].nunique()}")
    else:
        print("[WARNING]  No clauses extracted")

    return df


def get_clauses_df(data_path: Union[Path, str]) -> pd.DataFrame:
    """
    Load clauses from CUADv1.json in the given data directory.
    Returns the same DataFrame as extract_clauses_from_cuadv1(cuadv1_path).
    """
    data_path = Path(data_path)
    cuadv1_file = data_path / "CUADv1.json"
    print("EXTRACTING CLAUSES FROM CUADv1.json")
    return extract_clauses_from_cuadv1(cuadv1_file)
