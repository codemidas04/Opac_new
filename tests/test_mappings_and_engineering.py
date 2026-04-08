import ast
import io
import re
import textwrap
import ast
import pandas as pd
import numpy as np
import pytest


BASE = "streamlit_app.py"


def _extract_dict_from_source(source, dict_name):
    # crude regex to find dict literal assigned to dict_name
    pat = rf"{dict_name}\s*=\s*\{{"
    m = re.search(pat, source)
    if not m:
        raise RuntimeError(f"{dict_name} not found")
    start = m.start()
    # find matching closing brace by scanning
    i = m.end() - 1
    depth = 0
    for j in range(i, len(source)):
        ch = source[j]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    else:
        raise RuntimeError("couldn't find end of dict")
    dict_src = source[m.start():end]
    # make valid Python by trimming anything before the first brace
    dict_src = dict_src.split('=', 1)[1].strip()
    return ast.literal_eval(dict_src)


def _extract_function_source(source, func_name):
    pat = rf"def\s+{func_name}\s*\(" 
    m = re.search(pat, source)
    if not m:
        raise RuntimeError(f"{func_name} not found")
    start = m.start()
    # naive: find next '\ndef ' at column 0 after start
    rest = source[start:]
    # look for '\n\ndef ' which indicates next top-level def; if not found, take to end
    m2 = re.search(r"\n\ndef\s+", rest)
    if m2:
        func_src = rest[:m2.start()]
    else:
        func_src = rest
    return func_src


def test_purpose_map_has_business_and_other():
    src = open(BASE, "r", encoding="utf-8").read()
    pm = _extract_dict_from_source(src, "Purpose_map")
    assert isinstance(pm, dict)
    # check requested mappings
    assert pm.get("Business") == "A49", "Business should map to A49"
    assert pm.get("Other") == "A410", "Other should map to A410"


def test_compute_engineered_features_age_bins():
    # extract function source and exec it in a safe namespace
    src = open(BASE, "r", encoding="utf-8").read()
    func_src = _extract_function_source(src, "compute_engineered_features")
    ns = {"pd": pd, "np": np}
    exec(textwrap.dedent(func_src), ns)
    compute_engineered_features = ns.get("compute_engineered_features")
    assert compute_engineered_features is not None

    df = pd.DataFrame([
        {"Duration_in_month": 12, "Credit_amount": 1200, "Age_in_years": 22},
        {"Duration_in_month": 24, "Credit_amount": 2400, "Age_in_years": 34},
        {"Duration_in_month": 36, "Credit_amount": 3600, "Age_in_years": 50},
        {"Duration_in_month": 48, "Credit_amount": 4800, "Age_in_years": 70},
    ])

    out = compute_engineered_features(df)
    assert "age_bin" in out.columns
    assert out.loc[0, "age_bin"] == "18-25"
    assert out.loc[1, "age_bin"] == "26-35"
    assert out.loc[2, "age_bin"] == "46-55" or out.loc[2, "age_bin"] == "36-45"  # depending on boundary
    assert out.loc[3, "age_bin"] == "65+"
