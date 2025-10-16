#!/usr/bin/env python3
"""
Spyder-run script:
 - Reads L_AIRCRAFT_TYPE.csv (mapping Code -> Description)
 - Reads T_T100_SEGMENT_ALL_CARRIER_20251014_172649.zip (finds inner CSV)
 - Filters rows whose AIRCRAFT_TYPE corresponds to the given Description
 - Plots DISTANCE vs PASSENGERS for those flights and saves a PNG

Edit the variables in the "USER CONFIG" section below, then press Run in Spyder.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ USER CONFIG (edit these) ------------------
# Path to ZIP (or leave as filename if it's in current working dir)
# ZIP_PATH = os.path.join(os.getcwd(), 'nils', "T_T100_SEGMENT_ALL_CARRIER_20251014_172649.zip")
ZIP_PATH = os.path.join(os.getcwd(), 'nils', "T_T100_SEGMENT_ALL_CARRIER_20251016_023349.zip")

# Path to mapping CSV (Code, Description)
MAP_CSV = os.path.join(os.getcwd(), 'nils', "L_AIRCRAFT_TYPE.csv")

# Description to match (exact case-insensitive preferred, otherwise substring)
# DESCRIPTION = "Aerospatiale/Aeritalia ATR-72"
DESCRIPTION = "A220-300 BD-500-1A11"
# DESCRIPTION = "Airbus Industrie A320-100/200"

# Output PNG path
OUT_PNG = os.path.join(os.getcwd(), "distance_vs_passengers_from_spyder.png")

# Chunk size for streaming the large CSV inside the ZIP
CHUNKSIZE = 200_000
# ---------------------------------------------------------------

def load_code_description_map(map_csv_path):
    """Load mapping CSV and return dict: code_str -> description_str"""

    df = pd.read_csv(map_csv_path, dtype=str, keep_default_na=False)
    # Heuristic detection for code/description columns
    cols_lower = [c.lower() for c in df.columns]
    desc_col = df.columns[cols_lower.index("description")]
    code_col = df.columns[cols_lower.index("code")]
    series = df.set_index(code_col)[desc_col].astype(str)
    # normalize keys/values
    map_dict = {str(k).strip(): str(v).strip() for k, v in series.to_dict().items()}
    
    return map_dict


def pick_inner_csv_from_zip(zip_path):
    """Return the name of the most likely CSV/text file inside the zip."""
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    # prefer files that look like CSV
    candidates = [n for n in names if n.lower().endswith((".csv", ".txt"))]
    if candidates:
        return candidates[0]
    
    raise RuntimeError("ZIP archive appears empty.")


def sniff_columns_from_sample(zip_path, inner_name, nbytes=200_000):
    """Read a small byte sample from the inner file and try to parse columns."""
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner_name, "r") as fh:
            sample = fh.read(nbytes)
    # Try default encoding then latin1 fallback
    sample_df = pd.read_csv(io.BytesIO(sample), nrows=5)
    cols = sample_df.columns.tolist()
    
    return cols, sample_df


def detect_required_columns(sample_cols, sample_df=None):
    """Detect names for aircraft, distance and passengers in the big CSV."""
    
    # common names
    for c in sample_cols:
        cl = c.lower()
        if cl == "aircraft_type":
            col_air = c
        if cl == "distance":
            col_distance = c
        if cl == "passengers":
            col_pass = c
        if cl == "departures_scheduled":
            col_deps_scheduled = c
        if cl == "departures_performed":
            col_deps_performed = c
        if cl == "seats":
            col_seats = c
            
    return col_air, col_distance, col_pass, col_deps_scheduled, col_deps_performed, col_seats


def find_codes_for_description(map_dict, description):
    """Return list of map codes whose description matches description (exact then substring).
       Also returns helpful message if none found."""
       
    desc_target = str(description).strip().lower()
    exact = [code for code, desc in map_dict.items() if str(desc).strip().lower() == desc_target]
    if exact:
        return exact
    # substring match
    sub = [code for code, desc in map_dict.items() if desc_target in str(desc).strip().lower()]
    
    return sub


def build_code_variants_set(codes):
    """Return set with original codes and numeric-normalized variants (strip leading zeros)."""
    s = set()
    for c in codes:
        sc = str(c).strip()
        s.add(sc)
        # strip leading zeros
        try:
            snz = str(int(sc))
            s.add(snz)
        except Exception:
            pass
        # also add variant without leading zeros if present
        s.add(sc.lstrip("0"))
    return s


def stream_filter_distance_passengers(zip_path, inner_name, aircraft_col, dist_col, pass_col, col_deps_scheduled, col_deps_performed, col_seats,
                                      matching_codes_set, chunksize=CHUNKSIZE):
    """Stream the big CSV inside ZIP in chunks; return list of (distance, passengers) floats."""
    pairs = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner_name, "r") as fh:
            # pandas can read a file-like object in chunks
            for chunk in pd.read_csv(fh, chunksize=chunksize, dtype=str):
                # ensure columns exist in this chunk
                if aircraft_col not in chunk.columns:
                    continue
                # coerce to string and strip spaces for safe comparison
                col_values = chunk[aircraft_col].astype(str).str.strip()
                mask = col_values.isin(matching_codes_set)
                if not mask.any():
                    continue
                sel = chunk.loc[mask, [dist_col, pass_col, col_deps_scheduled, col_deps_performed, col_seats]].copy()
                # convert to numeric, drop rows with missing
                sel[dist_col] = pd.to_numeric(sel[dist_col], errors="coerce")
                sel[pass_col] = pd.to_numeric(sel[pass_col], errors="coerce")
                sel[col_deps_scheduled] = pd.to_numeric(sel[col_deps_scheduled], errors="coerce")
                sel[col_deps_performed] = pd.to_numeric(sel[col_deps_performed], errors="coerce")
                sel[col_seats] = pd.to_numeric(sel[col_seats], errors="coerce")
                sel = sel.dropna(subset=[dist_col, pass_col, col_deps_scheduled, col_deps_performed, col_seats])
                if sel.shape[0] > 0:
                    arr = sel.to_numpy(dtype=float)
                    pairs.extend([(float(a), float(b), float(c), float(d), float(e)) for a, b, c, d, e in arr])
    return pairs

# ------------------ Execution (Spyder-friendly) ------------------
print("=== Fan-run: Filter & plot DISTANCE vs PASSENGERS ===")
print("ZIP_PATH:", ZIP_PATH)
print("MAP_CSV:", MAP_CSV)
print("DESCRIPTION:", DESCRIPTION)
print()

# 1) Load mapping
map_dict = load_code_description_map(MAP_CSV)
print(f"Loaded mapping entries: {len(map_dict)}")

# 2) Choose inner CSV from ZIP
inner = pick_inner_csv_from_zip(ZIP_PATH)
print("Using inner file from ZIP:", inner)

# 3) Read sample and detect columns
cols, sample_df = sniff_columns_from_sample(ZIP_PATH, inner)
air_col, dist_col, pass_col, col_deps_scheduled, col_deps_performed, col_seats = detect_required_columns(cols, sample_df)
print("Detected columns:", "AIRCRAFT=", air_col, "DISTANCE=", dist_col, "PASSENGERS=", pass_col, "DEPARTURES_SCHEDULED=", col_deps_scheduled)

# 4) Find matching map codes for the requested description
matched_codes = find_codes_for_description(map_dict, DESCRIPTION)
if not matched_codes:
    print("No codes found matching description. Sample of available descriptions (first 20):")
    for desc in list(map_dict.values())[:20]:
        print("   ", desc)
    raise RuntimeError("No matching aircraft codes found for DESCRIPTION: " + str(DESCRIPTION))
print("Matched mapping codes (count):", len(matched_codes), matched_codes[:20])

# Build set of code variants to be robust to leading zeros / numeric forms
matching_codes_set = build_code_variants_set(matched_codes)
print("Matching codes variants (examples):", list(matching_codes_set)[:20])

# 5) Stream filter the CSV inside ZIP and collect (distance, passengers) pairs
pairs = stream_filter_distance_passengers(ZIP_PATH, inner, air_col, dist_col, pass_col, col_deps_scheduled, col_deps_performed, col_seats,
                                          matching_codes_set, chunksize=CHUNKSIZE)
print("Collected pairs (distance, passengers):", len(pairs))

# 6) Plot
if len(pairs) == 0:
    print("No flights found for this DESCRIPTION and mapped codes. Nothing to plot.")
else:
    arr = np.array(pairs, dtype=float)
    distance = arr[:, 0]
    passengers = arr[:, 1]
    deps_scheduled = arr[:, 2]
    deps_performed = arr[:, 3]
    seats = arr[:, 4]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(distance, passengers / deps_performed, s=18, alpha=0.6)
    # ax.scatter(distance, seats / deps_performed, s=18, alpha=0.6)
    ax.set_xlabel("DISTANCE")
    ax.set_ylabel("PASSENGERS")
    ax.set_title(f"DISTANCE vs PASSENGERS for: {DESCRIPTION} (codes: {', '.join(matched_codes[:8])})")
    ax.grid(True)
    plt.tight_layout()
    # save and show
    fig.savefig(OUT_PNG, dpi=220)
    print("Saved plot to:", OUT_PNG)
    plt.show()

print("Done.")

#%%

atr72600_pr_data = pd.read_excel(os.path.join(os.getcwd(), 'nils', 'pr_data_ref_ac.xlsx'), header=None, skiprows=1, sheet_name='ATR 72-600').to_numpy()
a220300_pr_data = pd.read_excel(os.path.join(os.getcwd(), 'nils', 'pr_data_ref_ac.xlsx'), header=None, skiprows=1, sheet_name='A220-300').to_numpy()
a320neo_pr_data = pd.read_excel(os.path.join(os.getcwd(), 'nils', 'pr_data_ref_ac.xlsx'), header=None, skiprows=1, sheet_name='A320 neo').to_numpy()



