#!/usr/bin/env python3
"""
Minimal Spyder-run script:
 - reads mapping CSV and a zipped segment CSV
 - filters rows for DESCRIPTION
 - plots distance vs avg passengers/departure with payload-range overlays
 - adds histograms above (distance) and right (passengers per departure)
"""

import os, zipfile, io
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import gridspec

# ------------- USER CONFIG -------------
ZIP_PATH = os.path.join(os.getcwd(), 'nils', "T_T100_SEGMENT_ALL_CARRIER_20251016_023349.zip")
MAP_CSV  = os.path.join(os.getcwd(), 'nils', "L_AIRCRAFT_TYPE.csv")
# DESCRIPTION = "A220-300 BD-500-1A11"
DESCRIPTION = "Airbus Industrie A320-100/200"
OUT_PNG = os.path.join(os.getcwd(), "distance_vs_passengers_marginal_hist.png")
# names expected in the big CSV (must match the file)
COL_AIR = "AIRCRAFT_TYPE"
COL_DIST = "DISTANCE"
COL_PASS = "PASSENGERS"
COL_DEPS = "DEPARTURES_PERFORMED"
COL_PL = "PAYLOAD"
# payload-range data file and sheet names
PR_XLSX = os.path.join(os.getcwd(), 'nils', 'pr_data_ref_ac.xlsx')
# sheets in your file
SHEET_ATR72 = 'ATR 72-600'
SHEET_A220 = 'A220-300'
SHEET_A320NEO = 'A320 neo'
# ------------- end config -------------

# load mapping (assume 'Code' and 'Description' column names)
m = pd.read_csv(MAP_CSV, dtype=str)
m_index = m.set_index('Code')['Description'].astype(str).to_dict()
# find codes matching DESCRIPTION (case-insensitive substring)
desc_lower = DESCRIPTION.strip().lower()
matched_codes = [k for k, v in m_index.items() if desc_lower in v.strip().lower()]
if not matched_codes:
    raise SystemExit("No matching codes for DESCRIPTION.")

# pick first CSV-like entry in ZIP
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    inner_candidates = [n for n in zf.namelist() if n.lower().endswith(('.csv', '.txt'))]
    inner_name = inner_candidates[0] if inner_candidates else zf.namelist()[0]
    with zf.open(inner_name, 'r') as fh:
        df = pd.read_csv(io.BytesIO(fh.read()), dtype=str)

# filter rows for matched aircraft types (string compare)
codes_set = set([str(c).strip() for c in matched_codes])
df = df[df[COL_AIR].astype(str).str.strip().isin(codes_set)].copy()

# compute avg passengers per departure
df[COL_PASS] = pd.to_numeric(df[COL_PASS], errors='coerce')
df[COL_DEPS] = pd.to_numeric(df[COL_DEPS], errors='coerce')
df[COL_PL] = pd.to_numeric(df[COL_PL], errors='coerce')
df[COL_DIST] = pd.to_numeric(df[COL_DIST], errors='coerce')
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[COL_PASS, COL_DEPS, COL_DIST, COL_PL])
avg_pax = df[COL_PASS] / df[COL_DEPS]
# avg_pax = df[COL_PL] / df[COL_DEPS]
distance = df[COL_DIST]

# load payload-range data (each sheet: two columns, x,y)
atr72600_pr_data = pd.read_excel(PR_XLSX, sheet_name=SHEET_ATR72, header=None, skiprows=1).to_numpy()
a220300_pr_data  = pd.read_excel(PR_XLSX, sheet_name=SHEET_A220, header=None, skiprows=1).to_numpy()
a320neo_pr_data  = pd.read_excel(PR_XLSX, sheet_name=SHEET_A320NEO, header=None, skiprows=1).to_numpy()

#%%

from matplotlib_custom_settings import *

# plotting layout: top hist, left scatter, right hist
fig = plt.figure(figsize=(12,9))
gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[1,4], hspace=0.05, wspace=0.05)

ax_histx = fig.add_subplot(gs[0,0])      # top histogram (distance)
ax_scatter = fig.add_subplot(gs[1,0])    # main scatter
ax_histy = fig.add_subplot(gs[1,1])      # right histogram (avg pax)

# # top histogram
# ax_histx.hist(distance, bins=40, edgecolor='black', facecolor='grey')
# ax_histx.set_xlim(distance.min(), distance.max())
# ax_histx.set_xticklabels([])
# ax_histx.set_yticklabels([])
# ax_histx.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# # ax_histx.set_ylabel("Count")

# =============================================================================
ax_histx.boxplot(distance, vert=False, widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor='none', color='red'),
                 medianprops=dict(color='red', linewidth=1.5))
ax_histx.set_xlim(distance.min(), distance.max())
ax_histx.axis('off')  # cleaner look
# =============================================================================

# # right histogram (horizontal)
# ax_histy.hist(avg_pax, bins=40, orientation='horizontal', edgecolor='black', facecolor='grey')
# ax_histy.set_ylim(avg_pax.min(), avg_pax.max())
# ax_histy.set_yticklabels([])
# ax_histy.set_xticklabels([])
# ax_histy.tick_params(axis='y', which='both', left=False, labelleft=False)
# # ax_histy.set_xlabel("Count")

# =============================================================================
# --- REPLACE the previous ax_histy block with this minimal, robust version ---
bp = ax_histy.boxplot([avg_pax.dropna().values], vert=True, positions=[1],
                      widths=0.6, patch_artist=True, manage_ticks=False)

# style the boxplot artists so they are visible
box = bp['boxes'][0]; box.set_facecolor('none'); box.set_edgecolor('red'); box.set_linewidth(0.9)
for whisk in bp['whiskers']: whisk.set_color('black'); whisk.set_linewidth(0.8)
for cap in bp['caps']: cap.set_color('black'); cap.set_linewidth(0.8)
for med in bp['medians']: med.set_color('red'); med.set_linewidth(1.4)
for flier in bp.get('fliers', []): flier.set_marker('o'); flier.set_markeredgecolor('black'); flier.set_alpha(0.6)

# zoom into the narrow x-range that contains the single vertical box and align y-range with scatter
ax_histy.set_xlim(0.5, 1.5)
ax_histy.set_ylim(ax_scatter.get_ylim())

# hide ticks/spines (but not artists)
ax_histy.set_xticks([]); ax_histy.set_yticks([])
ax_histy.spines['top'].set_visible(False)
ax_histy.spines['right'].set_visible(False)
ax_histy.spines['left'].set_visible(False)
ax_histy.spines['bottom'].set_visible(False)
# =============================================================================

# # =============================================================================
# import matplotlib.patches as mpatches

# # --- after both boxplots (ax_histx, ax_histy) and before plt.tight_layout() ---

# # Get whisker limits from horizontal (distance) boxplot
# x_whiskers = [line.get_xdata() for line in ax_histx.lines if len(line.get_xdata()) == 2]
# x_vals = np.array(x_whiskers).ravel()
# x_min, x_max = np.min(x_vals), np.max(x_vals)

# # Get whisker limits from vertical (avg_pax) boxplot
# y_whiskers = [line.get_ydata() for line in ax_histy.lines if len(line.get_ydata()) == 2]
# y_vals = np.array(y_whiskers).ravel()
# y_min, y_max = np.min(y_vals), np.max(y_vals)

# # Draw rectangle on the main scatter axis
# rect = mpatches.Rectangle((x_min, y_min),   # bottom-left corner
#                           x_max - x_min,    # width
#                           y_max - y_min,    # height
#                           facecolor='none', edgecolor='red', linewidth=2, zorder=5)
# ax_scatter.add_patch(rect)
# # =============================================================================

# =============================================================================
import matplotlib.patches as mpatches
import numpy as np

# compute IQR box extents directly from data (ignore NaNs)
x_q1, x_q3 = np.nanpercentile(distance.values, [25, 75])
y_q1, y_q3 = np.nanpercentile(avg_pax.values,  [25, 75])

# create rectangle at (x_q1, y_q1) with width/height = Q3-Q1
rect = mpatches.Rectangle(
    (x_q1, y_q1),
    x_q3 - x_q1,
    y_q3 - y_q1,
    facecolor='none',
    edgecolor='red',
    linewidth=2.0,
    zorder=6
)
ax_scatter.add_patch(rect)
# =============================================================================

# main scatter
ax_scatter.scatter(distance, avg_pax, s=50, alpha=0.5, facecolor='grey', edgecolor='none', label='US flight data')
# # overlay payload-range curves
# if atr72600_pr_data.size:
#     ax_scatter.plot(atr72600_pr_data[:,0], atr72600_pr_data[:,1], '-k', lw=1.2, label='ATR72 payload-range')
# if a220300_pr_data.size:
#     ax_scatter.plot(a220300_pr_data[:,0], a220300_pr_data[:,1], '-C1', lw=1.2, label='A220-300 payload-range')
# if a320neo_pr_data.size:
# =============================================================================
a320neo_pax_max = np.nanmax(df[COL_PASS].to_numpy() / df[COL_DEPS].to_numpy())
# =============================================================================
ax_scatter.plot(a320neo_pr_data[:,0], a320neo_pr_data[:,1] * a320neo_pax_max / np.max(a320neo_pr_data[:,1]) , color='black', lw=1.2, label='Payload-range')

ax_scatter.set_xlabel("Distance (nm)")
ax_scatter.set_ylabel("Average passenger number per departure (-)")
ax_scatter.legend(loc='upper right', frameon=False)

ax_scatter.spines[['right', 'top']].set_visible(False)
ax_scatter.tick_params(axis='both', which='both', length=0)
# ax_scatter.tick_params(axis='x', which='both', length=0)

# align hist x limits with main scatter
ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histy.set_ylim(ax_scatter.get_ylim())

ax_histx.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
ax_histx.tick_params(axis='both', which='both', length=0)

ax_histy.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
ax_histy.tick_params(axis='both', which='both', length=0)

plt.tight_layout()
plt.show()
