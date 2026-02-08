"""

"""

__all__ = []

import os
import io
import sys
import glob
import zipfile
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FuncFormatter

from matplotlib_custom_settings import *

def one_sig_fig(x, pos):
    return f"{x:.1f}"

ac_type = 'ATR72'  # 'A220-100', 'A220-300', 'A320', or 'ATR72'
distribution_plot_type = 'histogram'  # 'histogram' or 'box_and_whisker'

#%% File imports

folder = r"C:\Users\nmb48\Documents\GitHub\eurocontrol-data-archive\nils"
zip_files = glob.glob(os.path.join(folder, 'T_T100_*.zip'))

# MAPPING DATA

# .csv mapping file that associates an integer "Code" to each aircraft type for filtering the .zip folder
map_csv_file = os.path.join(os.getcwd(), 'nils', "L_AIRCRAFT_TYPE.csv")

# OPERATING DATA

# .csv file contained in a .zip folder containing the operating data
zip_file = os.path.join(os.getcwd(), 'nils', "T_T100_SEGMENT_ALL_CARRIER_20251016_023349.zip")

# AIRCRAFT DATA

# Minimal data required to construct payload-range envelope for each aircraft type
pr_xlsx_file = os.path.join(os.getcwd(), 'nils', 'pr_data_ref_ac.xlsx')

# sys.exit()

fig = plt.figure(figsize=(12,9))
gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[1,4], hspace=0.2, wspace=0.05)

ax_histx = fig.add_subplot(gs[0,0])
ax_scatter = fig.add_subplot(gs[1,0])
ax_histy = fig.add_subplot(gs[1,1])

# ax_scatter.set_zorder(10)
# ax_histx.set_zorder(1)
# ax_histy.set_zorder(1)

#%% Data processing

all_xticks = []
all_yticks = []

ac_type_iter_list = [0, 4]
for ac_type_iter, ac_type in zip(ac_type_iter_list, ['A220-100', 'ATR72']):

    # AIRCRAFT DATA
    
    # Extract data from various sheets
    if ac_type == 'ATR72':
        pr_data = pd.read_excel(pr_xlsx_file, sheet_name='ATR 72-600', header=None, skiprows=1).to_numpy()
        m_1_pax = 7500 / 78 / 0.453592  # lbs
        pr_data[:,0] *= 1.852 / 1e3  # nm to 1000 km
        pr_data[:,1] = (pr_data[:,1] - np.min(pr_data[:,1])) / 1e3  # kg to tonnes
        N_bins_horz, N_bins_vert = 6, 15
    elif ac_type == 'A220-100':
        pr_data = pd.read_excel(pr_xlsx_file, sheet_name='A220-100', header=None, skiprows=1).to_numpy()
        pr_data[:,0] *= 1.852 / 1e3  # nm to 1000 km
        # pr_data[:,1] *= 0.453592  # 1000 lb to tonnes
        # pr_data[:,1] = 120 * (pr_data[:,1] - np.min(pr_data[:,1])) / (np.max(pr_data[:,1]) - np.min(pr_data[:,1]))
        pr_data[:,1] = (pr_data[:,1] - np.min(pr_data[:,1])) * 0.453592  # 1000 lb to tonnes
        m_1_pax = (104.53e3 - 78050) / 120  # lbs
        N_bins_horz, N_bins_vert = 40, 25
    elif ac_type == 'A220-300':
        pr_data = pd.read_excel(pr_xlsx_file, sheet_name='A220-300', header=None, skiprows=1).to_numpy()
        pr_data[:,0] *= 1852 / 1e3  # nm to 1000 km
        # pr_data[:,1] *= 0.453592 * 1e3  # 1000 lb to kg
        pr_data[:,1] = 140 * (pr_data[:,1] - np.min(pr_data[:,1])) / (np.max(pr_data[:,1]) - np.min(pr_data[:,1]))
    elif ac_type == 'A320':
        pr_data = pd.read_excel(pr_xlsx_file, sheet_name='A320 neo', header=None, skiprows=1).to_numpy()
    
    # MAPPING DATA
    
    # Choose aircraft type (check .csv file for available types)
    if ac_type == 'ATR72':
        ac_descr = "Aerospatiale/Aeritalia ATR-72"
    elif ac_type == 'A220-100':
        ac_descr = "A200-100 BD-500-1A10"
    elif ac_type == 'A220-300':
        ac_descr = "A220-300 BD-500-1A11"
    elif ac_type == 'A320':
        ac_descr = "Airbus Industrie A320-100/200"
    
    # Load mapping data
    m = pd.read_csv(map_csv_file, dtype=str)
    m_index = m.set_index('Code')['Description'].astype(str).to_dict()
    
    # Find codes matching aircraft type
    ac_descr_lower = ac_descr.strip().lower()
    ac_descr_matched_codes = [k for k, v in m_index.items() if ac_descr_lower in v.strip().lower()]
    if not ac_descr_matched_codes:
        raise SystemExit("No matching codes for ac_descr.")
    
    # OPERATING DATA
    
    df_list = []
    avg_pax_list = []
    avg_freight_list = []
    avg_mail_list = []
    avg_payload_list = []
    distance_list = []
    
    for zip_file in zip_files:
    
        # Create dataframe from contents of .csv file in .zip folder
        with zipfile.ZipFile(zip_file, 'r') as zf:
            inner_candidates = [n for n in zf.namelist() if n.lower().endswith(('.csv', '.txt'))]
            inner_name = inner_candidates[0] if inner_candidates else zf.namelist()[0]
            with zf.open(inner_name, 'r') as fh:
                df = pd.read_csv(io.BytesIO(fh.read()), dtype=str)
        
        # Filter rows of dataframe for matched aircraft types
        codes_set = set([str(c).strip() for c in ac_descr_matched_codes])
        df = df[df["AIRCRAFT_TYPE"].astype(str).str.strip().isin(codes_set)].copy()
        
        # Convert strings in dataframe to numbers where possible
        df["PASSENGERS"] = pd.to_numeric(df["PASSENGERS"], errors='coerce')
        df["FREIGHT"] = pd.to_numeric(df["FREIGHT"], errors='coerce')  # lbs
        df["MAIL"] = pd.to_numeric(df["MAIL"], errors='coerce')  # lbs
        df["DEPARTURES_PERFORMED"] = pd.to_numeric(df["DEPARTURES_PERFORMED"], errors='coerce')
        df["PAYLOAD"] = pd.to_numeric(df["PAYLOAD"], errors='coerce')  # NOTE: this is the certified payload - does NOT vary with mission and is therefore useless!
        df["DISTANCE"] = pd.to_numeric(df["DISTANCE"], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["PASSENGERS", "DEPARTURES_PERFORMED", "DISTANCE", "PAYLOAD"])  # convert infinities to nans and remove bad rows in specified columns
        
        # Filter out freighter versions (of ATR 72-600)
        df = df[df["PASSENGERS"] > 0].copy()
        
        # Compute avg passengers per departure and extract distance
        avg_pax = df["PASSENGERS"] / df["DEPARTURES_PERFORMED"]
        avg_freight = df["FREIGHT"] / df["DEPARTURES_PERFORMED"]
        avg_mail = df["MAIL"] / df["DEPARTURES_PERFORMED"]
        avg_payload = (df["PASSENGERS"] * m_1_pax + df["FREIGHT"] + df["MAIL"]) / df["DEPARTURES_PERFORMED"] * 0.453592 / 1e3  # lbs to tonnes
        distance = df["DISTANCE"] * 1.852 / 1e3  # nm to 1000 km
        
        df_list.append(df)
        avg_pax_list.append(avg_pax)
        avg_freight_list.append(avg_freight)
        avg_mail_list.append(avg_mail)
        avg_payload_list.append(avg_payload)
        distance_list.append(distance)
        
        
    df = pd.concat(df_list, axis=0, ignore_index=True)
    avg_pax = pd.concat(avg_pax_list, ignore_index=True)
    avg_freight = pd.concat(avg_freight_list, ignore_index=True)
    avg_mail = pd.concat(avg_mail_list, ignore_index=True)
    avg_payload = pd.concat(avg_payload_list, ignore_index=True)
    avg_pax = avg_payload
    distance = pd.concat(distance_list, ignore_index=True)
    
    #%% Plotting
    
    # Main plot
    
    # Compute average
    x_mean = np.nanmean(distance.values)
    y_mean = np.nanmean(avg_pax.values)
    
    ax_scatter.scatter(
        x_mean, y_mean,
        marker='s',
        edgecolor=colors[ac_type_iter],
        facecolor=opaque_color_from_hex(colors[ac_type_iter], alpha=0.4),
        zorder=100,
    )
    
    # Draw IQR window
    
    # Compute IQR box extents directly from data (ignore NaNs)
    x_q1, x_q3 = np.nanpercentile(distance.values, [25, 75])
    y_q1, y_q3 = np.nanpercentile(avg_pax.values,  [25, 75])
    
    # Create rectangle at (x_q1, y_q1) with width/height = Q3-Q1
    rect = mpatches.Rectangle(
        (x_q1, y_q1),
        x_q3 - x_q1,
        y_q3 - y_q1,
        facecolor='none',
        edgecolor=colors[ac_type_iter],
        linestyle='-',
        hatch='//',
        zorder=6
    )
    ax_scatter.add_patch(rect)
    # ax_scatter.plot([0, x_q1], [y_q3, y_q3], color=colors[ac_type_iter], linestyle='--')
    # ax_scatter.plot([0, x_q1], [y_q1, y_q1], color=colors[ac_type_iter], linestyle='--')
    # ax_scatter.plot([x_q1, x_q1], [0, y_q1], color=colors[ac_type_iter], linestyle='--')
    # ax_scatter.plot([x_q3, x_q3], [0, y_q1], color=colors[ac_type_iter], linestyle='--')
    
    # Raw operating data
    ax_scatter.scatter(distance, avg_pax, marker='.', color=opaque_color_from_hex(colors[ac_type_iter], alpha=0.4), label='US flight data')
    
    # Payload-range envelope
    ac_pax_max = np.nanmax(df["PASSENGERS"].to_numpy() / df["DEPARTURES_PERFORMED"].to_numpy())
    ax_scatter.plot(pr_data[:,0], pr_data[:,1], color=colors[ac_type_iter], label='Payload-range')
    if ac_type == 'A220-100':
        # ax_scatter.plot([0, pr_data[3,0]], [pr_data[3,1], pr_data[3,1]], color=colors[ac_type_iter])
        x_des, y_des = pr_data[3,0], pr_data[3,1]
        ax_scatter.scatter(x_des, y_des, color=colors[ac_type_iter])
    elif ac_type == 'ATR72':
        # ax_scatter.plot([0, pr_data[2,0]], [pr_data[2,1], pr_data[2,1]], color=colors[ac_type_iter])
        x_des, y_des = pr_data[2,0], pr_data[2,1]
        ax_scatter.scatter(x_des, y_des, color=colors[ac_type_iter])
    
    # all_xticks += [x_q1, x_q3]
    # all_yticks += [y_q1, y_q3]
    all_xticks += [x_mean, x_des]
    all_yticks += [y_mean, y_des]

    if distribution_plot_type == 'histogram':
        
        # Upper histogram
        ax_histx.hist(distance, bins=N_bins_horz, edgecolor=colors[ac_type_iter], facecolor=colors[ac_type_iter], alpha=0.4)
        ax_histx.set_xlim(distance.min(), distance.max())
        
        if ac_type_iter == 0:
            existing_xticks = []
        else:
            existing_xticks = list(ax_histx.get_xticks())
        new_xticks = existing_xticks + [x_q1, x_q3]
        ax_histx.set_xticks(sorted(set(new_xticks)))
        
        ax_histx.set_yticklabels([])
    
        # Right histogram
        ax_histy.hist(avg_pax, bins=N_bins_vert, orientation='horizontal', edgecolor=colors[ac_type_iter], facecolor=colors[ac_type_iter], alpha=0.4)
        ax_histy.set_ylim(avg_pax.min(), avg_pax.max())

        if ac_type_iter == 0:
            existing_yticks = []
        else:
            existing_yticks = list(ax_histy.get_yticks())
        new_yticks = existing_yticks + [y_q1, y_q3]
        ax_histy.set_yticks(sorted(set(new_yticks)))
        
        ax_histy.set_xticklabels([])
    
    elif distribution_plot_type == 'box_and_whisker':
        
        # Upper box-and-whisker plot
        ax_histx.boxplot(distance, vert=False, widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor='none', color='red'),
                         medianprops=dict(color='red', linewidth=1.5))
        ax_histx.set_xlim(distance.min(), distance.max())
        ax_histx.axis('off')
        
        # Right box-and-whisker plot
        
        bp = ax_histy.boxplot(
            [avg_pax.dropna().values], vert=True, positions=[1],
            widths=0.6, patch_artist=True, manage_ticks=False
        )
    
        # Style the boxplot artists so they are visible
        box = bp['boxes'][0]; box.set_facecolor('none'); box.set_edgecolor('red'); box.set_linewidth(0.9)
        for whisk in bp['whiskers']: whisk.set_color('black'); whisk.set_linewidth(0.8)
        for cap in bp['caps']: cap.set_color('black'); cap.set_linewidth(0.8)
        for med in bp['medians']: med.set_color('red'); med.set_linewidth(1.4)
        for flier in bp.get('fliers', []): flier.set_marker('o'); flier.set_markeredgecolor('black'); flier.set_alpha(0.6)
    
        # Zoom into the narrow x-range that contains the single vertical box and align y-range with scatter
        ax_histy.set_xlim(0.5, 1.5)
        ax_histy.set_ylim(ax_scatter.get_ylim())
    
        # Hide ticks/spines (but not artists)
        ax_histy.set_xticks([]); ax_histy.set_yticks([])
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['left'].set_visible(False)
        ax_histy.spines['bottom'].set_visible(False)

# Main plot
ax_scatter.set_xlabel(r"Distance ($\times 1000$ km)", labelpad=15)
ax_scatter.set_ylabel("Payload (t)", labelpad=15)
ax_scatter.spines[['right', 'top']].set_visible(False)
ax_scatter.tick_params(axis='both', which='both', length=0)
ax_scatter.tick_params(axis='both', which='major', pad=15)
ax_scatter.set_xlim(left=0, right=9)
ax_scatter.set_ylim(bottom=0)

# Upper histogram
ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histx.spines[['right', 'top', 'left']].set_visible(False)
ax_histx.tick_params(axis='x', which='major', length=6, top=False)
ax_histx.tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
ax_histx.set_xticks(sorted(set(all_xticks)))
ax_histx.xaxis.set_major_locator(FixedLocator(sorted(set(all_xticks))))
ax_histx.xaxis.set_major_formatter(FuncFormatter(one_sig_fig))

# Right histogram
ax_histy.set_ylim(ax_scatter.get_ylim())
ax_histy.spines[['right', 'top', 'bottom']].set_visible(False)
ax_histy.tick_params(axis='y', which='major', length=6, right=False)
ax_histy.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False, labeltop=False)
ax_histy.set_yticks(sorted(set(all_yticks)))
ax_histy.yaxis.set_major_locator(FixedLocator(sorted(set(all_yticks))))
ax_histy.yaxis.set_major_formatter(FuncFormatter(one_sig_fig))

# Custom legend
legend_elements = [
    Line2D([0], [0], color='k', marker='.', linestyle='none', label='Flight data'),
    Line2D([0], [0], color='k', marker='o', linestyle='none', label='Design point'),
    Line2D([0], [0], color='k', marker='s', linestyle='none', label='Average'),
    Line2D([0], [0], color='k', linestyle='-', label='PRD'),
    Patch(
        facecolor='none', edgecolor='k', hatch='//',
        label='IQR',
    ),
]
ax_scatter.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False, ncols=1)

# plt.savefig('prd_atr72_a220100_raw.svg', format='svg')

plt.tight_layout()
plt.show()

#%% TO BE DELETED

# ax_scatter.plot(pr_data[:,0], pr_data[:,1] * ac_pax_max / np.max(pr_data[:,1]) , color='black', lw=1.2, label='Payload-range')

# df_filtered = df[df["DEPARTURES_PERFORMED"] == 1].copy()
# # print('df_filtered.to_numpy() =', df_filtered.to_numpy())
# # print(df_filtered['PASSENGERS'])

# print("df_filtered['PAYLOAD'].to_numpy() =", df_filtered['PAYLOAD'].to_numpy())
# print()
# print("df_filtered['FREIGHT'].to_numpy() + df_filtered['MAIL'].to_numpy() =", max(df_filtered['FREIGHT'].to_numpy() + df_filtered['MAIL'].to_numpy()))
    
# sys.exit()
