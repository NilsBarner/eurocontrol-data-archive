import os
import sys
import pandas as pd
import contextily as ctx
import matplotlib.pyplot as plt
from archive import (
    Archive,
    Sample,
    get_trajs,
    get_airspaces,
    get_all_airports,
)

#%% Load archive 

# archive_path = "201912.zip"  # unfiltered
archive_path = "201912_A20N.zip"  # filtered
archive = Archive(archive_path)

#%% Filter archive for aircraft type

# ICAO = "A20N"
# df = archive.flights_df.query("`AC Type` == @ICAO")
df = archive.flights_df

#%% Create a filtered archive .zip file for the aircraft type of interest

# # Get the list of ECTRL IDs for the chosen aircraft type
# ICAO = "A20N"
# ids = archive.flights_df.loc[archive.flights_df["AC Type"] == ICAO, "ECTRL ID"].unique().tolist()
# print(f"Found {len(ids)} flights for AC Type {ICAO}")
# archive.update_from_ids(ids)      # keeps only flights matching ids
# archive.to_archive("201912_A20N.zip")  # writes a new zip with the filtered data

# # Remove the temporary gz files created by to_archive() automatically
# tmp_files = [
#     archive.flights_name,
#     archive.flight_points_filed_name,
#     archive.flight_points_actual_name,
#     archive.flight_firs_filed_name,
#     archive.flight_firs_actual_name,
#     archive.flight_auas_filed_name,
#     archive.flight_auas_actual_name,
#     archive.routes_name,
#     archive.firs_name,
# ]
# for fn in tmp_files:
#     if os.path.exists(fn):
#         os.remove(fn)

#%% Identify and plot 20 most popular routes for that aircraft

# Count how often each ADEP–ADES pair occurs
route_counts = (
    df.groupby(["ADEP", "ADES"])
    .size()
    .sort_values(ascending=False)
    .head(20)
)

route_counts.plot(
    kind="barh",
    figsize=(15, 8),
    color="skyblue",
    grid=True,
    title="20 Busiest Routes (ADEP → ADES)",
)

plt.xlabel("Number of Flights")
plt.ylabel("Route (ADEP → ADES)")
plt.gca().invert_yaxis()  # so the busiest route appears at the top
plt.show()

#%% Extract trajectories of those routes

# # All routes in archive
# trajs = get_trajs(archive.flight_points_actual_df)

# # All flights along those routes

# # build a set of the top-20 route (ADEP, ADES) tuples
# top_routes = set(route_counts.index.tolist())

# # add a temporary route column (tuple) to flights_df
# df['route'] = list(zip(df['ADEP'], df['ADES']))

# # get ECTRL IDs for flights on those top routes
# selected_ids = df.loc[df['route'].isin(top_routes), 'ECTRL ID'].unique()

# # filter the points DataFrame to those flights and extract trajectories
# trajs = get_trajs(archive.flight_points_actual_df[archive.flight_points_actual_df['ECTRL ID'].isin(selected_ids)])

# # optional: remove the temporary column if you don't want to keep it
# df.drop(columns=['route'], inplace=True)

# One flight along those routes

# build set of top-20 routes (tuple)
top_routes = set(route_counts.index.tolist())

# ensure df has route tuples
df['route'] = list(zip(df['ADEP'], df['ADES']))

# pick one ECTRL ID per route (the first)
one_per_route_ids = (
    df.loc[df['route'].isin(top_routes)]
      .groupby('route')['ECTRL ID']
      .first()
      .tolist()
)

trajs = get_trajs(archive.flight_points_actual_df[archive.flight_points_actual_df['ECTRL ID'].isin(one_per_route_ids)])

# trajs.loc[236565661].values[0]  # enter in console to plot trajectory

#%% Plot top-20 trajectories

figsize = (20, 20)  # (18, 18)
first_cmap = "Spectral"
flights_cmap = "cool"
ctx_zoom = 6
file_name = "sample.png"

# See https://epsg.io/3857
trajs_3857 = trajs.to_crs(epsg=3857)
trajs_3857.reset_index(inplace=True)

f, ax = plt.subplots(figsize=figsize)
leg1 = ax.get_legend()
trajs_3857.plot(
    ax=ax,
    column="ECTRL ID",
    categorical=True,
    legend=True,
    cmap=flights_cmap,
    legend_kwds={"loc": "upper left"},
)
ctx.add_basemap(ax, zoom=ctx_zoom)
ax.set_axis_off()






