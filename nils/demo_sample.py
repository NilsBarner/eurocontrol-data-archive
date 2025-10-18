from io import BytesIO
import requests
from zipfile import ZipFile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from archive import Sample, get_trajs, get_airspaces, get_all_airports

#%%

sample_zip_url = "https://www.eurocontrol.int/archive_download/all/node/12418"

#%%

def get_zip(file_url):
    url = requests.get(file_url)
    zipfile = ZipFile(BytesIO(url.content))
    zip_names = zipfile.namelist()
    if len(zip_names) == 1:
        file_name = zip_names.pop()
        extracted_file = zipfile.open(file_name)
        return extracted_file
    return [zipfile.open(file_name) for file_name in zip_names]

#%%

sample_zip = get_zip(sample_zip_url)  # or use directly zip file that you saved locally
sample = Sample(sample_zip)

#%%

sample.flights_df.head()

#%%

airports = get_all_airports(sample.flights_df)
airports.head()

#%%

sample.flight_points_actual_df.head()

#%%

trajs = get_trajs(sample.flight_points_actual_df)
trajs.head()

#%%

trajs.loc[184410592].values[0]

#%%

sample.firs_df.head()

#%%

airspaces = get_airspaces(sample.firs_df)
airspaces.head(7)

#%%

airspaces.loc["LFFFUIR"].values[0]  # France UIR

#%%

figsize = (20, 20)  # (18, 18)
first_cmap = "Spectral"
flights_cmap = "cool"
airport_icon_path = "http://maps.google.com/mapfiles/kml/shapes/airports.png"
# airport_icon = plt.imread(airport_icon_path)
# =============================================================================
import urllib.request
from PIL import Image
import numpy as np

airport_icon = np.array(Image.open(urllib.request.urlopen(airport_icon_path)))
# =============================================================================
airport_icon_zoom = 0.4
airport_name_vertical_shift = 60000
airport_name_fontsize = 12
airspace_alpha = 0.3
ctx_zoom = 6
file_name = "sample.png"

#%%

airspaces_3857 = airspaces.to_crs(epsg=3857)  # web mercator
trajs_3857 = trajs.to_crs(epsg=3857)
airspaces_3857.reset_index(inplace=True)
trajs_3857.reset_index(inplace=True)
airports_3857 = airports.to_crs(epsg=3857)

#%%

f, ax = plt.subplots(figsize=figsize)
airspaces_3857.plot(
    ax=ax,
    edgecolor="black",
    alpha=airspace_alpha,
    cmap=first_cmap,
    legend=True,
    column="index",
)
leg1 = ax.get_legend()
trajs_3857.plot(
    ax=ax,
    column="ECTRL ID",
    categorical=True,
    legend=True,
    cmap=flights_cmap,
    legend_kwds={"loc": "upper left"},
)
airports_3857.plot(ax=ax)

for i, r in airports_3857.iterrows():
    posx, posy = (r[1].x, r[1].y)
    ab = AnnotationBbox(
        OffsetImage(airport_icon, zoom=airport_icon_zoom), (posx, posy), frameon=False,
    )
    ax.add_artist(ab)
    plt.text(
        posx,
        posy - airport_name_vertical_shift,
        r[0],
        horizontalalignment="center",
        fontsize=airport_name_fontsize,
    )

ctx.add_basemap(ax, zoom=ctx_zoom)
ax.set_axis_off()
ax.add_artist(leg1)
plt.savefig(file_name, bbox_inches="tight", pad_inches=-0.1)













