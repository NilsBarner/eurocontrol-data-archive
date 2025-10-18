import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from archive import Archive, get_trajs, get_airspaces

#%%

archive_path = "201912.zip"
new_archive_filename = "201912_filtered.zip"

#%%

archive = Archive(archive_path)

#%%

archive.datetime_filtering("2018-12-24 08:00:00", "2018-12-24 10:00:00")
archive.clip(["EDUUUIR", "EDVVUIR"])  # German upper airspace
archive.distance_filtering(100, 810)  # short-haul