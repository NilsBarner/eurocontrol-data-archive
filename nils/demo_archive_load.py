from archive import Archive

#%%

archive_path = "201912.zip"

#%%

archive = Archive(archive_path)

#%%

print(archive.flights_name)

#%%

print(archive.flights_df.head())

#%%

print(f"Nb of flights in archive: {archive.flights_df.shape[0]}")

#%%

print(archive.flight_points_filed_df.head())

#%%

print(archive.flight_points_actual_df.sample(5))

#%%

print(archive.flight_firs_filed_df.head())

#%%

print(archive.flight_firs_actual_df.head())

#%%

print(archive.flight_auas_filed_df.head())

#%%

print(archive.flight_auas_actual_df.head())

#%%

print(archive.routes_df.head())

#%%

print(archive.firs_df.head())




