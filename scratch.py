from data_manager import ged_consolidated_from_disk_or_api

df = ged_consolidated_from_disk_or_api()

df = df[df.dyad_new_id == 828]
df = df[df.where_prec.isin([4,6])]
print(df.month_id.unique())