import csv
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
data = pd.read_csv("/data/MAESTRO/maestro-v2.0.0/maestro-v2.0.0_cleaned.csv")
data = data.drop(["split", "year", "audio_filename", "duration"], axis=1)
# print(data)
# print(data.columns)
# ['canonical_composer', 'canonical_title', 'split', 'year','midi_filename', 'audio_filename', 'duration']
composers = dict(data["canonical_composer"].value_counts())
df_total = pd.DataFrame()
for key, value in composers.items():
    if value > 25:
        df_temp = data[data["canonical_composer"] == key]  # new df
        # print(df_temp)
        df_total = df_total.append(df_temp)
        # tracks = dict(df_temp['canonical_title'].value_counts())
        # print(tracks)
print(df_total)
# iter_list = list(zip(df_total['canonical_composer'], df_total['canonical_title'], df_total['midi_filename']))
