import pandas as pd


genres = ['Classical', 'Jazz', 'Pop', 'Rock', 'Country']
for genre in genres:
	load_df = pd.read_pickle('./pickles/'+ genre + '.pickle')
	new_df = load_df.dropna(how = "any")
	new_df.to_pickle('./pickles/D_' + genre + '.pickle')