from __future__ import division
import warnings
import math
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
pd.set_option('max_columns', 60)
pd.set_option('max_rows', 500)
warnings.filterwarnings("ignore")

def playerCleaner():
	# import player data
	player_df = pd.read_csv("Data/PlayerStats_2016.csv")
	player_type_df = pd.read_csv("Data/pcaPlayerClusters.csv")

	#print player_df.head()

	player_df = player_df.merge(player_type_df, on="Player")

	player_df.drop(['Rk', 'G', 'WinLoss'], axis=1, inplace=True)

	new_col_order = ['Player', 'Type', 'Date','Age', 'Tm', 'HomeAway', 'Opp','GS','MP', 
	                 'PTS','FG', 'FG%', 'FGA', 'FT', 'FT%','FTA','3P', '3P%',
	                 '3PA', 'AST','ORB','DRB','TRB','BLK','STL', 'TOV',
	                 '+/-','GmSc', 'PF']

	player_df = player_df[new_col_order]

	# convert date string to datetime...
	def Date_convert(y):
	    return pd.to_datetime(y)
	player_df["Date"] = player_df["Date"].apply(Date_convert)


	# replace Age with float number...
	def Age_replace(y):
	#    if type(y) == float:
	#        return
	    p = re.compile('\d+-')
	    years = float(p.search(y).group()[:-1])
	    days = round(float(y[-3:])/365,3)
	    return years + days  
	player_df["Age"] = player_df["Age"].apply(Age_replace)


	# replace Home game = 1, Away game = 0
	def HomeAway_replace(y):
	    if y == '@':
	        return 0
	    else:
	        return 1
	player_df["HomeAway"] = player_df["HomeAway"].apply(HomeAway_replace)

	# remove all NaN's
	player_df.fillna(0.0, inplace=True)


	# replace MP with float number...
	def MP_replace(y):
	    if type(y) == float:
	        return 0.0
	    p = re.compile('\d+:')
	    minutes = float(p.search(y).group()[:-1])
	    seconds = round(float(y[-2:])/60,2)
	    return minutes + seconds  
	player_df["MP"] = player_df["MP"].apply(MP_replace)



	# compute Double-Doubles and Triple-Doubles:
	def doubles(df):
	    rebs = df["TRB"].tolist()
	    asts = df["AST"].tolist()
	    pts = df["PTS"].tolist()

	    df["3X2X"] = 0
	    df["2X2X"] = 0
	    # compute double-double's and triple-double's...
	    for i in range(len(rebs)):
	        if rebs[i] >= 10.0 and asts[i] >= 10.0 and pts[i] >= 10.0:
	            df["3X2X"][i] = 1
	        elif rebs[i] >= 10.0 and asts[i] >= 10.0:
	            df["2X2X"][i] = 1
	        elif asts[i] >= 10.0 and pts[i] >= 10.0:
	            df["2X2X"][i] = 1
	        elif rebs[i] >= 10.0 and pts[i] >= 10.0:
	            df["2X2X"][i] = 1
	    return df
	player_df = doubles(player_df)



	# compute "Fantasy Points" using Draft Kings formula... 
	def FPs(df):
	    df["FP"] = df["PTS"] +  0.5 * df["3P"] + 1.25 * df["TRB"] + \
	            1.5 * df["AST"] + 2.0 * df["STL"] + 2.0 * df["BLK"] + \
	            (-0.5) * df["TOV"] + 1.5 * df["2X2X"] + 3.0 * df["3X2X"]
	    return df
	player_df = FPs(player_df)


	player_df = player_df.rename(columns = {'FG%':'FGpct', 'FT%': 'FTpct', '3P%': '3Ppct'})
	player_df.to_csv('Data/PlayerStats_clean_2016.csv')

	return












