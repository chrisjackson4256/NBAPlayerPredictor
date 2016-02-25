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

def teamCleaner():

	# import player data
	team_df = pd.read_csv("Data/TeamStats_2016.csv")
	team_df = team_df.rename(columns = {'Unnamed: 0':'Date'})
	team_df.drop(['Rk', 'G','Empty', 'Empty.1'], axis=1, inplace=True)

	new_col_order = ['Team','Date', 'HomeAway', 'Opp', 'WinLoss', 'TmScore', 'OppScore', 'ORtg', 'DRtg', 'Pace', 'FTr', 
	                 '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'O_eFG%', 'O_TOV%', 'O_ORB%', 'O_FTFTA', 
	                 'D_eFG%', 'D_TOV%', 'D_ORB%', 'D_FTFTA']
	team_df = team_df[new_col_order]


	# replace Home game = 1, Away game = 0
	def HomeAway_replace(y):
	    if y == '@':
	        return 0
	    else:
	        return 1
	team_df["HomeAway"] = team_df["HomeAway"].apply(HomeAway_replace)

	# remove all NaN's
	team_df.fillna(0.0, inplace=True)


	# replace Win = 1, Loss = 0
	def WinLoss_replace(y):
	    if y == 'L':
	        return 0
	    elif y == 'W':
	        return 1
	team_df["WinLoss"] = team_df["WinLoss"].apply(WinLoss_replace)


	# convert date string to datetime...
	def Date_convert(y):
	    return pd.to_datetime(y)
	team_df["Date"] = team_df["Date"].apply(Date_convert)


	# export to new csv file
	team_df.to_csv("Data/TeamStats_clean_2016.csv")

	return












