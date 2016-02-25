from __future__ import division
import warnings
import math
import re
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from sklearn.cross_validation import cross_val_score, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
pd.set_option('max_columns', 60)
pd.set_option('max_rows', 500)
warnings.filterwarnings("ignore")


def dailyPlayer():
	# import the nba schedule
	schedule = pd.read_csv("Data/nbaSchedule2015-16.csv")


	# extract lists of the home and away teams for a particular day

	today = str(datetime.date.today())

	def trim_time(y):
	    return y[:6]

	#schedule["Datetime"] = schedule["Datetime"].apply(trim_time)

	todays_games = schedule[schedule["Datetime"] == today]

	home_teams = todays_games["Home"].tolist()
	away_teams = todays_games["Away"].tolist()

	team_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 
	            'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers', 
	            'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 
	            'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 
	            'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 
	            'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves', 
	            'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder',
	            'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns', 
	            'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs',
	            'Toronto Raptors', 'Utah Jazz', 'Washington Wizards']

	team_abbrevs = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 
	                'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
	                'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO',
	                'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

	home_team = []
	for team in home_teams:
	    if team in team_names:
	        home_team.append(team_abbrevs[team_names.index(team)])
	        
	away_team = []
	for team in away_teams:
	    if team in team_names:
	        away_team.append(team_abbrevs[team_names.index(team)])
	        
	matchups = zip(home_team, away_team)
	        
	today_teams = home_team + away_team


	# import all player data (2015-16) WITH TYPES INCLUDED:
	player_2016_df = pd.read_csv('Data/PlayerStats_clean_2016.csv')
	player_2016_df.drop('Unnamed: 0', axis=1, inplace = True)
	player_2016_df = player_2016_df.rename(columns = {'Tm':'Team'})


	# import all team data (2015-16):
	team_2016_df = pd.read_csv('Data/TeamStats_clean_2016.csv')
	team_2016_df.drop('Unnamed: 0', axis=1, inplace = True)

	# Merge with the team's dataframe
	player_team_2016_df = pd.merge(player_2016_df, team_2016_df, on=["Date","Opp", "Team", "HomeAway"])

	# pick out the players that are playing that night by checking if Team is in Home/Away lists
	today_players = pd.DataFrame()
	for team in today_teams:
	        today_players = today_players.append(
	            player_team_2016_df[player_team_2016_df["Team"] == team])

	player_list = today_players["Player"].tolist()
	team_list = today_players["Team"].tolist()

	player_team_list = set(zip(player_list, team_list))

	# make a list (of tuples) of today's players and their opponents to feed into random forest
	player_opp_list = []
	for player_team in player_team_list:
	    if player_team[1] in home_team:
	        opp_team = away_team[home_team.index(player_team[1])]
	        player_opp_list.append((player_team[0], opp_team))
	    elif player_team[1] in away_team:
	        opp_team = home_team[away_team.index(player_team[1])]
	        player_opp_list.append((player_team[0], opp_team))


	# import all player data (2010 - present) WITH TYPES INCLUDED:
	player_2011_15_df = pd.read_csv('Data/PlayerStats_clean_2011-15.csv')
	player_2011_15_df.drop('Unnamed: 0', axis=1, inplace = True)
	player_2011_16_df = pd.concat([player_2011_15_df, player_2016_df])
	player_2011_16_df.drop(['Team'], axis=1, inplace = True)
	player_2011_16_df = player_2011_16_df.rename(columns = {'Tm':'Team'})

	# import all team data (2010 - present):
	team_2011_15_df = pd.read_csv('Data/TeamStats_clean_2011-15.csv')
	team_2011_16_df = pd.concat([team_2011_15_df, team_2016_df])
	team_2011_16_df.drop('Unnamed: 0', axis=1, inplace = True)


	today_type_df = pd.DataFrame()
	type_prediction_accuracy = []
	predFP = {}

	for player, opp in player_opp_list:
	    # the players data frame:
	    player_df = player_2011_16_df[player_2011_16_df["Player"] == player]

	    # the player types data frame:
	    players_type = player_2011_16_df[player_2011_16_df["Player"] == player]["Type"].unique()[0]
	    pl_type_df = player_2011_16_df[(player_2011_16_df["Type"] == players_type) & \
	                                   (player_2011_16_df["Player"] != player)]
	    
	    pl_type_team_df = pd.merge(pl_type_df, team_2011_16_df, on=["Date", "Opp", "Team", "HomeAway"])
	    
	#    player_team_opp_df = player_team_df[player_team_df["Opp"] == opp]
	#    player_team_opp_df = player_team_opp_df[player_team_opp_df["GS"] == '1']
	    
	    pl_type_team_opp_df = pl_type_team_df[pl_type_team_df["Opp"] == opp]
	    pl_type_team_opp_df = pl_type_team_opp_df[pl_type_team_opp_df["GS"] == '1']

	    type_dict = {"Type": players_type, "Opp": opp}

	    # drop all of the stats that go into FP
	    pl_type_team_df.drop(['Player', 'Date', 'Team', 'HomeAway', 'Opp', 'GS', 'PTS','FG', 
	                'FGpct', 'FT', 'FTpct', '3P', '3Ppct', 'AST', 'ORB','DRB', 'TRB', 
	                'BLK', 'STL', 'TOV', '+/-', 'GmSc','3X2X', '2X2X', 'WinLoss', 
	                'TmScore', 'OppScore', 'Age', 'PF'], 
	                axis=1, inplace=True)
	                
	    pl_type_team_opp_df.drop(['Player', 'Opp', 'Date', 'Team', 'HomeAway', 'GS', 'PTS','FG', 
	                'FGpct', 'FT', 'FTpct', '3P', '3Ppct', 'AST', 'ORB','DRB', 'TRB', 
	                'BLK', 'STL', 'TOV', '+/-', 'GmSc','3X2X', '2X2X', 'WinLoss', 
	                'TmScore', 'OppScore', 'Age', 'PF'], 
	                axis=1, inplace=True)
	    
	    
	    # remove all NaN's
	    pl_type_team_opp_df.fillna(0.0, inplace=True)
	    
	    #
	    #  player_type-opponent 2010-2016
	    #

	    # split features from labels
	    train_labels = pl_type_team_opp_df["FP"]
	    train_labels = train_labels.astype(float)
	    pl_type_team_opp_df.drop(["FP"], axis=1, inplace=True)
	    train_features = pl_type_team_opp_df
	        
	    if train_features.empty:
	        continue
	        
	    predictors = pl_type_team_opp_df.columns.values
	    avg_vals_preds = pl_type_team_opp_df.mean().tolist()

	    n_folds = 10
	    pred_values = []
	    r2_values = []
	    for i in range(n_folds):
	        X_train, X_test, y_train, y_test =  \
	            train_test_split(train_features, train_labels, test_size=0.4, random_state=42)

	        reg = LinearRegression()
	        reg.fit(X_train, y_train)
	        scores_type = reg.score(X_test, y_test)
	        pred_type = reg.predict(avg_vals_preds)

	        pred_values.append(pred_type)
	        r2_values.append(scores_type)
	    if np.mean(r2_values) > 0.5:
	        type_prediction_accuracy.append(np.mean(r2_values))
	    type_dict["Pred. FP"] = round(np.mean(pred_type), 1)
	    
	    today_type_df = today_type_df.append(type_dict, ignore_index=True)
	    today_type_df = today_type_df[["Type", "Opp","Pred. FP"]]
	    today_type_df = today_type_df.drop_duplicates()
	'''    
	plt.plot(type_prediction_accuracy, 'ro')
	plt.title("Linear Regression R^2 (2/3/2016)")
	plt.xlabel("Player ID")
	plt.ylabel("R^2")
	plt.axis([0, 100, 0, 1])
	plt.show()
	'''
	avg_today_type = today_type_df.groupby(["Type", "Opp"])["Pred. FP"].mean().to_frame()


	today_player_df = pd.DataFrame()
	player_prediction_accuracy = []
	predFP = {}

	for (player, opp) in player_opp_list:
	    
	    # the players data frame:
	    player_df = player_2011_16_df[player_2011_16_df["Player"] == player]

	    # the player types data frame:
	    players_type = player_2011_16_df[player_2011_16_df["Player"] == player]["Type"].unique()[0]
	    
	    player_team_df = pd.merge(player_df, team_2011_16_df, on=["Date", "Opp", "Team", "HomeAway"])
	    
	    avgMP = round(player_team_df["MP"].mean(),1)
	    avgFP = round(player_team_df["FP"].mean(),1)
	    if np.isnan(avgMP):
	        continue
	    if avgMP < 20.0:
	        continue
	#    if not np.isnan(avgMP):
	    player_dict = {"Player": player, "Type": players_type, "Opp": opp, "Avg. MP": avgMP, "Avg. FP": avgFP}
	        
	    # drop all of the stats that go into FP
	    player_team_df.drop(['Player', 'Date', 'Team', 'HomeAway', 'Opp', 'GS', 'PTS','FG', 
	                'FGpct', 'FT', 'FTpct', '3P', '3Ppct', 'AST', 'ORB','DRB', 'TRB', 
	                'BLK', 'STL', 'TOV', '+/-', 'GmSc','3X2X', '2X2X', 'WinLoss', 
	                'TmScore', 'OppScore', 'Age', 'PF'], 
	                axis=1, inplace=True)
	    
	    # remove all NaN's
	    player_team_df.fillna(0.0, inplace=True)
	    
	    #
	    #  player-team 2010-2016
	    #
	    
	    # split features from labels
	    train_labels = player_team_df["FP"]
	    train_labels = train_labels.astype(float)
	    player_team_df.drop(["FP"], axis=1, inplace=True)
	    train_features = player_team_df
	        
	    predictors = player_team_df.columns.values
	    avg_vals_preds = player_team_df.mean().tolist()
	    
	    n_folds = 10
	    pred_values = []
	    r2_values = []

	    for i in range(n_folds):

	        X_train, X_test, y_train, y_test =  \
	            train_test_split(train_features, train_labels, test_size=0.4, random_state=42)
	    
	        if X_train.empty:
	            continue
	            
	        reg = LinearRegression()
	        reg.fit(X_train, y_train)
	        scores_player = reg.score(X_test, y_test)
	        pred_player = reg.predict(avg_vals_preds)
	            
	        pred_values.append(pred_player)
	        r2_values.append(scores_player)
	        
	    player_prediction_accuracy.append(np.mean(r2_values))
	    
	    type_prediction = round(avg_today_type.loc[(avg_today_type.index.get_level_values('Type') == players_type) & 
	                        (avg_today_type.index.get_level_values('Opp') == opp)]["Pred. FP"].values[0], 1)
	        
	    player_prediction = round(np.mean(pred_player), 1)
	    
	    weighted_prediction = (3 * player_prediction + type_prediction) / 4
	    
	    player_dict["Pred. FP"] = round(weighted_prediction, 1)
	    
	    today_player_df = today_player_df.append(player_dict, ignore_index=True)
	    today_player_df = today_player_df[["Player", "Type", "Opp", "Avg. MP", "Avg. FP", "Pred. FP"]]
	    today_player_df = today_player_df.drop_duplicates()


	today_player_df = today_player_df[np.isfinite(today_player_df['Pred. FP'])]
	today_player_df["Diff. FP"] = today_player_df["Pred. FP"] - today_player_df["Avg. FP"]

	today_player_df = today_player_df[["Player", "Avg. MP", "Avg. FP", "Pred. FP", "Diff. FP"]]

	today_player_df = today_player_df.sort("Diff. FP", ascending=False)
	#print today_player_df

	top_performers = today_player_df.head(10)
	#print top_performers

	bottom_performers = today_player_df.tail(10).sort("Diff. FP")
	#print bottom_performers

	today_player_df.to_csv("Data/players" + today + ".csv", encoding="utf-8")

	return

