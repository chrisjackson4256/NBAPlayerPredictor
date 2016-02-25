import io
import time
import json
import string
import pandas as pd
import logging
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

pd.options.display.max_columns = 999

def getCurrentPlayerNamesAndURLS(supressOutput=True):

    names = []

    for letter in string.ascii_lowercase:

        letter_page = getSoupFromURL('http://www.basketball-reference.com/players/%s/' % (letter), supressOutput)

        # we know that all the currently active players have <strong> tags, so we'll limit our names to those
        try: 
            current_names = letter_page.findAll('strong')
        except AttributeError:
            continue
        for n in current_names:
            name_data = n.children.next()
            names.append((name_data.contents[0], 'http://www.basketball-reference.com' + name_data.attrs['href']))
        time.sleep(1) # sleeping to be kind for requests

    return dict(names)


def getSoupFromURL(url, supressOutput=True):
    
    """
    This function grabs the url and returns the BeautifulSoup object
    """
    if not supressOutput:
        print url

    try:  
        r = requests.get(url)
    except:
        return None

    return BeautifulSoup(r.text)
    
    
    
def soupTableToDF(table_soup, header):
    """
    Parses the HTML/Soup table for the gamelog stats.
    Returns a pandas DataFrame
    """
    if not table_soup:
        return None
    else:
        rows = table_soup[0].findAll('tr')[1:]  # all rows but the header

        # remove blank rows
        rows = [r for r in rows if len(r.findAll('td')) > 0]

        parsed_table = [[col.getText() for col in row.findAll('td')] for row in rows] # build 2d list of table values
        return pd.io.parsers.TextParser(parsed_table, names=header, index_col=2, parse_dates=True).get_chunk()



def buildTeamDFs(supressOutput=True):

   # team and year lists
    team_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 
                'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers', 
                'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 
                'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 
                'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 
                'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves', 
                'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder',
                'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns', 
                'Portland Trailblazers', 'Sacramento Kings', 'San Antonio Spurs',
                'Toronto Raptors', 'Utah Jazz', 'Washington Wizards']
                
    team_abbrevs = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 
                    'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA',
                    'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO',
                    'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'] 
                
                    
    years = range(2016, 2017)
    
    # build a dictionary with every team (key) and their game log links (value list)
    teamsGameLogLinks = {}
    for i in range(len(team_names)):
        teamsGameLogLinks[team_names[i]] = []
        for year in years:
            # catch team name changes
            if year <= 2012:
                team_abbrevs[2] = 'NJN'
            else:
                team_abbrevs[2] = 'BRK'
            if year <= 2015:
                team_abbrevs[3] = 'CHA'
            else:
                team_abbrevs[3] = 'CHO'
            if year <= 2012:
                team_abbrevs[18] = 'NOH'
            else:
                team_abbrevs[18] = 'NOP'
            
            teamsGameLogLinks[team_names[i]].append("http://www.basketball-reference.com" + '/teams/' + team_abbrevs[i] + '/' + str(year) + '/gamelog/')

    # build dataframes for each team
    teamDFs = {}
    totalDF = pd.DataFrame()
    for team in team_names:
        teamDFList = []
        for url in teamsGameLogLinks[team]:
            
            glsoup = getSoupFromURL(url)

            reg_season_table = glsoup.findAll('table', attrs={'id': 'tgl_advanced'})  # id for reg season table
            playoff_table = glsoup.findAll('table', attrs={'id': 'tgl_advanced_playoffs'}) # id for playoff table
                          
            header = [u'Rk', u'G', u'HomeAway', u'Opp', u'WinLoss', u'TmScore', u'OppScore',
                      u'ORtg', u'DRtg', u'Pace', u'FTr', u'3PAr', u'TS%', u'TRB%', u'AST%', u'STL%', u'BLK%', u'Empty', 
                      u'O_eFG%',u'O_TOV%', u'O_ORB%', u'O_FTFTA', u'Empty', u'D_eFG%', u'D_TOV%', u'D_ORB%', u'D_FTFTA']
    

            reg = soupTableToDF(reg_season_table, header)
            
                                    
            playoff = soupTableToDF(playoff_table, header)

            if reg is None:
                seasonDF = playoff
            elif playoff is None:
                seasonDF =  reg
            else:
                seasonDF = pd.concat([reg, playoff])
            teamDFList.append(seasonDF)
        teamDF = pd.concat(teamDFList)
        teamDF["Team"] = team_abbrevs[team_names.index(team)]
        teamDFs[team] = teamDF
        totalDF = totalDF.append(teamDF)  
        print team
        
    # write DFs to csv files 
#    for i in range(len(team_names)):
#        file_name = team_abbrevs[i] + '_' + str(years[0]) + '-' + str(years[-1]) + '.csv'
#        teamDFs[team_names[i]].to_csv(file_name, encoding='utf-8')
    
    totalDF.to_csv('Data/TeamStats_2016.csv', encoding='utf-8')
    
    return 
    
def buildPlayerDFs(supressOutput=True):
    
    playerNamesAndURLs = getCurrentPlayerNamesAndURLS()
        
    player_names = playerNamesAndURLs.keys()
    
    # to build gamelog url's... need to chop off ".html" and add "/gamelog/year/"
    years = range(2016, 2017)
    
    playersGameLogLinks = {}
    for i in range(len(player_names)):
        playersGameLogLinks[player_names[i]] = []
        for year in years:
            gameLogURL = playerNamesAndURLs[player_names[i]][:-5] + '/gamelog/' + str(year) + '/'
            playersGameLogLinks[player_names[i]].append(gameLogURL)
        
        
    # build dataframes for each player
    playerDFs = {}
    totalDF = pd.DataFrame()
    for name in player_names:
        playerDFList = []
        for url in playersGameLogLinks[name]:
            
            glsoup = getSoupFromURL(url)
            
            try:
                reg_season_table = glsoup.findAll('table', attrs={'id': 'pgl_basic'})  # id for reg season table
                playoff_table = glsoup.findAll('table', attrs={'id': 'pgl_basic_playoffs'}) # id for playoff table
            except AttributeError:
                continue
                
            # parse the table header.  we'll use this for the creation of the DataFrame
            header = []
            try:
                for th in reg_season_table[0].findAll('th'):
                    if not th.getText() in header:
                        header.append(th.getText())
            except IndexError:
                continue

            # add in headers for home/away and w/l columns. a must to get the DataFrame to parse correctly
            header[5] = u'HomeAway'
            header.insert(7, u'WinLoss')

            try:
                reg = soupTableToDF(reg_season_table, header)
                playoff = soupTableToDF(playoff_table, header)
            except ValueError:
                continue
                
            if reg is None:
                seasonDF = playoff
            elif playoff is None:
                seasonDF = reg
            else:
                seasonDF = pd.concat([reg, playoff])
            playerDFList.append(seasonDF)
        try:
            playerDF = pd.concat(playerDFList)
        except ValueError:
            continue
        playerDF["Player"] = name
        playerDFs[name] = playerDF
        print name
        totalDF = totalDF.append(playerDF)
    # write DFs to csv files
#    for i in range(len(player_names)):
#        p_name = player_names[i].encode('ascii')
#        name_as_list = p_name.split(' ')
#        player_name = ('_').join(name_as_list)
#        file_name = player_name + '_' + str(years[0]) + '-' + str(years[-1]) + '.csv'
#        try:
#            playerDFs[player_names[i]].to_csv('/Users/chris/Desktop/InsightProject/Pick2Click/Data/Players/' + file_name, encoding='utf-8')
#        except KeyError:
#            print player_names[i]       
    totalDF.to_csv('Data/PlayerStats_2016.csv', encoding='utf-8')
    
    return
