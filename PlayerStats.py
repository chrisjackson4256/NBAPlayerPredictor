from bbRefScraper import buildTeamDFs, buildPlayerDFs
from playerCleaner import playerCleaner
from teamCleaner import teamCleaner
from dailyPlayer import dailyPlayer

buildTeamDFs()
buildPlayerDFs()

playerCleaner()
teamCleaner()

dailyPlayer()
