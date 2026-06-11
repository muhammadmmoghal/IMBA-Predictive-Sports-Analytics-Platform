# scraper.py
#
# PURPOSE:
#   Scrape player and game statistics from imbaonline.com and save them as
#   raw CSV files in data/raw/.
#
# PLANNED STEPS:
#   1. Identify the target URLs (player profiles, game logs, season stats).
#   2. Send HTTP requests (requests + BeautifulSoup) or drive a headless
#      browser (Selenium) if the site is JavaScript-rendered.
#   3. Parse each page and extract relevant fields:
#        - player name, team, game date, opponent
#        - kills, deaths, assists, and any other tracked stats
#   4. Collect all records into a Pandas DataFrame.
#   5. Save the raw DataFrame to data/raw/raw_stats.csv (no cleaning here).
#
# USAGE (once implemented):
#   python src/scraper.py
