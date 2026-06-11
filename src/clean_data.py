# clean_data.py
#
# PURPOSE:
#   Load the raw scraped data from data/raw/ and produce a clean, normalized
#   player-game dataset saved to data/processed/clean_stats.csv.
#
# PLANNED STEPS:
#   1. Load data/raw/raw_stats.csv into a Pandas DataFrame.
#   2. Standardize column names (snake_case, no special characters).
#   3. Parse and normalize data types:
#        - dates → datetime
#        - numeric stat columns → float/int
#        - categorical columns → category dtype
#   4. Handle missing values (drop, fill, or flag as appropriate).
#   5. Remove duplicate rows.
#   6. Validate data ranges (e.g., no negative kill counts).
#   7. Save cleaned DataFrame to data/processed/clean_stats.csv.
#
# USAGE (once implemented):
#   python src/clean_data.py
