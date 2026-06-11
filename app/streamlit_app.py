# streamlit_app.py
#
# PURPOSE:
#   Interactive web dashboard for exploring player stats, EDA visualizations,
#   and next-game performance predictions.
#
# PLANNED SECTIONS:
#   1. Sidebar controls:
#        - Select league / season / team / player
#        - Choose prediction target stat
#   2. Player Profile tab:
#        - Season stat summary table
#        - Rolling performance trend line chart
#   3. Exploratory Data Analysis tab:
#        - Stat distribution histograms
#        - Correlation heatmap
#        - Head-to-head comparison tool
#   4. Predictions tab:
#        - Next-game predicted score for selected player(s)
#        - Feature importance bar chart
#        - Model accuracy metrics (MAE, RMSE, R²)
#   5. Raw Data tab:
#        - Filterable / sortable data table (st.dataframe)
#        - CSV download button
#
# USAGE (once implemented):
#   streamlit run app/streamlit_app.py
