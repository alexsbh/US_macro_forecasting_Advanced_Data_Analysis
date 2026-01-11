# US_macro_forecasting_Advanced_Data_Analysis
## How to run the project

### Folder structure
All scripts write to the `Data/` folder located at the project root.
Paths are automatically detected and are platform-independent.

### TA/grading instructions
The repository already includes the raw datasets (FRED + Refinitiv exports).

**For grading, start from Script 04:**
1. `python 04a_pca_factors.py`
2. `python 04b_baselines_var_linear.py`
3. `python 04c_xgb.py`
4. `python 04d_rnn.py`
5. `python 04e_merge_and_report.py`
6. `python 04f_plot_forecasts.py`

**Or: you can directly run master_run.py, which will effectively do the same thing**

**You will find all the results (charts and tables) in the two branches (horizon 1 month, and horizon 12 months)**

Scripts `01_data_scraper.py` and `02_build_panel.py` are included for full
reproducibility, but are not required for grading and can be skipped.
