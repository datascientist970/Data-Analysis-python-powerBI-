# Data Analysis Projects — Python & Power BI

> Data cleaning, exploration, time-series analysis and interactive dashboards built with Python, Streamlit, MySQL and Microsoft Power BI.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-Reports-F2C811?logo=powerbi&logoColor=black)
![MySQL](https://img.shields.io/badge/MySQL-Database-4479A1?logo=mysql&logoColor=white)

## Overview

This repository collects practical data-analysis work: Python notebooks for exploratory and time-series analysis, a Power BI report, and a complete Streamlit finance/insurance analytics dashboard backed by MySQL. Each project follows the same workflow — clean the raw data, explore it, visualise it, and turn it into business-ready insight.

## Key Features / Projects

| Project | What it does | Location |
|---------|--------------|----------|
| **Finance / Insurance Analytics Dashboard** | Streamlit dashboard that loads policy data from MySQL, with sidebar filters for Region, Location and Construction type, KPI cards (total income, mode, average, median, rating) and Plotly charts | [`Finance_Dashboard_with_streamlit_&_mysql/`](Finance_Dashboard_with_streamlit_&_mysql/) |
| **Power BI Finance Report** | Interactive Power BI report file ready to open in Power BI Desktop | [`finance.pbix`](finance.pbix) |
| **Tesla (TSLA) Stock EDA** | Exploratory data analysis of Tesla stock data in a Jupyter notebook | [`TSLA_EDA.ipynb`](TSLA_EDA.ipynb) |
| **Time Series Analysis** | Time-series exploration and analysis notebook | [`Time_series.ipynb`](Time_series.ipynb) |

Datasets included with the dashboard: `data.csv`, `insurance_dataset.csv`, and `data_synthetic.csv` (a large synthetic dataset).

## Tech Stack

- **Languages / Tools:** Python, SQL, Power BI Desktop
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit, streamlit-option-menu, numerize
- **Database:** MySQL (`mysql-connector-python`)
- **Environment:** Jupyter Notebook

## Project Structure

```text
Data-Analysis-python-powerBI-/
├── finance.pbix                              # Power BI report
├── TSLA_EDA.ipynb                            # Tesla stock EDA notebook
├── Time_series.ipynb                         # Time-series analysis notebook
└── Finance_Dashboard_with_streamlit_&_mysql/
    ├── dashboard.py                          # Streamlit dashboard app
    ├── query.py                              # MySQL connection & data fetch
    ├── data.csv / insurance_dataset.csv      # Sample datasets
    ├── data_synthetic.csv                    # Large synthetic dataset
    └── logo.png
```

## Getting Started

### Notebooks

```bash
git clone https://github.com/datascientist970/Data-Analysis-python-powerBI-.git
cd Data-Analysis-python-powerBI-
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook
```

### Streamlit Dashboard

1. Create a MySQL database named `mydatabase` with a `Data` table containing the insurance/policy columns used in `dashboard.py` (Policy, Expiry, Location, State, Region, Investment, Construction, BusinessType, Earthquake, Flood, Rating), or load one of the included CSV files into it.
2. Update the connection settings in `query.py` if your MySQL credentials differ from the local defaults.
3. Run the app:

```bash
cd Finance_Dashboard_with_streamlit_&_mysql
pip install streamlit pandas plotly mysql-connector-python streamlit-option-menu numerize
streamlit run dashboard.py
```

### Power BI Report

Open `finance.pbix` in Power BI Desktop to explore the interactive report.

## Usage

Use the sidebar filters in the Streamlit dashboard to slice the policy data by region, location and construction type — the KPI cards and charts update to the current selection.

## Author

**Muhammad Dawood** — AI & Data Science Engineer
GitHub: https://github.com/datascientist970 · Mardan, Pakistan
LinkedIn: https://www.linkedin.com/in/dawood406/

⭐ If you find this repository useful, consider giving it a star.
