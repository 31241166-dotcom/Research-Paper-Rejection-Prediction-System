# Simple Paper Rejection Predictor

A beginner-friendly Python dashboard for a research-methodology demonstration.

## Contents
- `app.py`: Streamlit dashboard and Decision Tree model
- `data/paper_dataset.csv`: 300 synthetic sample records
- `database/predictions.db`: SQLite database (created/updated by app)
- `requirements.txt`: Python dependencies

## Run in VS Code (Windows)
1. Extract the ZIP and open this folder in VS Code.
2. Open Terminal in VS Code.
3. Install dependencies:
   ```powershell
   py -m pip install -r requirements.txt
   ```
4. Start the dashboard:
   ```powershell
   py -m streamlit run app.py
   ```
5. Open the local URL shown in the terminal, usually `http://localhost:8501`.

The SQLite database is included and the table is initialized automatically.

## Research note
The CSV contains synthetic data and simulated outcomes, not real journal decisions. Treat model outputs as demonstrations only. Use a real, documented dataset and external validation for research claims.
