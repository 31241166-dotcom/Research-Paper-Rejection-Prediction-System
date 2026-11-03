from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "paper_dataset.csv"
DB_PATH = ROOT / "database" / "predictions.db"

FEATURES = ["sentence_cohesion", "paragraph_cohesion", "research_gap_clarity", "objective_clarity"]

st.set_page_config(page_title="Simple Paper Rejection Predictor", page_icon="📄", layout="wide")

def setup_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                paper_name TEXT,
                rejection_probability REAL,
                predicted_result TEXT
            )
        """)

def get_model():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["rejected"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=7)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, df

setup_db()
model, accuracy, data = get_model()

st.title("📄 Simple Paper Rejection Predictor")
st.write("A beginner-friendly project to explore whether introduction cohesion metrics relate to paper rejection.")
st.warning("Demo only: this project uses synthetic sample data and simulated outcomes. It cannot reliably predict real journal decisions.")

page = st.sidebar.radio("Choose Page", ["Dashboard", "Predict", "Dataset", "History", "About"])

if page == "Dashboard":
    c1, c2, c3 = st.columns(3)
    c1.metric("Total sample papers", len(data))
    c2.metric("Sample rejection rate", f"{data['rejected'].mean()*100:.1f}%")
    c3.metric("Demo model accuracy", f"{accuracy*100:.1f}%")
    st.subheader("Paper outcomes")
    counts = data["outcome"].value_counts().reindex(["Accepted", "Rejected"], fill_value=0)
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Number of papers")
    st.pyplot(fig)
    st.subheader("Average cohesion scores")
    averages = data.groupby("outcome")[FEATURES].mean().T
    fig2, ax2 = plt.subplots()
    averages.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Average score (0–1)")
    ax2.tick_params(axis="x", rotation=25)
    st.pyplot(fig2)

elif page == "Predict":
    st.subheader("Enter paper introduction scores")
    paper_name = st.text_input("Paper name", "My Research Paper")
    st.caption("Use scores from 0 to 1. Higher scores mean stronger cohesion/clarity.")
    vals = {}
    labels = {
        "sentence_cohesion": "Sentence cohesion",
        "paragraph_cohesion": "Paragraph cohesion",
        "research_gap_clarity": "Research-gap clarity",
        "objective_clarity": "Research-objective clarity"
    }
    cols = st.columns(2)
    defaults = [0.6, 0.6, 0.6, 0.6]
    for i, feature in enumerate(FEATURES):
        with cols[i % 2]:
            vals[feature] = st.slider(labels[feature], 0.0, 1.0, defaults[i], 0.05)
    if st.button("Predict", type="primary"):
        X_new = pd.DataFrame([vals], columns=FEATURES)
        p = float(model.predict_proba(X_new)[0][1])
        result = "Rejected" if p >= 0.5 else "Accepted"
        a, b = st.columns(2)
        a.metric("Estimated rejection probability", f"{p*100:.1f}%")
        b.metric("Demo prediction", result)
        if st.button("Save result to database"):
            with sqlite3.connect(DB_PATH) as con:
                con.execute(
                    "INSERT INTO predictions (paper_name, rejection_probability, predicted_result) VALUES (?, ?, ?)",
                    (paper_name, p, result)
                )
            st.success("Saved to database. Open the History page to view it.")
        st.info("This is a demonstration result based on synthetic training data, not a real editorial forecast.")

elif page == "Dataset":
    st.subheader("Sample dataset")
    outcome = st.multiselect("Outcome filter", ["Accepted", "Rejected"], default=["Accepted", "Rejected"])
    view = data[data["outcome"].isin(outcome)]
    st.write(f"Records shown: {len(view)}")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download dataset CSV", view.to_csv(index=False).encode("utf-8"), "paper_dataset.csv", "text/csv")

elif page == "History":
    st.subheader("Saved predictions")
    with sqlite3.connect(DB_PATH) as con:
        history = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    if history.empty:
        st.info("No saved predictions yet. Go to Predict, make a prediction, and save it.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button("Download history CSV", history.to_csv(index=False).encode("utf-8"), "prediction_history.csv", "text/csv")

else:
    st.subheader("About this project")
    st.markdown("""
**Goal:** demonstrate a simple model for exploring the relationship between introduction cohesion and paper outcomes.

**Input metrics:** sentence cohesion, paragraph cohesion, research-gap clarity, and research-objective clarity.

**Algorithm:** Decision Tree Classifier.

**Database:** SQLite stores saved predictions.

**Steps:** Load CSV dataset → train model → enter scores → predict demo outcome → save history.
""")
    st.write("The dataset is synthetic and is included for learning/demo purposes.")
