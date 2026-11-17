import pandas as pd
import streamlit as st

from model import LABEL_ORDER, TARGET, load_data, predict, train

st.set_page_config(page_title="Obesity Level Predictor", layout="centered")

st.markdown(
    """
    <style>
    .block-container {max-width: 860px; padding-top: 2.2rem;}
    h1 {letter-spacing: -0.02em;}
    div[data-testid="stForm"] {border: 1px solid #CBD5E1; border-radius: 10px; padding: 1.2rem 1.4rem;}
    .result {border-left: 5px solid #0E7490; background: #F1F5F9; padding: 0.9rem 1.2rem; border-radius: 4px; margin: 0.6rem 0 1rem 0;}
    .result .label {font-size: 1.7rem; font-weight: 700; color: #0F172A;}
    .result .sub {color: #475569; font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the model on 2,111 records...")
def get_bundle():
    df = load_data()
    return df, train(df)


df, bundle = get_bundle()
pretty = lambda s: str(s).replace("_", " ")

# Plain-language field names and scale hints (from the dataset's documentation).
FIELDS = {
    "Gender": "Gender",
    "Age": "Age (years)",
    "Height": "Height (m)",
    "Weight": "Weight (kg)",
    "family_history_with_overweight": "Family history of overweight",
    "FAVC": "Eats high-calorie food often",
    "FCVC": "Vegetables in meals (1 = never, 3 = always)",
    "NCP": "Main meals per day (1 to 4)",
    "CAEC": "Eats between meals",
    "SMOKE": "Smokes",
    "CH2O": "Daily water (1 = under 1 L, 3 = over 2 L)",
    "SCC": "Tracks calories eaten",
    "FAF": "Physical activity (0 = none, 3 = 4-5 days a week)",
    "TUE": "Screen time (0 = 0-2 h, 2 = over 5 h)",
    "CALC": "Alcohol",
    "MTRANS": "Main transport",
}
DEFAULTS = {"Age": 25, "Height": 1.70, "Weight": 70.0}

st.title("Obesity Level Predictor")
st.write(
    "Enter eating habits and physical details to see which weight category a "
    "Random Forest model predicts. Trained on the UCI obesity dataset "
    "(2,111 records)."
)

with st.form("inputs"):
    values = {}
    cols = st.columns(2)
    for i, col in enumerate(bundle["feature_cols"]):
        label = FIELDS.get(col, col)
        with cols[i % 2]:
            if col in bundle["cat_cols"]:
                options = sorted(df[col].unique(), key=lambda v: (v not in ("no", "yes"), v))
                default = df[col].mode()[0]
                values[col] = st.selectbox(
                    label, options, index=options.index(default), format_func=lambda v: pretty(v).capitalize()
                )
            elif col == "Age":
                values[col] = float(st.number_input(label, 14, 61, int(DEFAULTS["Age"])))
            elif col == "Height":
                values[col] = float(st.number_input(label, 1.40, 2.10, DEFAULTS["Height"], step=0.01))
            elif col == "Weight":
                values[col] = float(st.number_input(label, 35.0, 180.0, DEFAULTS["Weight"], step=0.5))
            else:
                lo, hi = int(round(df[col].min())), int(round(df[col].max()))
                values[col] = float(st.slider(label, lo, hi, int(round(df[col].median()))))
    submitted = st.form_submit_button("Predict category", type="primary")

if submitted:
    label, probs = predict(bundle, values)
    bmi = values["Weight"] / values["Height"] ** 2
    st.markdown(
        f"""<div class="result"><div class="label">{pretty(label)}</div>
        <div class="sub">Model confidence {probs[label]:.0%} &nbsp;|&nbsp; BMI from your inputs: {bmi:.1f}</div></div>""",
        unsafe_allow_html=True,
    )
    chart = probs.rename(index=pretty)
    st.bar_chart(chart, horizontal=True, height=300)

with st.expander("How the model works, and its limits"):
    st.write(
        f"Pipeline: label-encode categorical columns, scale features with StandardScaler, "
        f"80/20 train-test split ({bundle['n_train']} train / {bundle['n_test']} test), "
        f"Random Forest with 300 trees."
    )
    st.metric("Accuracy on held-out test data", f"{bundle['accuracy']:.1%}")
    st.write("Which inputs the model relies on most:")
    st.bar_chart(bundle["importances"].rename(index=lambda c: FIELDS.get(c, c).split(" (")[0]), horizontal=True, height=380)
    st.write(
        "Limits to keep in mind: the dataset's labels were assigned from BMI, so Height and "
        "Weight dominate the result, and most records are synthetic (generated with SMOTE). "
        "This is a learning project and not medical advice."
    )
