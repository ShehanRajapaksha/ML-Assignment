# Sri Lankan Used Car Price Predictor - Streamlit dashboard
# Code generated with assistance from Claude (Anthropic), reviewed and adapted by Shehan Rajapaksha.

import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Sri Lanka Car Price Predictor", page_icon="🚗", layout="wide")

# Hide Streamlit's Deploy button and top-right menu.
st.markdown("""
<style>
[data-testid="stToolbar"] {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------- Load the model and saved files (cached so it loads only once) ----------
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/model.pkl")

    # Put the XGBoost part back in (saved separately in portable JSON format)
    booster = xgb.Booster()
    booster.load_model("artifacts/xgb_part.json")
    model.named_estimators_["xgboost"]._Booster = booster

    info = joblib.load("artifacts/model_info.pkl")
    return model, info

@st.cache_data
def load_tables():
    results = pd.read_csv("artifacts/results.csv")
    importance = pd.read_csv("artifacts/feature_importance.csv")
    arima = pd.read_csv("artifacts/arima_results.csv")
    return results, importance, arima

model, info = load_model()
results, importance, arima = load_tables()


# ---------- Turn the form inputs into the exact columns the model was trained on ----------
def prepare_input(car):
    row = pd.DataFrame([car])
    row["Age"] = max(info["ref_year"] - row.loc[0, "YOM"], 0)
    row["Feature_count"] = row[["AC", "Power_Steering", "Power_Mirror", "Power_Window"]].sum(axis=1)

    row = row[info["cat_cols"] + info["num_cols"]]
    row = pd.get_dummies(row, columns=info["cat_cols"], dtype=int)

    if info["clean_names"]:
        row = row.rename(columns=lambda c: re.sub(r"[^A-Za-z0-9_]", "_", c))

    # Add any missing one-hot columns as 0, and put columns in the training order
    return row.reindex(columns=info["columns"], fill_value=0)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("About")
    st.write("Predicts the listed price of a used car in Sri Lanka using **ensemble learning** "
             "(bagging, boosting, stacking and hybrid models).")
    st.write("**Data:** ~9,500 car listings from **ikman.lk** and **Riyasewana.com** (Dec 2024 – Feb 2025).")
    st.write(f"**Model in use:** {info['model_name']}")
    st.metric("Average error (MAE)", f"{info['scores']['MAE (lakhs)']:.2f} lakhs")
    st.metric("Average error (%)", f"{info['scores']['MAPE (%)']:.1f}%")
    st.metric("R² score", f"{info['scores']['R2 test']:.3f}")
    st.divider()
    st.caption("⚠️ Code generated with assistance from Claude (Anthropic).")


st.title("Sri Lankan Used Car Price Predictor")
tab1, tab2, tab3 = st.tabs(["💰 Price Predictor", "📊 Model Comparison", "🔍 Insights"])


# =====================================================================
# TAB 1: Price Predictor
# =====================================================================
with tab1:
    st.subheader("Enter the car details")

    col1, col2, col3 = st.columns(3)

    with col1:
        brands = sorted(info["brand_models"].keys())
        brand = st.selectbox("Brand", brands, index=brands.index("TOYOTA") if "TOYOTA" in brands else 0)
        base_model = st.selectbox("Model", info["brand_models"][brand])
        yom = st.slider("Year of manufacture", info["yom_range"][0], info["yom_range"][1], 2015)
        engine = st.number_input("Engine capacity (cc)", min_value=600, max_value=6500,
                                 value=int(info["engine_median"]), step=50)

    with col2:
        gear = st.selectbox("Gear", info["gears"])
        fuel = st.selectbox("Fuel type", info["fuels"])
        condition = st.selectbox("Condition", info["conditions"])
        town = st.selectbox("Town", info["towns"],
                            index=info["towns"].index("COLOMBO") if "COLOMBO" in info["towns"] else 0)

    with col3:
        st.write("**Features**")
        ac = st.checkbox("Air condition", value=True)
        power_steering = st.checkbox("Power steering", value=True)
        power_mirror = st.checkbox("Power mirror", value=True)
        power_window = st.checkbox("Power window", value=True)
        leasing = st.checkbox("Leasing ongoing", value=False)

    if st.button("Predict price", type="primary", use_container_width=True):
        car = {
            "Brand": brand, "Base_Model": base_model, "YOM": yom, "Engine": engine,
            "Gear": gear, "Fuel": fuel, "Town": town, "Condition": condition,
            "Leasing": int(leasing), "AC": int(ac), "Power_Steering": int(power_steering),
            "Power_Mirror": int(power_mirror), "Power_Window": int(power_window),
        }

        price = float(np.exp(model.predict(prepare_input(car))[0]))   # model predicts log(price)
        error = info["scores"]["MAPE (%)"] / 100
        low, high = price * (1 - error), price * (1 + error)

        st.success(f"### Predicted price: **{price:,.2f} lakhs**  (≈ Rs {price * 100000:,.0f})")
        st.write(f"Likely range: **{low:,.2f} – {high:,.2f} lakhs** (± the model's average error of {error:.0%})")

        st.caption("Note: this is the expected *listing* price on ikman.lk / Riyasewana, not a guaranteed sale price. "
                   "Mileage isn't used, because it was found to be unreliable in the dataset.")


# =====================================================================
# TAB 2: Model Comparison
# =====================================================================
with tab2:
    st.subheader("How did each model perform on unseen cars?")

    best = results.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", best["Model"])
    c2.metric("MAE", f"{best['MAE (lakhs)']:.2f} lakhs")
    c3.metric("R²", f"{best['R2 test']:.3f}")

    colors = {"Baseline": "gray", "Bagging": "steelblue", "Boosting": "darkorange",
              "Stacking": "green", "Hybrid RF+ANN": "purple"}

    fig = px.bar(results.sort_values("MAE (lakhs)", ascending=False),
                 x="MAE (lakhs)", y="Model", color="Group", orientation="h",
                 color_discrete_map=colors, title="Average error per model (lower is better)")
    st.plotly_chart(fig, use_container_width=True)

    r2 = results.melt(id_vars="Model", value_vars=["R2 train", "R2 test"], var_name="Set", value_name="R²")
    fig2 = px.bar(r2, x="Model", y="R²", color="Set", barmode="group",
                  title="Train vs Test R² (a big gap means the model memorises = overfitting)")
    fig2.update_yaxes(range=[0.75, 1.0])
    st.plotly_chart(fig2, use_container_width=True)

    st.write("**Full results table**")
    st.dataframe(results.round(3), use_container_width=True, hide_index=True)

    st.write("**Random Forest + ARIMA hybrid** (time-based test on the newest listings)")
    st.dataframe(arima.round(3), use_container_width=True, hide_index=True)
    st.info("Over the 64-day window the market showed no trend, so the ARIMA component added nothing. "
            "With several years of data it could capture market-wide price changes.")

    with st.expander("Why do the models behave this way?"):
        st.markdown("""
- **Decision Tree** memorises the training cars (train R² ≈ 1.0), so it does worse on new cars.
- **Bagging** (Random Forest, Extra Trees) averages many trees, which cancels out their mistakes and reduces overfitting.
- **Boosting** (XGBoost, LightGBM, CatBoost) builds trees one after another to fix earlier errors, which reduces bias.
- **Stacking** combines XGBoost and Extra Trees, which make different mistakes, so it scores best.
- **The ANN** struggles on small tabular data with many 0/1 columns, so it's the weakest model.
""")


# =====================================================================
# TAB 3: Insights
# =====================================================================
with tab3:
    st.subheader("What drives used-car prices in Sri Lanka?")

    fig3 = px.bar(importance.sort_values("importance"), x="importance", y=importance.columns[0],
                  orientation="h", title="Feature importance (XGBoost)")
    fig3.update_layout(yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
**Key findings (from SHAP analysis):**
1. **Age:** the strongest factor. Cars lose value quickly in the first ~15 years.
2. **Automatic gear:** a large premium over manual.
3. **Toyota:** a clear brand premium (reliability and resale value).
4. **Engine size:** bigger engines cost more (import tax rises with cc).
5. **Comfort features:** cars with AC, power steering, mirrors and windows sell for more.
- Micro, Tata, Maruti and Alto pull prices down. Mercedes-Benz and Toyota Premio push them up.
""")

    st.warning("**Data-quality note:** mileage in the dataset was found to be age × 11,000 km for almost every car "
               "(correlation with age = 1.00), so it was excluded. With real mileage, accuracy would likely improve.")