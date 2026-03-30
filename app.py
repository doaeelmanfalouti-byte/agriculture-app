import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AgroSense AI", page_icon="🌱", layout="wide")

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]

# ================= DATA =================
@st.cache_data
def load_data():
    base = Path(__file__).parent
    for name in ["sensors_data.csv","Crop_recommendation.csv"]:
        p = base / name
        if p.exists():
            return pd.read_csv(p)
    return None

df_raw = load_data()

if df_raw is None:
    st.error("❌ CSV not found")
    st.stop()

df_raw.columns = [c.strip() for c in df_raw.columns]

for col in FEATURES:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

df_raw = df_raw.dropna().drop_duplicates()

# ================= MODELS =================
@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df["label"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM": SVC(probability=True),
        "Deep Learning (MLP)": MLPClassifier(max_iter=500)
    }

    results = {}
    trained = {}

    for name,model in models.items():
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test,pred)
        results[name] = acc
        trained[name] = model

    return trained,results

trained_models,results = train_models(df_raw)

best_model_name = max(results,key=results.get)
best_model = trained_models[best_model_name]

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🌱 AgroSense AI")

    page = st.radio("Navigation",
        ["Accueil","Architecture","Dashboard","Analyse","Models","Prediction"]
    )

# ================= ACCUEIL =================
if page == "Accueil":
    st.title("🌱 Smart Agriculture System")

    st.markdown("""
    This system uses:
    - Machine Learning models
    - Deep Learning (MLP)
    - Intelligent decision system
    """)

# ================= ARCHITECTURE =================
elif page == "Architecture":
    st.title("🧱 Architecture")

    st.code("""
    Data → Preprocessing → ML Models → Deep Learning → Prediction → Decision
    """)

# ================= DASHBOARD =================
elif page == "Dashboard":
    st.title("📊 Dashboard")

    st.metric("Température",round(df_raw["temperature"].mean(),1))
    st.metric("Humidité",round(df_raw["humidity"].mean(),1))

    fig = px.line(df_raw,x=df_raw.index,y="temperature")
    st.plotly_chart(fig)

# ================= ANALYSE =================
elif page == "Analyse":
    st.title("🔍 Analyse")

    fig = px.scatter(df_raw,x="temperature",y="humidity",color="label")
    st.plotly_chart(fig)

    # ALERTES
    if df_raw["temperature"].mean() > 35:
        st.warning("🔥 Température élevée")
    if df_raw["humidity"].mean() < 40:
        st.warning("💧 Humidité faible")

# ================= MODELS =================
elif page == "Models":
    st.title("🧠 Comparaison des modèles")

    df_res = pd.DataFrame(list(results.items()),columns=["Model","Accuracy"])
    st.dataframe(df_res)

    fig = px.bar(df_res,x="Model",y="Accuracy")
    st.plotly_chart(fig)

    st.success(f"Best Model: {best_model_name}")

# ================= PREDICTION =================
elif page == "Prediction":
    st.title("🤖 Prediction")

    n = st.slider("N",0,140,50)
    p = st.slider("P",0,140,50)
    k = st.slider("K",0,200,50)
    t = st.slider("Temp",0,50,25)
    h = st.slider("Humidity",0,100,60)
    ph = st.slider("pH",0.0,14.0,6.5)
    r = st.slider("Rainfall",0,300,100)

    model_choice = st.selectbox("Model",list(trained_models.keys()))

    if st.button("Predict"):
        data = pd.DataFrame([[n,p,k,t,h,ph,r]],columns=FEATURES)
        model = trained_models[model_choice]

        pred = model.predict(data)[0]
        st.success(f"🌾 Recommended: {pred}")
