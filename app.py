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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


st.set_page_config(page_title="AgroSense PFE", page_icon="🌱", layout="wide")

# =========================
# SESSION STATE
# =========================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =========================
# STYLE
# =========================
st.markdown("""
<style>
    .app-header {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
    }
    .app-header h1 { margin: 0; font-size: 2rem; }
    .app-header p  { margin: 5px 0 0; opacity: 0.9; font-size: 1rem; }

    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    .kpi-card .label { color: #666; font-size: 0.85rem; margin-bottom: 4px; }
    .kpi-card .value { color: #1B5E20; font-size: 1.7rem; font-weight: 700; }
    .kpi-card .sub   { color: #999; font-size: 0.75rem; }

    .info-card {
        background: #F1F8E9;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #C5E1A5;
        height: 100%;
    }
    .info-card h4 { color: #2E7D32; margin-top: 0; }
    .info-card p  { color: #555; font-size: 0.9rem; }

    .recommendation-box {
        background: #E8F5E9;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #A5D6A7;
        margin-top: 15px;
    }

    .prediction-result {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        margin-top: 20px;
    }
    .prediction-result h2 { font-size: 2rem; margin: 10px 0; }
    .prediction-result p { opacity: 0.9; }

    .small-note {
        color: #666;
        font-size: 0.85rem;
    }

    [data-testid="stSidebar"] { background-color: #1B5E20; }
    [data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# =========================
# HELPERS
# =========================
def kpi(label, value, sub=""):
    st.markdown(
        f'''
        <div class="kpi-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


@st.cache_data
def load_data():
    base = Path(__file__).parent
    for name in ["sensors_data.csv", "sensors_data.csv.csv", "Crop_recommendation.csv"]:
        p = base / name
        if p.exists():
            return pd.read_csv(p)
    return None


@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.drop_duplicates()
    return df


@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "SVM": SVC(probability=True)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Modèle"]
    best_model = trained_models[best_model_name]

    return trained_models, results_df, best_model_name, best_model


@st.cache_resource
def train_deep_learning_model(df):
    X = df[FEATURES]
    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(np.unique(y_encoded))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_scaled,
        y_train_cat,
        validation_split=0.2,
        epochs=30,
        batch_size=16,
        verbose=0
    )

    loss, dl_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)

    return model, scaler, label_encoder, dl_accuracy, history


def predict_with_deep_learning(input_data, dl_model, dl_scaler, dl_label_encoder):
    input_scaled = dl_scaler.transform(input_data)
    dl_pred = dl_model.predict(input_scaled, verbose=0)
    class_index = np.argmax(dl_pred, axis=1)[0]
    prediction = dl_label_encoder.inverse_transform([class_index])[0]
    confidence = float(np.max(dl_pred) * 100)

    top3_idx = np.argsort(dl_pred[0])[::-1][:3]
    top3 = []
    for idx in top3_idx:
        label = dl_label_encoder.inverse_transform([idx])[0]
        prob = float(dl_pred[0][idx] * 100)
        top3.append((label, prob))

    return prediction, confidence, top3


def get_alerts_from_values(temp, hum, ph_val, rain):
    alerts = []
    if temp > 35:
        alerts.append("🌡️ Température élevée : risque de stress thermique.")
    if hum < 40:
        alerts.append("💧 Humidité faible : irrigation recommandée.")
    if ph_val < 5.5:
        alerts.append("⚗️ pH acide : correction du sol conseillée.")
    if ph_val > 7.5:
        alerts.append("⚗️ pH élevé : attention à l’absorption des nutriments.")
    if rain < 50:
        alerts.append("🌧️ Faibles précipitations : surveiller les besoins en eau.")
    return alerts


def show_alerts(alerts):
    st.markdown("#### 🚨 Alertes intelligentes")
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("Aucune alerte majeure détectée. Les conditions semblent globalement stables.")


# =========================
# DATA
# =========================
df_raw = load_data()
if df_raw is None:
    st.error("Fichier CSV introuvable.")
    st.stop()

df_raw = preprocess_data(df_raw)

trained_models, results_df, best_model_name, best_model = train_models(df_raw)
dl_model, dl_scaler, dl_label_encoder, dl_accuracy, dl_history = train_deep_learning_model(df_raw)

dl_row = pd.DataFrame([{
    "Modèle": "Deep Learning (MLP)",
    "Accuracy": dl_accuracy,
    "Precision": np.nan,
    "Recall": np.nan,
    "F1-score": np.nan
}])

results_all = pd.concat([results_df, dl_row], ignore_index=True)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🌱 AgroSense PFE")
    st.markdown("*🌾 Projet de fin d'études • Monitoring agricole intelligent*")
    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio(
        "",
        [
            "🏠 Accueil",
            "🧱 Architecture",
            "📊 Tableau de Bord",
            "🔍 Analyse",
            "🧠 Modèles ML",
            "🤖 Deep Learning",
            "🎯 Prédiction",
            "📜 Historique",
            "📋 Données"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Filtres")
    all_crops = sorted(df_raw["label"].dropna().unique().tolist())
    selected_crops = st.multiselect("Type de culture", all_crops, default=all_crops)
    df = df_raw[df_raw["label"].isin(selected_crops)] if selected_crops else df_raw.copy()

    st.markdown(f"**Lignes filtrées : {len(df)} / {len(df_raw)}**")

    st.markdown("---")
    st.markdown("### Modèle de prédiction")
    prediction_model_choice = st.selectbox(
        "Choisir le modèle",
        ["Meilleur modèle automatique", "Random Forest", "Decision Tree", "KNN", "Logistic Regression", "SVM", "Deep Learning (MLP)"]
    )


# =========================
# PAGES
# =========================

# ACCUEIL
if page == "🏠 Accueil":
    st.markdown(
        '<div class="app-header"><h1>🌱 AgroSense — Système agricole intelligent</h1><p>Analyse, visualisation, prédiction et aide à la décision pour les données de capteurs agricoles</p></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Bienvenue sur **AgroSense**. Cette application ne se limite pas à la visualisation des données : "
        "elle intègre également plusieurs modèles de **Machine Learning**, un modèle de **Deep Learning**, "
        "ainsi qu’un **système d’alertes intelligentes** pour renforcer l’aspect informatique du projet."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("🌡️ Température moyenne", f"{df_raw['temperature'].mean():.1f} °C", "Toutes les mesures")
    with c2:
        kpi("💧 Humidité moyenne", f"{df_raw['humidity'].mean():.1f} %", "Humidité de l'air")
    with c3:
        kpi("🌧️ Précipitations moy.", f"{df_raw['rainfall'].mean():.1f} mm", "Moyenne globale")
    with c4:
        kpi("📋 Nombre de mesures", f"{len(df_raw)}", "Après nettoyage")

    st.markdown("#### Ce que fait l’application")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="info-card"><h4>📊 Analyse des données</h4><p>Visualisation des mesures issues des capteurs agricoles et exploration des tendances.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="info-card"><h4>🤖 Intelligence artificielle</h4><p>Comparaison entre plusieurs modèles classiques et un modèle de deep learning.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="info-card"><h4>🚨 Aide à la décision</h4><p>Détection automatique des situations critiques via des alertes intelligentes.</p></div>', unsafe_allow_html=True)


# ARCHITECTURE
elif page == "🧱 Architecture":
    st.markdown(
        '<div class="app-header"><h1>🧱 Architecture Informatique</h1><p>Organisation modulaire du système AgroSense</p></div>',
        unsafe_allow_html=True
    )

    st.markdown("""
### Architecture du système
Le système est organisé en plusieurs modules :

1. **Module d’acquisition des données**  
   Chargement des données issues des capteurs agricoles depuis un fichier CSV.

2. **Module de prétraitement**  
   Nettoyage des données, suppression des valeurs manquantes, suppression des doublons et conversion des types.

3. **Module de Machine Learning**  
   Entraînement et comparaison de plusieurs modèles de classification :
   Random Forest, Decision Tree, KNN, Logistic Regression et SVM.

4. **Module de Deep Learning**  
   Intégration d’un réseau de neurones artificiels de type **MLP**.

5. **Module de décision intelligente**  
   Génération d’alertes automatiques selon les conditions climatiques et agronomiques.

6. **Interface utilisateur**  
   Tableau de bord interactif développé avec **Streamlit**.
""")

    st.markdown("### Pipeline de traitement")
    st.code("Capteurs / CSV → Prétraitement → Analyse → ML / Deep Learning → Prédiction → Alertes / Décision")


# TABLEAU DE BORD
elif page == "📊 Tableau de Bord":
    st.markdown(
        '<div class="app-header"><h1>📊 Tableau de Bord</h1><p>Vue globale des conditions climatiques et nutritionnelles</p></div>',
        unsafe_allow_html=True
    )
    if df.empty:
        st.warning("Aucune donnée disponible.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("🌡️ Température moyenne", f"{df['temperature'].mean():.1f} °C", "Mesures filtrées")
    with c2:
        kpi("💧 Humidité moyenne", f"{df['humidity'].mean():.1f} %", "Mesures filtrées")
    with c3:
        kpi("🌧️ Précipitations moy.", f"{df['rainfall'].mean():.1f} mm", "Mesures filtrées")
    with c4:
        kpi("📋 Nombre de mesures", f"{len(df)}", "Après filtrage")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🌡️ Tendance de la température")
        fig = px.line(
            df.reset_index(),
            x="index",
            y="temperature",
            labels={"index": "Index", "temperature": "Température (°C)"}
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🌾 Répartition des cultures")
        fig = px.pie(df, names="label", color_discrete_sequence=px.colors.sequential.Greens_r)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🧪 Moyenne N/P/K par culture")
        npk = df.groupby("label")[["N", "P", "K"]].mean().reset_index()
        npk_m = npk.melt(id_vars="label", var_name="Nutriment", value_name="Valeur")
        fig = px.bar(npk_m, x="label", y="Valeur", color="Nutriment", barmode="group")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🌡️💧 Température vs Humidité")
        fig = px.scatter(df, x="temperature", y="humidity", color="label", opacity=0.7)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔥 Heatmap de corrélation")
    corr = df[FEATURES].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


# ANALYSE
elif page == "🔍 Analyse":
    st.markdown(
        '<div class="app-header"><h1>🔍 Analyse Détaillée</h1><p>Visualisations avancées et recommandations automatiques</p></div>',
        unsafe_allow_html=True
    )
    if df.empty:
        st.warning("Aucune donnée disponible.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🌾 Distribution température par culture")
        fig = px.box(df, x="label", y="temperature", color="label", color_discrete_sequence=px.colors.sequential.Greens_r)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20), showlegend=False)
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🧪 Comparaison N/P/K par culture")
        npk = df.groupby("label")[["N", "P", "K"]].mean().reset_index()
        npk_m = npk.melt(id_vars="label", var_name="Nutriment", value_name="Valeur")
        fig = px.bar(npk_m, x="label", y="Valeur", color="Nutriment", barmode="group")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🌧️ pH vs Précipitations")
    fig = px.scatter(df, x="ph", y="rainfall", color="label", opacity=0.7)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🧑‍🌾 Recommandations automatiques")
    avg_temp = df["temperature"].mean()
    avg_hum = df["humidity"].mean()
    avg_rain = df["rainfall"].mean()

    recs = []
    if avg_temp > 35:
        recs.append("⚠️ Température élevée — Arrosage recommandé.")
    if avg_hum < 40:
        recs.append("⚠️ Humidité faible — Irrigation nécessaire.")
    if avg_rain < 50:
        recs.append("⚠️ Faibles précipitations — Irrigation recommandée.")
    if not recs:
        recs.append("✅ Conditions optimales pour la plupart des cultures.")

    html_items = "".join(f"<li>{r}</li>" for r in recs)
    st.markdown(
        f'<div class="recommendation-box"><strong>Recommandations :</strong><ul>{html_items}</ul><small style="color:#666;">Ces recommandations ne remplacent pas l’expertise agronomique.</small></div>',
        unsafe_allow_html=True
    )

    show_alerts(get_alerts_from_values(avg_temp, avg_hum, df["ph"].mean(), avg_rain))


# MODELES ML
elif page == "🧠 Modèles ML":
    st.markdown(
        '<div class="app-header"><h1>🧠 Comparaison des Modèles ML</h1><p>Évaluation des algorithmes de classification</p></div>',
        unsafe_allow_html=True
    )

    st.markdown("### Résultats des modèles")
    st.dataframe(results_all, use_container_width=True)

    st.markdown(f"### ✅ Meilleur modèle classique : {best_model_name}")

    fig = px.bar(
        results_all,
        x="Modèle",
        y="Accuracy",
        text=results_all["Accuracy"].round(3),
        title="Comparaison des performances"
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Importance (Random Forest)")
    rf_model = trained_models["Random Forest"]
    fi_df = pd.DataFrame({
        "Variable": FEATURES,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(fi_df, x="Variable", y="Importance", text=fi_df["Importance"].round(3))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<p class="small-note">La feature importance permet d’identifier les variables les plus influentes dans la décision du modèle Random Forest.</p>',
        unsafe_allow_html=True
    )


# DEEP LEARNING
elif page == "🤖 Deep Learning":
    st.markdown(
        '<div class="app-header"><h1>🤖 Deep Learning</h1><p>Modèle de réseau de neurones artificiels (MLP)</p></div>',
        unsafe_allow_html=True
    )

    k1, k2 = st.columns(2)
    with k1:
        kpi("🧠 Modèle", "MLP", "Multi-Layer Perceptron")
    with k2:
        kpi("🎯 Accuracy", f"{dl_accuracy * 100:.2f}%", "Jeu de test")

    st.markdown("""
### Description du modèle
Le modèle de deep learning utilisé est un **réseau de neurones artificiels de type MLP**.
Il est composé de :
- une couche dense de 128 neurones avec fonction d’activation **ReLU**
- une couche **Dropout** pour limiter l’overfitting
- une deuxième couche dense de 64 neurones
- une couche de sortie avec **Softmax** pour la classification multi-classes
""")

    hist_df = pd.DataFrame({
        "Epoch": range(1, len(dl_history.history["accuracy"]) + 1),
        "Accuracy": dl_history.history["accuracy"],
        "Validation Accuracy": dl_history.history["val_accuracy"]
    })

    fig = px.line(hist_df, x="Epoch", y=["Accuracy", "Validation Accuracy"], title="Évolution de l'entraînement du modèle MLP")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)


# PREDICTION
elif page == "🎯 Prédiction":
    st.markdown(
        '<div class="app-header"><h1>🎯 Prédiction de Culture</h1><p>Choisissez les conditions de la parcelle et le modèle de prédiction</p></div>',
        unsafe_allow_html=True
    )

    if prediction_model_choice == "Meilleur modèle automatique":
        selected_model_name = best_model_name
    else:
        selected_model_name = prediction_model_choice

    if selected_model_name == "Deep Learning (MLP)":
        st.markdown(f"ℹ️ Modèle sélectionné : **{selected_model_name}** — Accuracy : **{dl_accuracy * 100:.1f}%**")
    else:
        selected_accuracy = float(results_df[results_df["Modèle"] == selected_model_name]["Accuracy"].iloc[0])
        st.markdown(f"ℹ️ Modèle sélectionné : **{selected_model_name}** — Accuracy : **{selected_accuracy * 100:.1f}%**")

    st.markdown("---")
    st.markdown("#### Entrez les conditions de votre parcelle :")

    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.slider("🌡️ Température (°C)", 0.0, 50.0, float(df_raw["temperature"].mean()), 0.1)
        hum = st.slider("💧 Humidité (%)", 0.0, 100.0, float(df_raw["humidity"].mean()), 0.1)
        rain = st.slider("🌧️ Précipitations (mm)", 0.0, 300.0, float(df_raw["rainfall"].mean()), 0.1)

    with c2:
        n_val = st.slider("🧪 Azote — N", 0, 140, int(df_raw["N"].mean()))
        p_val = st.slider("🧪 Phosphore — P", 0, 145, int(df_raw["P"].mean()))
        k_val = st.slider("🧪 Potassium — K", 0, 205, int(df_raw["K"].mean()))

    with c3:
        ph_val = st.slider("⚗️ pH du sol", 0.0, 14.0, float(df_raw["ph"].mean()), 0.1)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Prédire la culture", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame(
            [[n_val, p_val, k_val, temp, hum, ph_val, rain]],
            columns=FEATURES
        )

        if selected_model_name == "Deep Learning (MLP)":
            prediction, confidence, top3 = predict_with_deep_learning(
                input_data, dl_model, dl_scaler, dl_label_encoder
            )
        else:
            model = trained_models[selected_model_name]
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            confidence = float(np.max(proba) * 100)

            top3_idx = np.argsort(proba)[::-1][:3]
            classes = model.classes_
            top3 = [(classes[idx], float(proba[idx] * 100)) for idx in top3_idx]

        st.markdown(f"""
        <div class="prediction-result">
            <p>Culture recommandée pour vos conditions :</p>
            <h2>🌾 {prediction.upper()}</h2>
            <p>Confiance du modèle : <strong>{confidence:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Top 3 cultures recommandées :")
        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for i, (label, prob) in enumerate(top3):
            with cols[i]:
                kpi(f"{medals[i]} {label}", f"{prob:.1f}%", "Probabilité")

        alerts = get_alerts_from_values(temp, hum, ph_val, rain)
        show_alerts(alerts)

        st.session_state.prediction_history.append({
            "Modèle": selected_model_name,
            "N": n_val,
            "P": p_val,
            "K": k_val,
            "Température": temp,
            "Humidité": hum,
            "pH": ph_val,
            "Précipitations": rain,
            "Culture prédite": prediction,
            "Confiance (%)": round(confidence, 2)
        })


# HISTORIQUE
elif page == "📜 Historique":
    st.markdown(
        '<div class="app-header"><h1>📜 Historique des Prédictions</h1><p>Suivi des prédictions réalisées par l’application</p></div>',
        unsafe_allow_html=True
    )

    if not st.session_state.prediction_history:
        st.info("Aucune prédiction enregistrée pour le moment.")
    else:
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, use_container_width=True, height=450)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Exporter l’historique (CSV)",
            data=csv,
            file_name="historique_predictions.csv",
            mime="text/csv"
        )


# DONNEES
elif page == "📋 Données":
    st.markdown(
        '<div class="app-header"><h1>📋 Données</h1><p>Tableau détaillé des mesures issues des capteurs (après filtrage)</p></div>',
        unsafe_allow_html=True
    )

    if df.empty:
        st.warning("Aucune donnée disponible.")
    else:
        st.info(f"**{len(df)} enregistrements** affichés après filtrage.")
        st.dataframe(df, use_container_width=True, height=500)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Exporter les données filtrées (CSV)",
            data=csv,
            file_name="donnees_capteurs_filtrees.csv",
            mime="text/csv"
        )
