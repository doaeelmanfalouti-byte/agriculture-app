import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AgroSense PFE", page_icon="🌱", layout="wide")

st.markdown("""
<style>
    .app-header { background: linear-gradient(135deg, #1B5E20, #2E7D32); padding: 20px 30px; border-radius: 12px; margin-bottom: 25px; color: white; }
    .app-header h1 { margin: 0; font-size: 2rem; }
    .app-header p  { margin: 5px 0 0; opacity: 0.85; font-size: 1rem; }
    .kpi-card { background: white; border-radius: 12px; padding: 18px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 5px solid #4CAF50; margin-bottom: 10px; }
    .kpi-card .label { color: #666; font-size: 0.85rem; margin-bottom: 4px; }
    .kpi-card .value { color: #1B5E20; font-size: 1.7rem; font-weight: 700; }
    .kpi-card .sub   { color: #999; font-size: 0.75rem; }
    .info-card { background: #F1F8E9; border-radius: 12px; padding: 20px; border: 1px solid #C5E1A5; height: 100%; }
    .info-card h4 { color: #2E7D32; margin-top: 0; }
    .info-card p  { color: #555; font-size: 0.9rem; }
    .recommendation-box { background: #E8F5E9; border-radius: 12px; padding: 20px; border: 1px solid #A5D6A7; margin-top: 15px; }
    .prediction-result { background: linear-gradient(135deg, #1B5E20, #2E7D32); border-radius: 16px; padding: 30px; text-align: center; color: white; margin-top: 20px; }
    .prediction-result h2 { font-size: 2rem; margin: 10px 0; }
    .prediction-result p { opacity: 0.85; }
    [data-testid="stSidebar"] { background-color: #1B5E20; }
    [data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base = Path(__file__).parent
    for name in ["sensors_data.csv", "sensors_data.csv.csv", "Crop_recommendation.csv"]:
        p = base / name
        if p.exists():
            return pd.read_csv(p)
    return None

@st.cache_resource
def train_model(df):
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

df_raw = load_data()
if df_raw is None:
    st.error("Fichier CSV introuvable.")
    st.stop()

df_raw.columns = [c.strip() for c in df_raw.columns]
for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

model, accuracy = train_model(df_raw)

with st.sidebar:
    st.markdown("## 🌱 AgroSense PFE")
    st.markdown("*🌾 Projet de fin d'études • Monitoring agricole*")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["🏠 Accueil", "📊 Tableau de Bord", "🔍 Analyse", "🤖 Prédiction", "📋 Données"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Filtres")
    all_crops = sorted(df_raw["label"].dropna().unique().tolist())
    selected_crops = st.multiselect("Type de culture", all_crops, default=all_crops)
    df = df_raw[df_raw["label"].isin(selected_crops)] if selected_crops else df_raw.copy()
    st.markdown(f"**Lignes filtrées : {len(df)} / {len(df_raw)}**")

def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi-card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ACCUEIL
# ══════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown('<div class="app-header"><h1>🌱 AgroSense — Tableau de bord agricole intelligent</h1><p>Application de suivi, d\'analyse et de prédiction des données de capteurs agricoles (PFE)</p></div>', unsafe_allow_html=True)
    st.markdown("Bienvenue sur **AgroSense**. Cette application vous permet de visualiser, analyser et interpréter les données de vos capteurs afin d'optimiser la gestion de vos parcelles.")
    st.markdown("#### Comment utiliser cette application")
    st.markdown("- **Filtrez** les données dans la barre latérale.\n- Accédez au **Tableau de bord** pour une vue globale.\n- Utilisez **Analyse** pour des graphiques détaillés.\n- Utilisez **Prédiction** pour recommander une culture selon vos conditions.\n- Consultez **Données** pour explorer et exporter le tableau.")
    st.markdown("#### Vue d'ensemble rapide")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("🌡️ Température moyenne", f"{df_raw['temperature'].mean():.1f} °C", "Sur l'ensemble des mesures.")
    with c2: kpi("💧 Humidité moyenne", f"{df_raw['humidity'].mean():.1f} %", "Humidité de l'air.")
    with c3: kpi("🌧️ Précipitations moy.", f"{df_raw['rainfall'].mean():.1f} mm", "Précipitations moyennes.")
    with c4: kpi("📋 Nombre de mesures", f"{len(df_raw)}", "Enregistrements totaux.")
    st.markdown("#### Ce que fait l'application")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="info-card"><h4>☀️ Suivi climatique</h4><p>Visualisez température, humidité et précipitations pour détecter les périodes de stress.</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="info-card"><h4>🧪 Analyse des nutriments</h4><p>Étudiez N, P, K et pH par type de culture pour optimiser la fertilisation.</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="info-card"><h4>🤖 Prédiction ML</h4><p>Un modèle Random Forest recommande la culture la plus adaptée à vos conditions.</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TABLEAU DE BORD
# ══════════════════════════════════════════════
elif page == "📊 Tableau de Bord":
    st.markdown('<div class="app-header"><h1>📊 Tableau de Bord</h1><p>Vue d\'ensemble des conditions climatiques et nutritionnelles.</p></div>', unsafe_allow_html=True)
    if df.empty:
        st.warning("Aucune donnée disponible.")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("🌡️ Température moyenne", f"{df['temperature'].mean():.1f} °C", "Mesures filtrées.")
    with c2: kpi("💧 Humidité moyenne", f"{df['humidity'].mean():.1f} %", "Humidité filtrée.")
    with c3: kpi("🌧️ Précipitations moy.", f"{df['rainfall'].mean():.1f} mm", "Précipitations filtrées.")
    with c4: kpi("📋 Nombre de mesures", f"{len(df)}", "Lignes après filtrage.")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🌡️ Tendance de la température")
        fig = px.line(df.reset_index(), x="index", y="temperature", color_discrete_sequence=["#4CAF50"], labels={"index": "Index", "temperature": "Température (°C)"})
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
        fig = px.bar(npk_m, x="label", y="Valeur", color="Nutriment", barmode="group", color_discrete_sequence=["#1B5E20", "#4CAF50", "#A5D6A7"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### 🌡️💧 Température vs Humidité")
        fig = px.scatter(df, x="temperature", y="humidity", color="label", opacity=0.7, labels={"temperature": "Température (°C)", "humidity": "Humidité (%)"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 🔥 Heatmap de corrélation")
    corr = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# ANALYSE
# ══════════════════════════════════════════════
elif page == "🔍 Analyse":
    st.markdown('<div class="app-header"><h1>🔍 Analyse Détaillée</h1><p>Visualisations avancées pour mieux comprendre les données.</p></div>', unsafe_allow_html=True)
    if df.empty:
        st.warning("Aucune donnée disponible.")
        st.stop()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🌾 Distribution température par culture")
        fig = px.box(df, x="label", y="temperature", color="label", color_discrete_sequence=px.colors.sequential.Greens_r, labels={"label": "Culture", "temperature": "Température (°C)"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20), showlegend=False)
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### 🧪 Comparaison N/P/K par culture")
        npk = df.groupby("label")[["N", "P", "K"]].mean().reset_index()
        npk_m = npk.melt(id_vars="label", var_name="Nutriment", value_name="Valeur")
        fig = px.bar(npk_m, x="label", y="Valeur", color="Nutriment", barmode="group", color_discrete_sequence=["#1B5E20", "#4CAF50", "#A5D6A7"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 🌧️ pH vs Précipitations")
    fig = px.scatter(df, x="ph", y="rainfall", color="label", opacity=0.7, labels={"ph": "pH du sol", "rainfall": "Précipitations (mm)"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111", font_color="white", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### 🧑‍🌾 Recommandations automatiques")
    avg_temp = df["temperature"].mean()
    avg_hum  = df["humidity"].mean()
    avg_rain = df["rainfall"].mean()
    recs = []
    if avg_temp > 35: recs.append("⚠️ Température élevée — Arrosage recommandé.")
    if avg_hum < 40:  recs.append("⚠️ Humidité faible — Irrigation nécessaire.")
    if avg_rain < 50: recs.append("⚠️ Faibles précipitations — Irrigation recommandée.")
    if not recs: recs.append("✅ Conditions optimales pour la plupart des cultures.")
    html_items = "".join(f"<li>{r}</li>" for r in recs)
    st.markdown(f'<div class="recommendation-box"><strong>Recommandations :</strong><ul>{html_items}</ul><small style="color:#666;">Ces recommandations ne remplacent pas l\'expertise agronomique.</small></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PREDICTION ML
# ══════════════════════════════════════════════
elif page == "🤖 Prédiction":
    st.markdown('<div class="app-header"><h1>🤖 Prédiction de Culture</h1><p>Entrez les conditions de votre parcelle — le modèle recommande la culture la plus adaptée.</p></div>', unsafe_allow_html=True)

    st.markdown(f"ℹ️ Modèle **Random Forest** entraîné sur {len(df_raw)} mesures — Précision : **{accuracy*100:.1f}%**")
    st.markdown("---")

    st.markdown("#### Entrez les conditions de votre parcelle :")

    c1, c2, c3 = st.columns(3)
    with c1:
        temp  = st.slider("🌡️ Température (°C)", 0.0, 50.0, float(df_raw["temperature"].mean()), 0.1)
        hum   = st.slider("💧 Humidité (%)", 0.0, 100.0, float(df_raw["humidity"].mean()), 0.1)
        rain  = st.slider("🌧️ Précipitations (mm)", 0.0, 300.0, float(df_raw["rainfall"].mean()), 0.1)
    with c2:
        n_val = st.slider("🧪 Azote — N", 0, 140, int(df_raw["N"].mean()))
        p_val = st.slider("🧪 Phosphore — P", 0, 145, int(df_raw["P"].mean()))
        k_val = st.slider("🧪 Potassium — K", 0, 205, int(df_raw["K"].mean()))
    with c3:
        ph_val = st.slider("⚗️ pH du sol", 0.0, 14.0, float(df_raw["ph"].mean()), 0.1)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Prédire la culture", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([[n_val, p_val, k_val, temp, hum, ph_val, rain]],
                                   columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        confidence = max(proba) * 100

        st.markdown(f"""
        <div class="prediction-result">
            <p>Culture recommandée pour vos conditions :</p>
            <h2>🌾 {prediction.upper()}</h2>
            <p>Confiance du modèle : <strong>{confidence:.1f}%</strong></p>
        </div>""", unsafe_allow_html=True)

        # Top 3
        st.markdown("#### Top 3 cultures recommandées :")
        classes = model.classes_
        top3_idx = np.argsort(proba)[::-1][:3]
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        medals = ["🥇", "🥈", "🥉"]
        for i, idx in enumerate(top3_idx):
            with cols[i]:
                kpi(f"{medals[i]} {classes[idx]}", f"{proba[idx]*100:.1f}%", "Probabilité")

# ══════════════════════════════════════════════
# DONNEES
# ══════════════════════════════════════════════
elif page == "📋 Données":
    st.markdown('<div class="app-header"><h1>📋 Données</h1><p>Tableau détaillé des mesures issues des capteurs (après filtrage).</p></div>', unsafe_allow_html=True)
    if df.empty:
        st.warning("Aucune donnée disponible.")
    else:
        st.info(f"**{len(df)} enregistrements** affichés après filtrage.")
        st.dataframe(df, use_container_width=True, height=500)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label="💾 Exporter les données filtrées (CSV)", data=csv, file_name="donnees_capteurs_filtrees.csv", mime="text/csv")
