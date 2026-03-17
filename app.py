import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_TITLE = "Tableau de bord agricole intelligent"
APP_ICON = "🌱"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
  .block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.3rem;
  }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 55%, #1B5E20 100%);
    color: #ffffff;
  }
  [data-testid="stSidebar"] * {
    color: #F1F8E9 !important;
  }

  .app-header {
    padding: 0.6rem 0 0.3rem 0;
    border-bottom: 1px solid #E0E0E0;
    margin-bottom: 0.8rem;
  }
  .app-header-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1B5E20;
    letter-spacing: -0.03em;
  }
  .app-header-subtitle {
    font-size: 0.95rem;
    color: #616161;
  }

  .pfe-badge {
    display: inline-flex;
    gap: .4rem;
    align-items: center;
    padding: .30rem .65rem;
    border-radius: 999px;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.22);
    font-size: .82rem;
  }

  .kpi-card {
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    background: linear-gradient(145deg, #ffffff 0%, #F1F8E9 100%);
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.06);
    border: 1px solid #E0E0E0;
  }
  .kpi-title {
    font-size: .88rem;
    color: #616161;
    margin-bottom: .15rem;
    display: flex;
    align-items: center;
    gap: .35rem;
  }
  .kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #1B5E20;
  }
  .kpi-sub {
    font-size: .8rem;
    color: #757575;
    margin-top: .1rem;
  }

  .info-card {
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    background: #ffffff;
    box-shadow: 0 8px 22px rgba(0,0,0,0.04);
    border: 1px solid #E0E0E0;
  }
  .info-title {
    font-size: 1.0rem;
    font-weight: 600;
    margin-bottom: .35rem;
    color: #2E7D32;
  }
  .info-text {
    font-size: .9rem;
    color: #616161;
  }

  .recommendation-box {
    border-radius: 16px;
    padding: 1.0rem 1.1rem;
    background: #F1F8E9;
    border: 1px solid #C5E1A5;
  }
  .recommendation-title {
    font-weight: 600;
    color: #2E7D32;
    margin-bottom: .4rem;
  }
  .recommendation-item {
    font-size: .92rem;
    margin-bottom: .15rem;
    color: #37474F;
  }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Charge le fichier de données de capteurs."""
    here = pathlib.Path(__file__).parent
    candidates = [
        here / "sensors_data.csv",
        here / "sensors_data.csv.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df.attrs["__source_path__"] = str(p)
            return df
    raise FileNotFoundError(
        "Impossible de trouver `sensors_data.csv` à côté de `app.py`.\n"
        "Fichiers attendus : sensors_data.csv ou sensors_data.csv.csv."
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def kpi_card(title: str, icon: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-title"><span>{icon}</span><span>{title}</span></div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def info_card(title: str, icon: str, text: str) -> None:
    st.markdown(
        f"""
<div class="info-card">
  <div class="info-title">{icon} {title}</div>
  <div class="info-text">{text}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def compute_basic_stats(df: pd.DataFrame) -> dict:
    return {
        "temp_mean": float(df["Temperature"].mean(skipna=True)) if not df.empty else np.nan,
        "hum_mean": float(df["Humidity"].mean(skipna=True)) if not df.empty else np.nan,
        "mois_mean": float(df["Moisture"].mean(skipna=True)) if not df.empty else np.nan,
        "count": int(len(df)),
    }


def build_recommendations(stats: dict) -> list[str]:
    recos: list[str] = []
    if np.isfinite(stats["temp_mean"]) and stats["temp_mean"] > 35:
        recos.append("⚠️ Température élevée - Arrosage recommandé.")
    if np.isfinite(stats["hum_mean"]) and stats["hum_mean"] < 40:
        recos.append("⚠️ Humidité faible - Irrigation nécessaire.")
    if np.isfinite(stats["mois_mean"]) and stats["mois_mean"] < 30:
        recos.append("⚠️ Sol sec - Augmenter l'irrigation.")
    if not recos:
        recos.append("✅ Conditions optimales pour la plupart des cultures.")
    return recos


df_raw = load_data()
df_raw = standardize_columns(df_raw)

required = [
    "Temperature",
    "Humidity",
    "Moisture",
    "Soil_Type",
    "Crop_Type",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
    "Fertilizer_Name",
]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Colonnes manquantes dans le CSV : {missing}")
    st.stop()

df = coerce_numeric(
    df_raw,
    ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"],
)


with st.container():
    st.markdown(
        f"""
<div class="app-header">
  <div class="app-header-title">{APP_ICON} {APP_TITLE}</div>
  <div class="app-header-subtitle">
    Application de suivi et d'analyse des données de capteurs agricoles (PFE).
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


st.sidebar.markdown(f"### {APP_ICON} AgroSense PFE")
st.sidebar.markdown(
    '<span class="pfe-badge">🌾 Projet de fin d\'études • Monitoring agricole</span>',
    unsafe_allow_html=True,
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "📊 Tableau de Bord",
        "🔍 Analyse",
        "📋 Données",
    ],
    index=1,
)

soil_options = sorted([x for x in df["Soil_Type"].dropna().astype(str).unique().tolist()])
crop_options = sorted([x for x in df["Crop_Type"].dropna().astype(str).unique().tolist()])

selected_soils = st.sidebar.multiselect(
    "Type de sol",
    options=soil_options,
    default=soil_options,
)
selected_crops = st.sidebar.multiselect(
    "Type de culture",
    options=crop_options,
    default=crop_options,
)

filtered = df.copy()
if selected_soils:
    filtered = filtered[filtered["Soil_Type"].astype(str).isin(selected_soils)]
else:
    filtered = filtered.iloc[0:0]
if selected_crops:
    filtered = filtered[filtered["Crop_Type"].astype(str).isin(selected_crops)]
else:
    filtered = filtered.iloc[0:0]

st.sidebar.divider()
st.sidebar.caption(f"Lignes filtrées : **{len(filtered):,}** / {len(df):,}")
source_path = df_raw.attrs.get("__source_path__")
if source_path and source_path.endswith(".csv.csv"):
    st.sidebar.warning(
        "Le fichier est nommé `sensors_data.csv.csv`. "
        "Vous pouvez le renommer en `sensors_data.csv` si nécessaire."
    )
st.sidebar.caption(f"Source des données : `{pathlib.Path(source_path).name}`" if source_path else "")


color_seq = ["#1B5E20", "#2E7D32", "#4CAF50", "#8BC34A", "#C0CA33", "#558B2F"]


stats_all = compute_basic_stats(df)
stats_filtered = compute_basic_stats(filtered)


if page.startswith("🏠"):
    st.markdown("### 🏠 Accueil")
    st.write(
        "Bienvenue sur votre **tableau de bord agricole intelligent**. "
        "Cette application vous permet de visualiser, analyser et interpréter les données "
        "de vos capteurs (température, humidité, humidité du sol, nutriments…) afin "
        "d'optimiser la gestion de vos parcelles."
    )

    st.markdown("#### 🌾 Comment utiliser cette application")
    st.markdown(
        "- **Filtrez** les données dans la barre latérale (type de sol, type de culture).\n"
        "- Accédez au **Tableau de bord** pour une vue globale.\n"
        "- Utilisez l'onglet **Analyse** pour des graphiques détaillés et des recommandations.\n"
        "- Consultez l'onglet **Données** pour explorer et exporter le tableau complet."
    )

    st.markdown("#### 📌 Vue d'ensemble rapide")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Température moyenne",
            "🌡️",
            f"{stats_all['temp_mean']:.1f} °C" if np.isfinite(stats_all["temp_mean"]) else "—",
            "Sur l'ensemble des mesures.",
        )
    with c2:
        kpi_card(
            "Humidité moyenne",
            "💧",
            f"{stats_all['hum_mean']:.1f} %" if np.isfinite(stats_all["hum_mean"]) else "—",
            "Humidité de l'air.",
        )
    with c3:
        kpi_card(
            "Humidité du sol",
            "🌱",
            f"{stats_all['mois_mean']:.1f}" if np.isfinite(stats_all["mois_mean"]) else "—",
            "Indice moyen d'humidité du sol.",
        )
    with c4:
        kpi_card(
            "Nombre de mesures",
            "📈",
            f"{stats_all['count']:,}",
            "Nombre total de lignes dans le jeu de données.",
        )

    st.markdown("#### 📘 Ce que fait l'application")
    c1, c2, c3 = st.columns(3)
    with c1:
        info_card(
            "Suivi des conditions climatiques",
            "☀️",
            "Visualisez la température et l'humidité pour détecter les périodes de stress thermique ou hydrique.",
        )
    with c2:
        info_card(
            "Analyse du sol et des nutriments",
            "🧪",
            "Étudiez les nutriments majeurs (azote, phosphore, potassium) par type de culture et type de sol.",
        )
    with c3:
        info_card(
            "Aide à la décision pour l'agriculteur",
            "🧑‍🌾",
            "Recevez des recommandations simples (irrigation, vigilance) à partir des mesures filtrées.",
        )

elif page.startswith("📊"):
    st.markdown("### 📊 Tableau de Bord")
    st.caption("Vue d'ensemble des conditions climatiques et de l'état des sols en fonction des filtres sélectionnés.")

    if filtered.empty:
        st.warning("Aucune donnée ne correspond aux filtres actuels. Veuillez ajuster le type de sol ou de culture dans la barre latérale.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Température moyenne",
            "🌡️",
            f"{stats_filtered['temp_mean']:.1f} °C" if np.isfinite(stats_filtered["temp_mean"]) else "—",
            "Sur les mesures filtrées.",
        )
    with c2:
        kpi_card(
            "Humidité moyenne",
            "💧",
            f"{stats_filtered['hum_mean']:.1f} %" if np.isfinite(stats_filtered["hum_mean"]) else "—",
            "Humidité de l'air filtrée.",
        )
    with c3:
        kpi_card(
            "Humidité du sol",
            "🌱",
            f"{stats_filtered['mois_mean']:.1f}" if np.isfinite(stats_filtered["mois_mean"]) else "—",
            "Indice moyen d'humidité du sol.",
        )
    with c4:
        kpi_card(
            "Nombre de mesures",
            "📈",
            f"{stats_filtered['count']:,}",
            "Nombre de lignes après filtrage.",
        )

    st.divider()

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        fig_temp = px.line(
            filtered.reset_index(drop=True),
            x=filtered.reset_index(drop=True).index,
            y="Temperature",
            title="🌡️ Tendance de la température (index des mesures)",
            labels={"x": "Index de la mesure", "Temperature": "Température (°C)"},
            color_discrete_sequence=[color_seq[0]],
        )
        fig_temp.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360)
        st.plotly_chart(fig_temp, use_container_width=True)

    with right:
        soil_counts = filtered["Soil_Type"].astype(str).value_counts(dropna=False).reset_index()
        soil_counts.columns = ["Soil_Type", "count"]
        fig_soil = px.pie(
            soil_counts,
            names="Soil_Type",
            values="count",
            title="🌍 Répartition des types de sol",
            color_discrete_sequence=color_seq,
            hole=0.35,
        )
        fig_soil.update_traces(textposition="inside", textinfo="percent+label")
        fig_soil.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360, legend_title_text="Type de sol")
        st.plotly_chart(fig_soil, use_container_width=True)

    left2, right2 = st.columns(2, gap="large")
    with left2:
        avg_npk = (
            filtered.groupby("Crop_Type", dropna=False)[["Nitrogen", "Phosphorous", "Potassium"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        avg_npk_melt = avg_npk.melt(id_vars="Crop_Type", var_name="Nutriment", value_name="Valeur")
        fig_npk = px.bar(
            avg_npk_melt,
            x="Crop_Type",
            y="Valeur",
            color="Nutriment",
            barmode="group",
            title="🧪 Moyenne des nutriments (N, P, K) par type de culture",
            labels={"Crop_Type": "Type de culture"},
            color_discrete_sequence=color_seq,
        )
        fig_npk.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)
        st.plotly_chart(fig_npk, use_container_width=True)

    with right2:
        fig_scatter = px.scatter(
            filtered,
            x="Temperature",
            y="Humidity",
            color="Crop_Type",
            title="🌡️💧 Température vs Humidité (coloré par type de culture)",
            labels={"Temperature": "Température (°C)", "Humidity": "Humidité (%)"},
            color_discrete_sequence=color_seq,
            hover_data=["Soil_Type", "Moisture", "Fertilizer_Name"],
        )
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380, legend_title_text="Type de culture")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = filtered[numeric_cols].corr()
        fig_heat = px.imshow(
            corr,
            text_auto=True,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            title="🔥 Heatmap de corrélation (variables numériques)",
        )
        fig_heat.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=450)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Pas assez de colonnes numériques pour calculer une matrice de corrélation.")

elif page.startswith("🔍"):
    st.markdown("### 🔍 Analyse détaillée")
    st.caption("Visualisations avancées pour mieux comprendre l'effet du type de sol et du type de culture.")

    if filtered.empty:
        st.warning("Aucune donnée ne correspond aux filtres actuels. Veuillez ajuster le type de sol ou de culture dans la barre latérale.")
        st.stop()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_box = px.box(
            filtered,
            x="Soil_Type",
            y="Temperature",
            color="Soil_Type",
            title="🌍 Distribution de la température par type de sol",
            labels={"Soil_Type": "Type de sol", "Temperature": "Température (°C)"},
            color_discrete_sequence=color_seq,
        )
        fig_box.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        npk_by_crop = (
            filtered.groupby("Crop_Type", dropna=False)[["Nitrogen", "Phosphorous", "Potassium"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        npk_melt = npk_by_crop.melt(id_vars="Crop_Type", var_name="Nutriment", value_name="Valeur")
        fig_npk2 = px.bar(
            npk_melt,
            x="Crop_Type",
            y="Valeur",
            color="Nutriment",
            barmode="group",
            title="🧪 Comparaison Azote / Phosphore / Potassium par culture",
            labels={"Crop_Type": "Type de culture"},
            color_discrete_sequence=color_seq,
        )
        fig_npk2.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        st.plotly_chart(fig_npk2, use_container_width=True)

    st.markdown("#### 🧑‍🌾 Recommandations pour l'agriculteur")
    stats = stats_filtered
    recos = build_recommendations(stats)
    st.markdown(
        """
<div class="recommendation-box">
  <div class="recommendation-title">Synthèse basée sur les mesures filtrées :</div>
""",
        unsafe_allow_html=True,
    )
    for r in recos:
        st.markdown(f'<div class="recommendation-item">• {r}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Ces recommandations sont basées sur des règles simples et ne remplacent pas l'expertise "
        "agronomique, mais elles peuvent guider des décisions rapides (irrigation, surveillance, etc.)."
    )

else:
    st.markdown("### 📋 Données")
    st.caption("Tableau détaillé des mesures issues des capteurs (après filtrage).")

    if filtered.empty:
        st.warning("Aucune donnée ne correspond aux filtres actuels. Veuillez ajuster le type de sol ou de culture dans la barre latérale.")
        st.stop()

    st.dataframe(filtered, use_container_width=True, height=520)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Exporter les données filtrées (CSV)",
        data=csv_bytes,
        file_name="donnees_capteurs_filtrees.csv",
        mime="text/csv",
        use_container_width=False,
    )

