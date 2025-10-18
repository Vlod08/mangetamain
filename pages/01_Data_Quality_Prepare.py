import os
import streamlit as st
import pandas as pd
from app_utils.io import load_data, validate_schema, artifact_path
from app_utils.ui import use_global_ui
use_global_ui("Mangetamain —  Données, Qualité & Préparation",     logo="image/image.jpg",
    subtitle="Vérification du schéma, qualité des données et étapes de préparation.",wide=True,
    logo_size_px=90,
    round_logo=True)


#st.title("")

# (Optionnel) bouton pour (re)générer l'artefact propre
with st.expander("Pipeline RAW → CLEAN (artefact)", expanded=False):
    st.caption("Exécute le script de prétraitement pour générer/mettre à jour l'artefact.")
    if st.button("🧹 Régénérer l’artefact propre"):
        code = os.system("poetry run python scripts/preprocess_dataset.py")
        if code == 0:
            st.success(f"Artefact régénéré : {artifact_path()}. Recharge la page (Ctrl/Cmd+R).")
        else:
            st.error("Échec de génération — vérifier la console / chemins.")

df = load_data()
report = validate_schema(df)

# Bandeau d'état du schéma
if report["ok"]:
    st.success("Schéma OK — colonnes minimales présentes.")
else:
    st.warning(f"Colonnes manquantes (affichage dégradé) : {', '.join(report['missing'])}")

# KPIs qualité
c1,c2,c3,c4 = st.columns(4)
c1.metric("Lignes", f"{report['rows']:,}")
c2.metric("Colonnes", report["cols"])
c3.metric("% minutes manquantes", f"{(df['minutes'].isna().mean()*100):.1f}%" if "minutes" in df else "—")
c4.metric("submitted parseable", "✅" if "submitted" in df and df["submitted"].notna().any() else "—")

# Schéma & complétude
st.subheader("Aperçu du schéma")
schema = pd.DataFrame({
    "colonne": df.columns,
    "type": [str(t) for t in df.dtypes],
    "% manquants": (df.isna().mean()*100).round(1)
})
st.dataframe(schema, use_container_width=True, hide_index=True)

# Manquants top 10
st.subheader("Qualité — manquants (top 10)")
miss = df.isna().mean().sort_values(ascending=False).head(10)*100
st.bar_chart(miss)

# Préparation (lecture seule)
st.subheader("Préparation effectuée (lecture seule)")
st.checkbox("Parsing des tags", value="tags" in df.columns, disabled=True)
st.checkbox("Split nutrition → colonnes", value=bool(set(df.columns) & {"calories","sodium","protein"}), disabled=True)
st.checkbox("Features n_steps / n_ingredients", value={"n_steps","n_ingredients"}.issubset(df.columns), disabled=True)

# Échantillon téléchargeable
st.subheader("Échantillon")
st.download_button(
    "⬇️ Télécharger un échantillon (CSV)",
    data=df.sample(min(500, len(df))).to_csv(index=False).encode("utf-8"),
    file_name="sample_recipes.csv",
    mime="text/csv"
)
