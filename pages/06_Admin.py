import streamlit as st
from app_utils.io import load_data
from app_utils.ui import use_global_ui
use_global_ui("Mangetamain — 🛠️ Admin",     logo="image/image.jpg",
    
    logo_size_px=90,
    round_logo=True, subtitle=None, wide=True)

#st.title("🛠️ Admin")
st.write("Niveau de logs (visuel uniquement pour l'instant) :")
st.radio("Logs", ["INFO","WARNING","ERROR"], horizontal=True)
df = load_data()
st.write("Statistiques rapides")
st.write({"rows": len(df), "cols": df.shape[1]})
st.success("Cache actif : load_data()")
