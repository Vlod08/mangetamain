from __future__ import annotations
from pathlib import Path
import sys
import streamlit as st
import os
from pathlib import Path
import time
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from app_utils.io import load_data
from app_utils.ui import use_global_ui
use_global_ui("Mangetamain — 🛠️ Admin",     logo="image/image.jpg",
    
    logo_size_px=90,
    round_logo=True, subtitle=None, wide=True)

# app/pages/06_Admin.py


# --- Localiser la racine du projet (dossier qui contient /logs)
def find_root(anchor: Path) -> Path:
    p = anchor.resolve()
    for cand in [p, *p.parents]:
        if (cand / "logs").exists():
            return cand
    return p

ROOT = find_root(Path(__file__))
LOG_DIR = ROOT / "logs"
APP_LOG = LOG_DIR / "log"
ERR_LOG = LOG_DIR / "error.log"

st.set_page_config(page_title="Admin — Logs", layout="wide", initial_sidebar_state="expanded")
st.title("🛠️ Admin — Centre de logs")

if not LOG_DIR.exists():
    st.info("Aucun dossier **logs/** trouvé. Les logs seront créés automatiquement dès qu'un événement est journalisé.")
    st.stop()

# --- Barre d'outils
with st.sidebar:
    st.header("Options")
    autorefresh = st.checkbox("Auto-refresh", value=True, help="Rafraîchit toutes les 5s")
    if autorefresh:
        st.experimental_rerun  # (no-op docstring to keep lints happy)
        st.autorefresh(interval=5000, key="log_autorefresh")

    file_choice = st.radio("Fichier", ["log", "error.log"])
    level_filter = st.multiselect("Niveaux", ["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"], default=[])
    query = st.text_input("Recherche (contient) :", placeholder="mot-clé, module, etc.")
    max_lines = st.slider("Dernières lignes", min_value=200, max_value=10000, value=2000, step=200)

# --- Fonctions utilitaires
def tail_lines(path: Path, n: int = 2000) -> list[str]:
    if not path.exists():
        return []
    # lecture efficace depuis la fin
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        buf = bytearray()
        lines = []
        while f.tell() > 0 and len(lines) < n:
            step = min(4096, f.tell())
            f.seek(-step, os.SEEK_CUR)
            buf.extend(f.read(step))
            f.seek(-step, os.SEEK_CUR)
            parts = buf.split(b"\n")
            # gardons seulement les lignes complètes
            lines = parts[-1].splitlines() + lines
            buf = parts[0]
        text = b"\n".join(lines[-n:]).decode("utf-8", errors="replace")
        return text.splitlines()

def parse_levels(lines: list[str]) -> pd.DataFrame:
    # Format par défaut: "YYYY-mm-dd HH:MM:SS,sss | LEVEL | logger | message"
    recs = []
    for ln in lines:
        parts = [p.strip() for p in ln.split("|", 3)]
        if len(parts) == 4:
            ts, lvl, logger, msg = parts
        else:
            ts, lvl, logger, msg = "", "", "", ln
        recs.append({"time": ts, "level": lvl, "logger": logger, "message": msg})
    return pd.DataFrame.from_records(recs)

# --- Choix du fichier
cur_log = APP_LOG if file_choice == "log" else ERR_LOG
lines = tail_lines(cur_log, n=max_lines)
df = parse_levels(lines)

# --- Filtres
if level_filter:
    df = df[df["level"].isin(level_filter)]
if query:
    q = query.lower()
    df = df[df.apply(lambda r: q in " ".join(map(str, r.values)).lower(), axis=1)]

# --- Header + actions
colA, colB, colC, colD = st.columns([2,1,1,1])
colA.subheader(f"📄 {file_choice}")
colB.metric("Taille", f"{cur_log.stat().st_size/1024:.1f} Ko" if cur_log.exists() else "—")
colC.metric("Lignes affichées", len(df))
colD.metric("Modifié", time.strftime("%H:%M:%S", time.localtime(cur_log.stat().st_mtime)) if cur_log.exists() else "—")

btn_cols = st.columns(3)
with btn_cols[0]:
    st.download_button("⬇️ Télécharger", data="\n".join(lines), file_name=file_choice, mime="text/plain")
with btn_cols[1]:
    if st.button("🧹 Purger le fichier", type="secondary"):
        open(cur_log, "w", encoding="utf-8").close()
        st.success(f"{file_choice} vidé."); st.experimental_rerun()
with btn_cols[2]:
    if st.button("🧪 Écrire un log de test (INFO)"):
        from mangetamain.core.logging import get_logger
        get_logger().info("Test log depuis la page Admin.")
        st.toast("Log de test écrit.")

# --- Affichage
st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("Voir brut"):
    st.code("\n".join(lines), language="text")
