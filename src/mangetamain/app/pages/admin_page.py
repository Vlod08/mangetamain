# app/pages/admin_page.py
from __future__ import annotations
from pathlib import Path
import streamlit as st
import os
import time
import pandas as pd
import logging

from app.app_utils.ui import use_global_ui
from config import ROOT_DIR


logger = logging.getLogger(__name__)


def app():

    use_global_ui(
        "Mangetamain — 🛠️ Admin",
        logo="assets/mangetamain-logo.jpg",
        logo_size_px=90,
        round_logo=True,
        subtitle=None,
        wide=True,
    )

    st.set_page_config(
        page_title="Admin — Logs", layout="wide", initial_sidebar_state="expanded"
    )
    st.title("🛠️ Admin — Logs center")

    logs_dir = ROOT_DIR / "logs"
    app_log = logs_dir / "log"
    err_log = logs_dir / "error.log"

    if not logs_dir.exists():
        st.info(
            "No directory **logs/** found. Logs will be created automatically as soon as an event is logged."
        )
        st.stop()

    # --- UI Sidebar
    with st.sidebar:
        st.header("Options")
        autorefresh = st.checkbox(
            "Auto-refresh", value=True, help="Refresh every 5 seconds"
        )
        if autorefresh:
            st.experimental_rerun  # (no-op docstring to keep lints happy)
            st.autorefresh(interval=5000, key="log_autorefresh")

        file_choice = st.radio("File", ["log", "error.log"])
        level_filter = st.multiselect(
            "Levels", ["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"], default=[]
        )
        query = st.text_input("Search (contains):", placeholder="keyword, module, etc.")
        max_lines = st.slider(
            "Last lines", min_value=200, max_value=10000, value=2000, step=200
        )

    # --- File selection
    cur_log = app_log if file_choice == "log" else err_log
    lines = tail_lines(cur_log, n=max_lines)
    df = parse_levels(lines)

    # --- Filters
    if level_filter:
        df = df[df["level"].isin(level_filter)]
    if query:
        q = query.lower()
        df = df[df.apply(lambda r: q in " ".join(map(str, r.values)).lower(), axis=1)]

    # --- Header + actions
    colA, colB, colC, colD = st.columns([2, 1, 1, 1])
    colA.subheader(f"📄 {file_choice}")
    colB.metric(
        "Size", f"{cur_log.stat().st_size/1024:.1f} Ko" if cur_log.exists() else "—"
    )
    colC.metric("Displayed lines", len(df))
    colD.metric(
        "Modified",
        (
            time.strftime("%H:%M:%S", time.localtime(cur_log.stat().st_mtime))
            if cur_log.exists()
            else "—"
        ),
    )

    btn_cols = st.columns(3)
    with btn_cols[0]:
        st.download_button(
            "⬇️ Download",
            data="\n".join(lines),
            file_name=file_choice,
            mime="text/plain",
        )
    with btn_cols[1]:
        if st.button("🧹 Purge file", type="secondary"):
            open(cur_log, "w", encoding="utf-8").close()
            st.success(f"{file_choice} emptied.")
            st.experimental_rerun()
    with btn_cols[2]:
        if st.button("🧪 Write test log (INFO)"):
            logger.info("Test log from Admin page.")
            st.toast("Test log written.")

    # --- Display
    st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("Raw log content"):
        st.code("\n".join(lines), language="text")


# --- Utilities
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
    # Default format: "YYYY-mm-dd HH:MM:SS,sss | LEVEL | logger | message"
    recs = []
    for ln in lines:
        parts = [p.strip() for p in ln.split("|", 3)]
        if len(parts) == 4:
            ts, lvl, logger, msg = parts
        else:
            ts, lvl, logger, msg = "", "", "", ln
        recs.append({"time": ts, "level": lvl, "logger": logger, "message": msg})
    return pd.DataFrame.from_records(recs)


if __name__ == "__main__":
    app()
