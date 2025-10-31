# app/pages/06_Admin.py
from __future__ import annotations  # must stay at the very top

import os
import time
from pathlib import Path
import pandas as pd
import streamlit as st

from app.app_utils.ui import use_global_ui
from mangetamain.core.app_logging import setup_logging, get_logger  # your logging module

# ---------- UI header ----------
use_global_ui(
    "Mangetamain — 🛠️ Admin: Log Center",
    logo="assets/mangetamain-logo.jpg",
    logo_size_px=90,
    round_logo=True,
    subtitle=None,
    wide=True,
)

# ---------- Helpers ----------
def _rerun() -> None:
    """Force a page reload (compatible with old/new Streamlit versions)."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        try:
            st.experimental_rerun()  # fallback for older versions
        except Exception:
            st.warning("Please manually refresh the page (Ctrl/Cmd + R).")


def find_root(anchor: Path) -> Path:
    """Find the project root by walking up to detect logs/, .git/, or pyproject.toml."""
    p = anchor.resolve()
    for cand in [p, *p.parents]:
        if (cand / "logs").exists() or (cand / "pyproject.toml").exists() or (cand / ".git").exists():
            return cand
    return p


ROOT = find_root(Path(__file__))
# Configure or reconfigure logging on every rerun (creates /logs if missing)
setup_logging(ROOT)

LOG_DIR = ROOT / "logs"
APP_LOG = LOG_DIR / "app.log"
ERR_LOG = LOG_DIR / "error.log"

# ---------- Sidebar / Options ----------
with st.sidebar:
    st.header("Options")
    autorefresh = st.checkbox("Auto-refresh", value=True, help="Refresh every 5 seconds")
    if autorefresh:
        INTERVAL_S = 5
        now = time.time()
        key = "_log_autorefresh_ts"
        if key not in st.session_state:
            st.session_state[key] = now
        elif now - st.session_state[key] >= INTERVAL_S:
            st.session_state[key] = now
            _rerun()

    file_choice = st.radio("Log file", ["app.log", "error.log"])
    level_filter = st.multiselect(
        "Levels", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default=[]
    )
    query = st.text_input("Search (contains):", placeholder="keyword, module, etc.")
    max_lines = st.slider("Last lines", min_value=200, max_value=10000, value=2000, step=200)

# ---------- I/O utilities ----------
def read_log_bytes(path: Path) -> bytes:
    """Return raw bytes of a log file (or b'' if not found)."""
    if not path.exists():
        return b""
    with open(path, "rb") as f:
        return f.read()


def last_lines(text: str, n: int = 2000) -> list[str]:
    """Return the last n lines from a text."""
    if not text:
        return []
    return text.splitlines()[-n:]


def parse_levels(lines: list[str]) -> pd.DataFrame:
    """Parse log lines of the form 'YYYY-mm-dd HH:MM:SS,sss | LEVEL | logger | message'."""
    recs = []
    for ln in lines:
        parts = [p.strip() for p in ln.split("|", 3)]
        if len(parts) == 4:
            ts, lvl, logger, msg = parts
        else:
            ts, lvl, logger, msg = "", "", "", ln
        recs.append({"time": ts, "level": lvl, "logger": logger, "message": msg})
    return pd.DataFrame.from_records(recs)


# ---------- Reading & Filtering ----------
cur_log = APP_LOG if file_choice == "app.log" else ERR_LOG

raw = read_log_bytes(cur_log)  # raw bytes (for download)
txt = raw.decode("utf-8", errors="replace")  # decoded text (for display)
lines = last_lines(txt, n=max_lines)  # last lines

df = parse_levels(lines)
if level_filter:
    df = df[df["level"].isin(level_filter)]
if query:
    q = query.lower()
    df = df[df.apply(lambda r: q in " ".join(map(str, r.values)).lower(), axis=1)]

# ---------- Header + Actions ----------
size_txt = f"{cur_log.stat().st_size/1024:.1f} KB" if cur_log.exists() else "—"
mtime_txt = (
    time.strftime("%H:%M:%S", time.localtime(cur_log.stat().st_mtime))
    if cur_log.exists()
    else "—"
)

colA, colB, colC, colD = st.columns([2, 1, 1, 1])
colA.subheader(f"📄 {file_choice}")
colB.metric("Size", size_txt)
colC.metric("Displayed lines", len(df))
colD.metric("Last modified", mtime_txt)

btn_cols = st.columns(3)
with btn_cols[0]:
    st.download_button(
        "⬇️ Download",
        data=raw,  # send raw bytes
        file_name=file_choice,
        mime="text/plain; charset=utf-8",
    )
with btn_cols[1]:
    if st.button("🧹 Clear file", type="secondary"):
        cur_log.parent.mkdir(parents=True, exist_ok=True)
        with open(cur_log, "w", encoding="utf-8"):
            pass
        get_logger().warning("Log file cleared: %s", cur_log.name)
        st.success(f"{file_choice} cleared.")
        _rerun()
with btn_cols[2]:
    if st.button("🧪 Write test log (INFO)"):
        logger = get_logger()
        logger.info("Test log generated from Admin page.")
        # flush handlers to ensure write on disk
        for h in getattr(logger, "handlers", []):
            try:
                h.flush()
            except Exception:
                pass
        st.toast("Test log written.")
        # _rerun()  # uncomment if you want immediate refresh

# ---------- Display ----------
if len(lines) == 0:
    st.info(
        "No lines to display (file is empty or missing). "
        "Click **Write test log** to ensure logging works."
    )
st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("View raw content"):
    st.code("\n".join(lines), language="text")
