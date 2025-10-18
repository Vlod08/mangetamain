import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
from app_utils.io import load_data
from app_utils.ui import use_global_ui
use_global_ui("Mangetamain —  Clustering (TF-IDF + KMeans)",    logo="image/image.jpg",
    
    logo_size_px=90,
    round_logo=True, subtitle=None, wide=True)


#st.title("🧪 Clustering (TF-IDF + KMeans)")

df = load_data().copy()
df["text"] = (df["name"].fillna("") + " " + df["ingredients"].astype(str) + " " + df["description"].fillna("")).str.lower()

k = st.slider("Nombre de clusters (k)", 3, 12, 6)
maxf = st.slider("Max features TF-IDF", 2000, 20000, 8000, step=1000)

@st.cache_resource(show_spinner=True)
def build_model(corpus: pd.Series, k: int, maxf: int):
    tfidf = TfidfVectorizer(max_features=maxf, stop_words="english")
    X = tfidf.fit_transform(corpus)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
    return km, tfidf

km, tfidf = build_model(df["text"], k, maxf)
df["cluster"] = km.labels_

# 2D via PCA rapide (évite UMAP si temps court)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
XY = svd.fit_transform(tfidf.transform(df["text"]))
df["x"], df["y"] = XY[:,0], XY[:,1]

fig = px.scatter(df.sample(min(4000, len(df))), x="x", y="y", color="cluster", hover_name="name", opacity=0.7)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top mots par cluster")
def top_terms_per_cluster(df, tfidf, km, topn=8):
    import numpy as np
    terms = tfidf.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    rows = []
    for i in range(km.n_clusters):
        words = [terms[ind] for ind in order_centroids[i, :topn]]
        rows.append({"cluster": i, "top_terms": ", ".join(words)})
    return pd.DataFrame(rows)

st.dataframe(top_terms_per_cluster(df, tfidf, km), use_container_width=True, hide_index=True)
