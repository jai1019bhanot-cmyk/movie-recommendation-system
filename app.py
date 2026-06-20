import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "807d788a66c491e07bb90872fc7cbba8"

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0d0d0d;
    color: #ffffff;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
.stApp { background-color: #0d0d0d; }

.hero {
    background: linear-gradient(135deg, #E50914 0%, #6b0000 100%);
    padding: 2.5rem 2rem;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 3.2rem; font-weight: 900; margin: 0; letter-spacing: -1px; }
.hero p  { font-size: 1rem; color: rgba(255,255,255,0.8); margin-top: 0.4rem; }

.stSelectbox label { color: #E50914 !important; font-weight: 700; }
.stSelectbox > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: #fff !important;
}
.stButton > button {
    background: linear-gradient(90deg, #E50914, #b20710);
    color: white; font-weight: 700; font-size: 1rem;
    border: none; border-radius: 8px;
    padding: 0.65rem 2rem; width: 100%;
}
.stButton > button:hover { background: linear-gradient(90deg, #ff2020, #E50914); }

.section-title {
    font-size: 1.5rem; font-weight: 800;
    border-bottom: 3px solid #E50914;
    padding-bottom: 0.4rem;
    display: inline-block;
    margin-bottom: 1.2rem;
}

.movie-card {
    background: #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2a2a2a;
    transition: transform 0.2s ease, border 0.2s ease;
    height: 100%;
}
.movie-card:hover { transform: translateY(-4px); border: 1px solid #E50914; }
.poster-img { width: 100%; aspect-ratio: 2/3; object-fit: cover; }
.poster-placeholder {
    width: 100%; aspect-ratio: 2/3;
    background: #2a2a2a;
    display: flex; align-items: center; justify-content: center;
    font-size: 3rem;
}
.card-body { padding: 0.8rem; }
.card-title { font-size: 0.9rem; font-weight: 700; color: #fff; margin-bottom: 0.3rem; }
.card-meta  { font-size: 0.75rem; color: #aaa; }
.card-rating { color: #E50914; font-weight: 700; font-size: 0.85rem; }
.badge {
    display: inline-block; background: #2a2a2a;
    color: #E50914; border-radius: 4px;
    padding: 1px 6px; font-size: 0.7rem;
    font-weight: 600; margin: 2px 2px 0 0;
}
.selected-box {
    background: #1a1a1a;
    border-left: 4px solid #E50914;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
}
.selected-box h3 { color: #E50914; margin: 0 0 0.3rem 0; font-size: 1.2rem; }
.selected-box p  { color: #aaa; margin: 0.2rem 0; font-size: 0.82rem; }
hr { border-color: #222; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🎬 Building recommendation engine...")
def load_and_build():
    df = pd.read_csv('movies.csv')
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Overview'].fillna('', inplace=True)
    df['Genre'].fillna('', inplace=True)
    df['Cast'].fillna('', inplace=True)
    df['Keywords'].fillna('', inplace=True)
    df['Director'].fillna('', inplace=True)
    df['Rating'].fillna(df['Rating'].median(), inplace=True)

    def collapse(lst): return [i.replace(' ', '') for i in lst if i]
    df['Genre']    = df['Genre'].apply(lambda x: collapse(str(x).split('|')))
    df['Cast']     = df['Cast'].apply(lambda x: collapse(str(x).split('|')[:3]))
    df['Keywords'] = df['Keywords'].apply(lambda x: collapse(str(x).split('|')))
    df['Director'] = df['Director'].apply(lambda x: str(x).replace(' ', ''))

    df['tags'] = (df['Genre'] + df['Cast'] + df['Keywords'] +
                  df['Director'].apply(lambda x: [x] if x else []) +
                  df['Overview'].apply(lambda x: str(x).split()))
    df['tags'] = df['tags'].apply(lambda x: ' '.join(x).lower())

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    movies_list = df[['MovieID','Title','Genre','Director','Cast','Rating','Year','Overview']].copy()
    movies_list['Genre'] = movies_list['Genre'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    movies_list['Cast']  = movies_list['Cast'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    return movies_list, similarity


@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(title)}"
        r = requests.get(search_url, timeout=5)
        data = r.json()
        if data['results']:
            path = data['results'][0].get('poster_path')
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except:
        pass
    return None


def recommend(movie_title, movies_df, similarity):
    idx = movies_df[movies_df['Title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    results = []
    for i in distances[1:6]:
        row = movies_df.iloc[i[0]]
        results.append({
            'title':    row['Title'],
            'genre':    row['Genre'],
            'director': row['Director'],
            'cast':     row['Cast'],
            'rating':   round(float(row['Rating']), 1),
            'year':     int(row['Year']),
            'overview': row['Overview']
        })
    return results


movies_df, similarity = load_and_build()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Content-Based Movie Recommendation Engine &nbsp;·&nbsp; 2,000+ Movies</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2.5], gap="large")

with left:
    st.markdown("**🎥 Pick a movie you like**")
    selected = st.selectbox("", sorted(movies_df['Title'].values), label_visibility="collapsed")

    sel = movies_df[movies_df['Title'] == selected].iloc[0]
    poster = fetch_poster(selected)

    if poster:
        st.image(poster, use_container_width=True)
    else:
        st.markdown('<div class="poster-placeholder">🎬</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="selected-box">
        <h3>{sel['Title']}</h3>
        <p>📅 {int(sel['Year'])} &nbsp;|&nbsp; ⭐ {round(float(sel['Rating']),1)}</p>
        <p>🎭 {sel['Genre']}</p>
        <p>🎬 {sel['Director']}</p>
        <p>👥 {sel['Cast']}</p>
    </div>
    """, unsafe_allow_html=True)

    btn = st.button("🔍 Find Similar Movies")

with right:
    if btn:
        st.markdown('<div class="section-title">Top 5 Recommendations</div>', unsafe_allow_html=True)
        recs = recommend(selected, movies_df, similarity)

        cols = st.columns(5)
        for i, (col, movie) in enumerate(zip(cols, recs)):
            poster_url = fetch_poster(movie['title'])
            genres_html = "".join([f'<span class="badge">{g.strip()}</span>'
                                   for g in movie['genre'].split(',')[:2]])
            with col:
                if poster_url:
                    st.markdown(f"""
                    <div class="movie-card">
                        <img class="poster-img" src="{poster_url}" />
                        <div class="card-body">
                            <div class="card-title">#{i+1} {movie['title']}</div>
                            <div class="card-rating">⭐ {movie['rating']}</div>
                            <div class="card-meta">📅 {movie['year']} · 🎬 {movie['director'].split()[0] if movie['director'] else ''}</div>
                            <div style="margin-top:0.3rem;">{genres_html}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="poster-placeholder">🎬</div>
                        <div class="card-body">
                            <div class="card-title">#{i+1} {movie['title']}</div>
                            <div class="card-rating">⭐ {movie['rating']}</div>
                            <div class="card-meta">📅 {movie['year']} · 🎬 {movie['director'].split()[0] if movie['director'] else ''}</div>
                            <div style="margin-top:0.3rem;">{genres_html}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:5rem 2rem; color:#444;">
            <div style="font-size:5rem;">🎬</div>
            <div style="font-size:1.2rem; margin-top:1rem;">
                Select a movie on the left<br>
                <span style="color:#E50914; font-weight:700;">and click Find Similar Movies</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#444; font-size:0.78rem; padding:0.8rem 0;">
    Built with Python · Scikit-learn · Streamlit · TMDB API &nbsp;|&nbsp; Content-Based Filtering · Cosine Similarity
</div>
""", unsafe_allow_html=True)
