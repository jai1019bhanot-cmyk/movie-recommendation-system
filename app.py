import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS — Netflix Dark Theme ───────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] {
        background-color: #141414;
        color: #FFFFFF;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stApp { background-color: #141414; }
    .hero {
        background: linear-gradient(135deg, #E50914 0%, #8B0000 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { font-size: 3rem; font-weight: 900; color: #fff; margin: 0; letter-spacing: -1px; }
    .hero p  { font-size: 1.1rem; color: rgba(255,255,255,0.85); margin-top: 0.5rem; }
    .stSelectbox label { color: #E50914 !important; font-weight: 600; font-size: 1rem; }
    .stSelectbox > div > div {
        background-color: #1f1f1f !important;
        border: 1px solid #333 !important;
        color: #fff !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #E50914, #B20710);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2.5rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff1a1a, #E50914);
        transform: scale(1.02);
    }
    .movie-card {
        background: #1f1f1f;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
        transition: border 0.2s ease, transform 0.2s ease;
    }
    .movie-card:hover { border: 1px solid #E50914; transform: translateY(-2px); }
    .movie-rank { font-size: 2rem; font-weight: 900; color: #E50914; line-height: 1; }
    .movie-title { font-size: 1.1rem; font-weight: 700; color: #ffffff; margin: 0.3rem 0 0.2rem 0; }
    .movie-meta { font-size: 0.8rem; color: #aaaaaa; }
    .badge {
        display: inline-block;
        background: #2a2a2a;
        color: #E50914;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
        margin-top: 4px;
    }
    .rating-badge {
        background: #E50914;
        color: #fff;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .selected-box {
        background: #1a1a1a;
        border-left: 4px solid #E50914;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
    }
    .selected-box h3 { color: #E50914; margin: 0 0 0.3rem 0; font-size: 1.3rem; }
    .selected-box p  { color: #aaa; margin: 0; font-size: 0.85rem; }
    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E50914;
        padding-bottom: 0.4rem;
        display: inline-block;
    }
    hr { border-color: #2a2a2a; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load & Build Model from CSV ───────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading movies and building model...")
def load_and_build():
    df = pd.read_csv('movies.csv')

    # Clean
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Overview'].fillna('', inplace=True)
    df['Genre'].fillna('', inplace=True)
    df['Cast'].fillna('', inplace=True)
    df['Keywords'].fillna('', inplace=True)
    df['Director'].fillna('', inplace=True)
    df['Rating'].fillna(df['Rating'].median(), inplace=True)

    # Feature engineering
    def collapse(lst): return [i.replace(' ', '') for i in lst if i]
    df['Genre']    = df['Genre'].apply(lambda x: collapse(str(x).split('|')))
    df['Cast']     = df['Cast'].apply(lambda x: collapse(str(x).split('|')[:3]))
    df['Keywords'] = df['Keywords'].apply(lambda x: collapse(str(x).split('|')))
    df['Director'] = df['Director'].apply(lambda x: str(x).replace(' ', ''))

    df['tags'] = (df['Genre'] + df['Cast'] + df['Keywords'] +
                  df['Director'].apply(lambda x: [x] if x else []) +
                  df['Overview'].apply(lambda x: str(x).split()))
    df['tags'] = df['tags'].apply(lambda x: ' '.join(x).lower())

    # Vectorize & similarity
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    # Clean display columns
    movies_list = df[['MovieID','Title','Genre','Director','Cast','Rating','Year','Overview']].copy()
    movies_list['Genre'] = movies_list['Genre'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    movies_list['Cast']  = movies_list['Cast'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    return movies_list, similarity


movies_df, similarity = load_and_build()


# ── Recommendation Function ───────────────────────────────────────────────────
def recommend(movie_title):
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
            'rating':   row['Rating'],
            'year':     int(row['Year']),
            'overview': row['Overview']
        })
    return results


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Content-Based Movie Recommendation Engine · 2,000+ Movies</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("**🎥 Select a Movie You Like**")
    selected_movie = st.selectbox(
        "Choose a movie",
        sorted(movies_df['Title'].values),
        label_visibility="collapsed"
    )

    sel = movies_df[movies_df['Title'] == selected_movie].iloc[0]
    st.markdown(f"""
    <div class="selected-box">
        <h3>{sel['Title']}</h3>
        <p>📅 {int(sel['Year'])} &nbsp;|&nbsp; 🎭 {sel['Genre']} &nbsp;|&nbsp; ⭐ {sel['Rating']}</p>
        <p style="margin-top:0.4rem;">🎬 {sel['Director']}</p>
        <p>👥 {sel['Cast']}</p>
    </div>
    """, unsafe_allow_html=True)

    recommend_btn = st.button("🔍 Find Similar Movies")

with col_right:
    if recommend_btn:
        st.markdown('<div class="section-title">Top 5 Recommendations</div>', unsafe_allow_html=True)
        recs = recommend(selected_movie)
        for idx, movie in enumerate(recs, 1):
            genres_html = "".join([f'<span class="badge">{g.strip()}</span>' for g in movie['genre'].split(',')])
            st.markdown(f"""
            <div class="movie-card">
                <div style="display:flex; align-items:flex-start; gap:1rem;">
                    <div class="movie-rank">#{idx}</div>
                    <div style="flex:1;">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-meta">
                            📅 {movie['year']} &nbsp;
                            🎬 {movie['director']} &nbsp;
                            <span class="rating-badge">⭐ {movie['rating']}</span>
                        </div>
                        <div style="margin-top:0.4rem;">{genres_html}</div>
                        <div class="movie-meta" style="margin-top:0.5rem;">👥 {movie['cast']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#555;">
            <div style="font-size:4rem;">🎬</div>
            <div style="font-size:1.1rem; margin-top:1rem;">Select a movie and click<br>
            <b style="color:#E50914;">Find Similar Movies</b></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.8rem; padding:1rem 0;">
    Built with ❤️ using Python · Scikit-learn · Streamlit &nbsp;|&nbsp; Content-Based Filtering · Cosine Similarity
</div>
""", unsafe_allow_html=True)
