import streamlit as st
import pandas as pd
from engine import build_tfidf, recommend_with_weights

st.set_page_config(page_title="Smart Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("movies_enriched.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        df = pd.read_csv("movies_clean.csv", encoding="utf-8-sig")
        import numpy as np
        np.random.seed(42)
        df["rating"] = np.random.uniform(5.0, 9.5, len(df))
        df["vote_count"] = np.random.randint(100, 500000, len(df))
        df["poster"] = "N/A"
        st.info("💡 Note: Using base data with simulated ratings.")
        
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    df["poster"] = df["poster"].fillna("N/A")
    vectorizer, matrix = build_tfidf(df)
    return df, vectorizer, matrix

try:
    df, vectorizer, tfidf_matrix = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ================= Sidebar =================
st.sidebar.header("🎯 Your Preferences")

st.sidebar.subheader("1. Genre Preferences")
st.sidebar.caption("Positive = like, Negative = dislike")
all_genres = ["science fiction", "action", "comedy", "romance", "horror", "thriller", "crime", "drama"]
genre_weights = {}
for g in all_genres:
    if g in df["primary_genre"].values:
        genre_weights[g] = st.sidebar.slider(f"{g.capitalize()}", -1.0, 1.0, 0.0, 0.1)

st.sidebar.subheader("2. Release Year")
year_mode_ui = st.sidebar.radio("Year Restriction Mode", ["No restriction", "Prioritize recent", "Strictly limit years"])
if year_mode_ui == "Strictly limit years":
    year_range = st.sidebar.slider("Select year range", 1900, 2024, (2015, 2024))
    recent_boost = 0.0
    y_mode = "hard"
elif year_mode_ui == "Prioritize recent":
    recent_boost = st.sidebar.slider("Recency boost", 0.0, 1.0, 0.5, 0.1)
    year_range = (1900, 2024)
    y_mode = "soft"
else:
    recent_boost = 0.0
    year_range = (1900, 2024)
    y_mode = "off"

st.sidebar.subheader("3. Popularity")
st.sidebar.caption("← Niche gems | Mainstream →")
popularity_weight = st.sidebar.slider(" ", -1.0, 1.0, 0.0, 0.1)

st.sidebar.subheader("4. Mood Keywords")
st.sidebar.caption("e.g., mind-bending, romantic, funny")
mood_text = st.sidebar.text_input("Enter keywords", "")

top_n = st.sidebar.number_input("Number of recommendations", 5, 30, 10)

# ================= Main UI =================
st.title("🎬 Smart Movie Recommender")
st.markdown("Adjust the sliders on the left to customize your watchlist!")

if "recommend_results" not in st.session_state:
    st.session_state.recommend_results = None

if st.sidebar.button("🚀 Start Recommendation", type="primary"):
    config = {
        "genre_weights": genre_weights,
        "year_mode": y_mode,
        "year_range": year_range,
        "recent_boost": recent_boost,
        "popularity_weight": popularity_weight,
        "mood_text": mood_text,
        "top_n": top_n
    }

    with st.spinner("Filtering movies for you..."):
        st.session_state.recommend_results = recommend_with_weights(df, tfidf_matrix, vectorizer, config)

if st.session_state.recommend_results is not None:
    results = st.session_state.recommend_results
    
    if results.empty:
        st.warning("No movies found matching your criteria. Try relaxing the conditions!")
    else:
        st.success(f"Found {len(results)} movies for you!")
        
        # 👇 Blind Box
        if st.button("🎲 Can't decide? Pick one for me!"):
            random_movie = results.sample(1).iloc[0]
            st.balloons() 
            
            title = random_movie['title']
            year = random_movie['year']
            genre = random_movie['genre'].upper()
            reason = random_movie['reason']
            plot = str(random_movie['plot'])[:300] + "..."
            
            # Poster Fallback
            poster = random_movie.get('poster', 'N/A')
            if poster == 'N/A' or pd.isna(poster) or not str(poster).startswith("http"):
                poster = f"https://picsum.photos/seed/{random_movie['title']}/200/300"
            
            card_html = f"""
            <style>
                .blindbox-card {{
                    background: linear-gradient(135deg, #1e1e2f, #2a2a40);
                    border-radius: 16px;
                    padding: 24px;
                    color: white;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    border: 1px solid #3c3c5a;
                    margin-top: 10px;
                    margin-bottom: 30px;
                    display: flex;
                    gap: 20px;
                }}
                .blindbox-card img {{
                    width: 150px;
                    height: 220px;
                    object-fit: cover;
                    border-radius: 8px;
                }}
                .blindbox-content {{ flex: 1; }}
                .blindbox-card h2 {{ margin-top: 0; color: #ffd700; font-size: 26px; }}
                .blindbox-card .year {{ font-size: 16px; color: #a0a0b0; font-weight: normal; }}
                .custom-tag {{ display: inline-block; padding: 6px 12px; border-radius: 20px; font-size: 13px; font-weight: bold; margin-right: 10px; margin-bottom: 10px; }}
                .tag-genre {{ background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; border: 1px solid #4CAF50; }}
                .tag-reason {{ background-color: rgba(33, 150, 243, 0.2); color: #2196F3; border: 1px solid #2196F3; }}
                .blindbox-card p {{ color: #cccccc; font-size: 15px; line-height: 1.6; }}
            </style>
            
            <div class="blindbox-card">
                <img src="{poster}" alt="Poster">
                <div class="blindbox-content">
                    <h2>✨ {title} <span class="year">({year})</span></h2>
                    <div>
                        <span class="custom-tag tag-genre">🎬 {genre}</span>
                        <span class="custom-tag tag-reason">💡 {reason}</span>
                    </div>
                    <p>{plot}</p>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        
        st.divider() 
        
        # 👇 List Cards
        for _, row in results.iterrows():
            with st.container(border=True):
                col_img, col_text, col_data = st.columns([1, 3, 1.5]) 

                with col_img:
                    poster = row.get('poster', 'N/A')
                    if poster != 'N/A' and pd.notna(poster) and str(poster).startswith("http"):
                        st.image(poster, use_container_width=True)
                    else:
                        fallback_url = f"https://picsum.photos/seed/{row['title']}/200/300"
                        st.image(fallback_url, use_container_width=True)
               
                with col_text:
                    st.markdown(f"### {row['title']} ({row['year']})")
                    st.markdown(f"**Genre**: {row['genre'].capitalize()}")
                    st.markdown(f"**Reason**: `{row['reason']}`")
                    st.text(row['plot'])
                    
                with col_data:
                    st.metric("Match Score", f"{row['score']}")
                    rating_display = f"⭐ {row['rating']}/10" if row['rating'] > 0 else "No rating"
                    st.metric("IMDb Rating", rating_display, f"{row['votes']:,} votes")
