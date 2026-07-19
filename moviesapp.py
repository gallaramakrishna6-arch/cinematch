import streamlit as st
import pandas as pd
import requests
import difflib
import concurrent.futures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "f478a6f024fb29c80600efeef15b8b66"
OMDB_API_KEY = "4836e5f4"

PLATFORM_LINKS = {
    "Netflix": "https://www.netflix.com/search?q=",
    "Amazon Prime Video": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase=",
    "Amazon Prime Video with Ads": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase=",
    "JioHotstar": "https://www.hotstar.com/in/search?q=",
    "Disney Plus Hotstar": "https://www.hotstar.com/in/search?q=",
    "SonyLIV": "https://www.sonyliv.com/search?q=",
    "ZEE5": "https://www.zee5.com/search?q=",
    "Apple TV": "https://tv.apple.com/search?term=",
    "YouTube": "https://www.youtube.com/results?search_query=",
    "VI movies and tv": "https://www.vimovies.com/",
}

MOOD_GENRE_MAP = {
    'happy': ['Comedy', 'Family', 'Music'], 'sad': ['Drama'], 'love': ['Romance'],
    'romantic': ['Romance'], 'romance': ['Romance'], 'scary': ['Horror'], 'horror': ['Horror'],
    'thrill': ['Thriller'], 'excited': ['Action', 'Adventure'], 'bored': ['Comedy', 'Action'],
    'motivate': ['Drama'], 'inspire': ['Drama'], 'relax': ['Family', 'Animation'],
    'cry': ['Drama'], 'laugh': ['Comedy'], 'funny': ['Comedy'],
}

def detect_mood_genres(query):
    query_lower = query.lower()
    detected = set()
    for keyword, genres in MOOD_GENRE_MAP.items():
        if keyword in query_lower:
            detected.update(genres)
    return list(detected)

@st.cache_data
def load_data():
    df = pd.read_csv("movies_data.csv")
    df['genres'] = df['genres'].apply(eval)
    return df

@st.cache_resource
def build_similarity(_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(_df['tags'].fillna(''))
    return cosine_similarity(vectors)

df = load_data()
similarity = build_similarity(df)

@st.cache_data(ttl=3600)
def get_watch_info(movie_id, country='IN'):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        providers = data.get('results', {}).get(country, {})
        flatrate = providers.get('flatrate', [])
        return [p['provider_name'] for p in flatrate] if flatrate else []
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        budget = data.get('budget', 0)
        revenue = data.get('revenue', 0)
        imdb_id = data.get('imdb_id', None)
        if budget == 0 or revenue == 0:
            status = "సమాచారం లేదు"
        elif revenue >= budget * 2:
            status = "🔥 Blockbuster"
        elif revenue > budget:
            status = "✅ Hit"
        else:
            status = "❌ Flop"
        return {"budget": budget, "revenue": revenue, "status": status, "imdb_id": imdb_id}
    except Exception:
        return {"budget": 0, "revenue": 0, "status": "సమాచారం దొరకలేదు", "imdb_id": None}

@st.cache_data(ttl=3600)
def get_director(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        crew = data.get('crew', [])
        directors = [c['name'] for c in crew if c.get('job') == 'Director']
        return ", ".join(directors) if directors else "N/A"
    except Exception:
        return "N/A"

@st.cache_data(ttl=3600)
def get_imdb_rating(imdb_id):
    if not imdb_id:
        return "N/A"
    url = f"https://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data.get('imdbRating', 'N/A')
    except Exception:
        return "N/A"

@st.cache_data(ttl=3600)
def get_all_details(movie_id):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_details = executor.submit(get_movie_details, movie_id)
        f_director = executor.submit(get_director, movie_id)
        f_platforms = executor.submit(get_watch_info, movie_id)
        details = f_details.result()
        director = f_director.result()
        platforms = f_platforms.result()
    imdb_rating = get_imdb_rating(details['imdb_id'])
    return details, director, platforms, imdb_rating

def format_money(amount):
    if amount == 0:
        return "N/A"
    return f"${amount:,}"

def recommend(movie_title, top_n=5):
    matches = df[df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return None, []
    idx = matches.index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:top_n+1]
    results = [df.iloc[i] for i, score in distances]
    return df.iloc[idx], results

def show_search_result_card(row):
    if pd.notna(row['poster_path']):
        st.image(f"https://image.tmdb.org/t/p/w200{row['poster_path']}")
    else:
        st.write("🎬 (poster లేదు)")
    if st.button(row['title'], key=f"pick_{row['title']}_{row['id']}", use_container_width=True):
        st.session_state.selected_movie = row['title']
        st.session_state.page_num = 0
        st.rerun()
    st.caption(f"⭐ {row['rating_5']}/5 | {', '.join(row['genres'])}")

def show_main_movie_panel(movie):
    """ప్రధాన movie కి - Google Knowledge Panel style, అన్ని వివరాలు వెంటనే"""
    with st.spinner("వివరాలు తీసుకువస్తున్నా..."):
        details, director, platforms, imdb_rating = get_all_details(movie['id'])

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            if pd.notna(movie['poster_path']):
                st.image(f"https://image.tmdb.org/t/p/w300{movie['poster_path']}", use_container_width=True)
        with c2:
            movie_year = movie['release_date'][:4] if pd.notna(movie['release_date']) and movie['release_date'] else "N/A"
            st.markdown(f"## {movie['title']}")
            st.caption(f"{movie_year} · {movie['language']} · {', '.join(movie['genres'])}")

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("⭐ TMDB", f"{movie['rating_5']}/5")
            g2.metric("🎯 IMDb", f"{imdb_rating}/10")
            g3.metric("💰 Budget", format_money(details['budget']))
            g4.metric("💵 Collections", format_money(details['revenue']))

            st.write(f"**🎬 Director:** {director}")
            st.write(f"**📊 Status:** {details['status']}")
            st.write(movie['overview'])

            if platforms:
                st.write("**📺 ఎక్కడ చూడొచ్చు:**")
                p_cols = st.columns(len(platforms))
                for i, p in enumerate(platforms):
                    link = PLATFORM_LINKS.get(p, "https://www.google.com/search?q=" + p.replace(" ", "+"))
                    with p_cols[i]:
                        st.link_button(p, link + movie['title'].replace(" ", "+"))
            else:
                st.write("**📺 ఏ OTT platform లో లేదు**")

            review = st.text_area("నీ review/notes రాయి", key=f"review_main_{movie['id']}")
            if st.button("💾 Review Save చేయి", key=f"save_main_{movie['id']}"):
                st.success("Review save అయ్యింది!")

def show_movie_card(movie):
    """Similar movies కి - fast loading, lazy details"""
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            if pd.notna(movie['poster_path']):
                st.image(f"https://image.tmdb.org/t/p/w200{movie['poster_path']}")
        with c2:
            movie_year = movie['release_date'][:4] if pd.notna(movie['release_date']) and movie['release_date'] else "N/A"
            st.markdown(f"### {movie['title']} ({movie['language']}, {movie_year})")
            st.write(f"⭐ TMDB Rating: {movie['rating_5']}/5")
            st.write(f"🎭 Genres: {', '.join(movie['genres'])}")
            st.write(movie['overview'][:150] + "...")

            if st.button("👁️ ఈ movie వివరాలు చూడు", key=f"view_{movie['id']}"):
                st.session_state.selected_movie = movie['title']
                st.rerun()

# ------------------ UI ------------------
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
input[type="text"] { font-size: 18px !important; padding: 10px !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("👤 About")
    st.write("**Developer:** Galla Ramakrishna")
    st.write("Data Science / CSE Student")

st.title("🎬 CineMatch")
st.write("English, Hindi, Telugu, Malayalam movies | Rating 2.5/5+ | with OTT info")

col1, col2 = st.columns(2)
with col1:
    lang_filter = st.multiselect("భాష select చేయి", options=df['language'].unique(), default=list(df['language'].unique()))
with col2:
    all_genres = sorted(set(g for genres in df['genres'] for g in genres if g))
    genre_filter = st.multiselect("Genre select చేయి (optional)", options=all_genres)

filtered_df = df[df['language'].isin(lang_filter)]
if genre_filter:
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda g: any(genre in g for genre in genre_filter))]

if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'page_num' not in st.session_state:
    st.session_state.page_num = 0

if st.session_state.selected_movie:
    if st.button("🔙 వెనక్కి వెళ్ళు (Movies list కి)"):
        st.session_state.selected_movie = None
        st.rerun()
    st.divider()

    selected, results = recommend(st.session_state.selected_movie)

    if selected is None:
        st.error("Movie దొరకలేదు")
    else:
        show_main_movie_panel(selected)

        st.divider()
        st.subheader("🎯 Similar Movies:")

        for movie in results:
            show_movie_card(movie)

else:
    col_a, col_b = st.columns(2)
    with col_a:
        search_query = st.text_input("🔍 Movie పేరు టైప్ చేయి")
    with col_b:
        mood_query = st.text_input("💭 నీ Mood/Feeling చెప్పు (ఉదా: 'happy ga unnanu', 'love movie kavali')")

    if mood_query:
        mood_genres = detect_mood_genres(mood_query)
        if mood_genres:
            st.info(f"🎭 నీ mood బట్టి, ఈ genres లో సూచిస్తున్నా: {', '.join(mood_genres)}")
            mood_movies = filtered_df[filtered_df['genres'].apply(lambda g: any(genre in g for genre in mood_genres))]
            mood_movies = mood_movies.sort_values('rating_5', ascending=False)
            st.write(f"**💝 నీకు నచ్చే Movies ({len(mood_movies)} దొరికాయి):**")
            rows = list(mood_movies.iterrows())
            for i in range(0, len(rows), 4):
                row_chunk = rows[i:i+4]
                cols = st.columns(4)
                for j, (_, m) in enumerate(row_chunk):
                    with cols[j]:
                        show_search_result_card(m)
        else:
            st.warning("ఈ mood కి సరిపోయే genre దొరకలేదు, వేరే పదం (happy, sad, love, scary, etc.) try చేయి")
        st.divider()

    elif search_query:
        exact_matches_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
        if exact_matches_df.empty:
            all_titles = filtered_df['title'].tolist()
            close_titles = difflib.get_close_matches(search_query, all_titles, n=9, cutoff=0.6)
            close_matches_df = filtered_df[filtered_df['title'].isin(close_titles)]
        else:
            close_matches_df = filtered_df.iloc[0:0]

        combined_df = pd.concat([exact_matches_df, close_matches_df]).drop_duplicates(subset='id')

        if not combined_df.empty:
            st.write(f"**🔎 {len(combined_df)} results దొరికాయి — క్రింద click చేయి:**")
            rows = list(combined_df.iterrows())
            for i in range(0, len(rows), 4):
                row_chunk = rows[i:i+4]
                cols = st.columns(4)
                for j, (_, m) in enumerate(row_chunk):
                    with cols[j]:
                        show_search_result_card(m)
        else:
            st.warning("ఏ movie దొరకలేదు, వేరే పదం try చేయి")
        st.divider()

    else:
        st.write(f"### 📋 మొత్తం Movies: {len(filtered_df)}")
        PER_PAGE = 12
        total_pages = max((len(filtered_df) - 1) // PER_PAGE + 1, 1)
        start = st.session_state.page_num * PER_PAGE
        end = start + PER_PAGE
        page_movies = filtered_df.iloc[start:end]

        rows = list(page_movies.iterrows())
        for i in range(0, len(rows), 4):
            row_chunk = rows[i:i+4]
            cols = st.columns(4)
            for j, (_, m) in enumerate(row_chunk):
                with cols[j]:
                    show_search_result_card(m)

        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.session_state.page_num > 0:
                if st.button("« Previous"):
                    st.session_state.page_num -= 1
                    st.rerun()
        with nav2:
            st.write(f"Page {st.session_state.page_num + 1} / {total_pages}")
        with nav3:
            if st.session_state.page_num < total_pages - 1:
                if st.button("Next »"):
                    st.session_state.page_num += 1
                    st.rerun()