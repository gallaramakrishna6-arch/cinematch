import streamlit as st
import pandas as pd
import requests
import difflib
import concurrent.futures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

API_KEY = st.secrets["API_KEY"]
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

PLATFORM_LINKS = {
    "Netflix": ("🅽", "https://www.netflix.com/search?q="),
    "Amazon Prime Video": ("▶️", "https://www.primevideo.com/search/ref=atv_nb_sr?phrase="),
    "Amazon Prime Video with Ads": ("▶️", "https://www.primevideo.com/search/ref=atv_nb_sr?phrase="),
    "JioHotstar": ("⭐", "https://www.hotstar.com/in/search?q="),
    "Disney Plus Hotstar": ("⭐", "https://www.hotstar.com/in/search?q="),
    "SonyLIV": ("📺", "https://www.sonyliv.com/search?q="),
    "ZEE5": ("🎬", "https://www.zee5.com/search?q="),
    "Apple TV": ("🍎", "https://tv.apple.com/search?term="),
    "YouTube": ("▶️", "https://www.youtube.com/results?search_query="),
    "VI movies and tv": ("📱", "https://www.vimovies.com/"),
}

DOMAIN_MAP = {
    "Netflix": "netflix.com", "Amazon Prime Video": "primevideo.com",
    "Amazon Prime Video with Ads": "primevideo.com", "JioHotstar": "hotstar.com",
    "Disney Plus Hotstar": "hotstar.com", "SonyLIV": "sonyliv.com",
    "ZEE5": "zee5.com", "Apple TV": "tv.apple.com", "YouTube": "youtube.com",
    "VI movies and tv": "vimovies.com",
}

MOOD_GENRE_MAP = {
    'happy': ['Comedy', 'Family', 'Music'], 'sad': ['Drama'], 'love': ['Romance'],
    'romantic': ['Romance'], 'romance': ['Romance'], 'scary': ['Horror'], 'horror': ['Horror'],
    'thrill': ['Thriller'], 'excited': ['Action', 'Adventure'], 'bored': ['Comedy', 'Action'],
    'motivate': ['Drama'], 'inspire': ['Drama'], 'relax': ['Family', 'Animation'],
    'cry': ['Drama'], 'laugh': ['Comedy'], 'funny': ['Comedy'],
    'stress': ['Comedy', 'Family'], 'stressed': ['Comedy', 'Family'], 'tired': ['Family', 'Animation'],
    'angry': ['Action', 'Thriller'], 'lonely': ['Romance', 'Drama'], 'nostalgic': ['Drama', 'Music'],
    'adventure': ['Adventure', 'Action'], 'adventurous': ['Adventure', 'Action'],
    'family': ['Family'], 'friends': ['Comedy', 'Adventure'], 'breakup': ['Drama', 'Romance'],
    'heartbroken': ['Drama', 'Romance'], 'exam': ['Comedy', 'Family'], 'weekend': ['Comedy', 'Action'],
    'festival': ['Family', 'Music'], 'night': ['Thriller', 'Horror'], 'rain': ['Romance', 'Drama'],
    'crime': ['Crime', 'Thriller'], 'mystery': ['Mystery', 'Thriller'], 'war': ['War', 'Action'],
    'history': ['History', 'Drama'], 'kids': ['Family', 'Animation'], 'alone': ['Drama', 'Mystery'],
}


def detect_mood_genres(query):
    query_lower = query.lower()
    detected = set()
    for keyword, genres in MOOD_GENRE_MAP.items():
        if keyword in query_lower:
            detected.update(genres)
    if not detected:
        words = query_lower.split()
        for word in words:
            close = difflib.get_close_matches(word, MOOD_GENRE_MAP.keys(), n=1, cutoff=0.75)
            if close:
                detected.update(MOOD_GENRE_MAP[close[0]])
    return list(detected)


@st.cache_data
def load_data():
    df = pd.read_csv("movies_data.csv")
    df['genres'] = df['genres'].apply(eval)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    return df


@st.cache_resource
def build_similarity(_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(_df['tags'].fillna(''))
    return cosine_similarity(vectors)


df = load_data()
similarity = build_similarity(df)


def fetch_with_retry(url, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            return response.json()
        except Exception:
            if attempt == retries - 1:
                return None
            continue
    return None


@st.cache_data(ttl=3600)
def get_watch_info(movie_id, country='IN'):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return []
    providers = data.get('results', {}).get(country, {})
    flatrate = providers.get('flatrate', [])
    return [p['provider_name'] for p in flatrate] if flatrate else []


@st.cache_data(ttl=3600)
def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return {"budget": 0, "revenue": 0, "status": "No data", "imdb_id": None, "runtime": 0}
    budget = data.get('budget', 0)
    revenue = data.get('revenue', 0)
    imdb_id = data.get('imdb_id', None)
    runtime = data.get('runtime', 0)
    if budget == 0 or revenue == 0:
        status = "No data"
    elif revenue >= budget * 2:
        status = "🔥 Blockbuster"
    elif revenue > budget:
        status = "✅ Hit"
    else:
        status = "❌ Flop"
    return {"budget": budget, "revenue": revenue, "status": status, "imdb_id": imdb_id, "runtime": runtime}


@st.cache_data(ttl=3600)
def get_director(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return "N/A"
    crew = data.get('crew', [])
    directors = [c['name'] for c in crew if c.get('job') == 'Director']
    return ", ".join(directors) if directors else "N/A"


@st.cache_data(ttl=3600)
def get_cast(movie_id, top_n=4):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return []
    cast = data.get('cast', [])[:top_n]
    return [{"name": c['name'], "photo": c.get('profile_path')} for c in cast]


@st.cache_data(ttl=3600)
def get_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return None
    for video in data.get('results', []):
        if video.get('type') == 'Trailer' and video.get('site') == 'YouTube':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None


@st.cache_data(ttl=3600)
def get_imdb_full(imdb_id):
    if not imdb_id:
        return "N/A", "N/A"
    url = f"https://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return "N/A", "N/A"
    rating = data.get('imdbRating', 'N/A')
    awards = data.get('Awards', 'N/A')
    return rating, awards


@st.cache_data(ttl=3600)
def get_all_details(movie_id):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_details = executor.submit(get_movie_details, movie_id)
        f_director = executor.submit(get_director, movie_id)
        f_platforms = executor.submit(get_watch_info, movie_id)
        f_trailer = executor.submit(get_trailer, movie_id)
        details = f_details.result()
        director = f_director.result()
        platforms = f_platforms.result()
        trailer = f_trailer.result()
    imdb_rating, awards = get_imdb_full(details['imdb_id'])
    return details, director, platforms, imdb_rating, trailer, awards


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
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    results = [df.iloc[i] for i, score in distances]
    return df.iloc[idx], results


def show_search_result_card(row):
    rating_10 = round(row['rating_5'] * 2, 1)
    if pd.notna(row['poster_path']):
        st.markdown(f"""
        <div style="position:relative;">
            <img src="https://image.tmdb.org/t/p/w300{row['poster_path']}" style="width:100%; border-radius:8px;">
            <div style="position:absolute; top:8px; left:8px; background:#000000cc; color:#FFD700;
                        padding:3px 8px; border-radius:6px; font-size:13px; font-weight:bold;">
                ⭐ {rating_10}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("🎬 (No poster)")

    if st.button(row['title'], key=f"pick_{row['title']}_{row['id']}", use_container_width=True):
        st.session_state.selected_movie = row['title']
        st.session_state.page_num = 0
        st.rerun()
    st.caption(f"{row['language']} · {', '.join(row['genres'][:2])}")


def show_main_movie_panel(movie):
    with st.spinner("Loading details..."):
        details, director, platforms, imdb_rating, trailer, awards = get_all_details(movie['id'])
        cast = get_cast(movie['id'])

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            if pd.notna(movie['poster_path']):
                st.image(f"https://image.tmdb.org/t/p/w300{movie['poster_path']}")
        with c2:
            movie_year = movie['release_date'][:4] if pd.notna(movie['release_date']) and movie['release_date'] else "N/A"
            runtime = details.get('runtime', 0)
            runtime_str = f"{runtime // 60}h {runtime % 60}m" if runtime else "N/A"

            title_col, heart_col = st.columns([5, 1])
            with title_col:
                st.markdown(f"# {movie['title']}")
            with heart_col:
                st.button("❤️", key=f"heart_{movie['id']}")

            st.markdown(f"""
            <p style="color:#aaa;">{movie_year} &nbsp;·&nbsp; {runtime_str} &nbsp;·&nbsp; {movie['language']}
            &nbsp;&nbsp; <span style="background:#FFD70022; color:#FFD700; padding:3px 10px; border-radius:6px; font-weight:bold;">⭐ {imdb_rating}/10 IMDb</span></p>
            """, unsafe_allow_html=True)

            genre_pills = " ".join([f'<span style="background:#2a2a2a; color:#ddd; padding:5px 12px; border-radius:6px; margin-right:6px; font-size:13px;">{g}</span>' for g in movie['genres']])
            st.markdown(f"<div style='margin-bottom:12px;'>{genre_pills}</div>", unsafe_allow_html=True)

            st.markdown("**Overview**")
            st.write(movie['overview'])

            st.write(f"**🎬 Director:** {director}")

            if cast:
                st.markdown("**Cast**")
                cast_cols = st.columns(6)
                for i, actor in enumerate(cast):
                    with cast_cols[i]:
                        if actor['photo']:
                            st.image(f"https://image.tmdb.org/t/p/w92{actor['photo']}", width=70)
                        st.caption(actor['name'])

            st.markdown("")
            btn_cols = st.columns([1, 1, 1.3, 0.5])
            with btn_cols[0]:
                if trailer:
                    st.link_button("▶️ Watch Trailer", trailer)
                else:
                    yt_search = f"https://www.youtube.com/results?search_query={movie['title'].replace(' ', '+')}+trailer"
                    st.link_button("▶️ Watch Trailer", yt_search)
            with btn_cols[1]:
                yt_songs = f"https://www.youtube.com/results?search_query={movie['title'].replace(' ', '+')}+songs"
                st.link_button("🎵 Songs", yt_songs)
            with btn_cols[2]:
                st.button("➕ Watchlist", key=f"watchlist_{movie['id']}")

            g1, g2, g3 = st.columns(3)
            g1.metric("💰 Budget", format_money(details['budget']))
            g2.metric("💵 Collections", format_money(details['revenue']))
            g3.metric("📊 Status", details['status'])
            st.caption(f"🏆 Awards: {awards}")

            if platforms:
                st.markdown("**Available on (OTT)**")
                p_cols = st.columns(len(platforms))
                for i, p in enumerate(platforms):
                    _, base_link = PLATFORM_LINKS.get(p, ("🔗", "https://www.google.com/search?q="))
                    domain = DOMAIN_MAP.get(p, "google.com")
                    favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
                    full_link = base_link + movie['title'].replace(" ", "+")
                    with p_cols[i]:
                        st.markdown(f"""
                        <a href="{full_link}" target="_blank" style="text-decoration:none;">
                            <div style="background:#1A1A1A; border:1px solid #333; border-radius:8px; padding:10px; text-align:center;">
                                <img src="{favicon}" width="24" style="margin-bottom:4px;"><br>
                                <span style="color:#ddd; font-size:12px;">{p}</span>
                            </div>
                        </a>
                        """, unsafe_allow_html=True)
            else:
                st.write("**📺 Not available on OTT**")

            st.markdown("**Your Review**")
            star_rating = st.feedback("stars", key=f"stars_{movie['id']}")
            review = st.text_area("Write your review...", key=f"review_main_{movie['id']}", label_visibility="collapsed")
            if st.button("Submit Review", key=f"save_main_{movie['id']}"):
                st.success("Review saved!")


def show_movie_card(movie):
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

            if st.button("👁️ View details", key=f"view_{movie['id']}"):
                st.session_state.selected_movie = movie['title']
                st.rerun()


# ------------------ UI ------------------
st.markdown("""
<style>
input[type="text"] { font-size: 18px !important; padding: 10px !important; }

.stButton button {
    border-radius: 6px !important;
    font-weight: 600 !important;
}

div[data-testid="stImage"] img {
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}

.hero-box {
    padding: 30px 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%);
    margin-bottom: 20px;
    border: 1px solid #E50914;
}

.stat-card {
    background-color: #1A1A1A;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    border: 1px solid #333;
}

@media (max-width: 992px) {
    input[type="text"] { font-size: 16px !important; }
    h1 { font-size: 28px !important; }
    h2 { font-size: 22px !important; }
}

@media (max-width: 600px) {
    input[type="text"] { font-size: 14px !important; padding: 8px !important; }
    h1 { font-size: 22px !important; }
    h2 { font-size: 18px !important; }
    h3 { font-size: 16px !important; }
    .stButton button { font-size: 12px !important; padding: 4px 8px !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <h1 style="margin-bottom:0; color:white;">🎬 Cine<span style="color:#E50914;">Match</span></h1>
    <p style="color:#aaa; font-size:18px; margin-top:4px;">Find Your Next <span style="color:#E50914; font-weight:bold;">Favorite Movie</span></p>
    <p style="color:#888;">AI-powered recommendations across English, Hindi, Telugu, Malayalam and more.</p>
</div>
""", unsafe_allow_html=True)

stat_cols = st.columns(4)
stats = [("🎬", f"{len(df)}+", "Movies"), ("🎭", f"{len(set(g for genres in df['genres'] for g in genres))}+", "Genres"),
         ("🌐", "4", "Languages"), ("⭐", "500K+", "Ratings")]
for col, (icon, num, label) in zip(stat_cols, stats):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:22px;">{icon} <b>{num}</b></div>
            <div style="color:#888; font-size:13px;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 👨‍💻 About")
    st.write("**Galla Ramakrishna**")
    st.caption("Data Science / CSE Student")
    st.divider()
    st.caption("Made with ❤️ using Streamlit")
    st.caption("Powered by TMDB · OMDb · YouTube")

col1, col2 = st.columns(2)
with col1:
    lang_filter = st.multiselect("Select Language", options=df['language'].unique(), default=list(df['language'].unique()))
with col2:
    all_genres = sorted(set(g for genres in df['genres'] for g in genres if g))
    genre_filter = st.multiselect("Select Genre (optional)", options=all_genres)

filtered_df = df[df['language'].isin(lang_filter)]
if genre_filter:
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda g: any(genre in g for genre in genre_filter))]

if 'selected_year' not in st.session_state:
    st.session_state.selected_year = None

if genre_filter:
    available_years = sorted(filtered_df['year'].dropna().unique().astype(int), reverse=True)
    if available_years:
        st.write("**📅 Filter by Year:**")
        if st.session_state.selected_year:
            if st.button(f"❌ Clear year filter ({st.session_state.selected_year})"):
                st.session_state.selected_year = None
                st.rerun()
        for i in range(0, len(available_years), 8):
            year_chunk = available_years[i:i + 8]
            cols = st.columns(8)
            for j, yr in enumerate(year_chunk):
                with cols[j]:
                    if st.button(str(yr), key=f"year_{yr}"):
                        st.session_state.selected_year = int(yr)
                        st.rerun()

    if st.session_state.selected_year:
        filtered_df = filtered_df[filtered_df['year'] == st.session_state.selected_year]

if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'page_num' not in st.session_state:
    st.session_state.page_num = 0

if st.session_state.selected_movie:
    if st.button("🔙 Back to movies list"):
        st.session_state.selected_movie = None
        st.rerun()
    st.divider()

    selected, results = recommend(st.session_state.selected_movie)

    if selected is None:
        st.error("Movie not found")
    else:
        show_main_movie_panel(selected)
        st.divider()
        st.subheader("🎯 Similar Movies:")
        for movie in results:
            show_movie_card(movie)

else:
    col_a, col_b = st.columns(2)
    with col_a:
        search_query = st.text_input("🔍 Search movie name")
    with col_b:
        mood_query = st.text_input("💭 How are you feeling? Describe your mood or situation (e.g. 'feeling stressed', 'want a fun family movie', 'just had a breakup')")

    if mood_query:
        mood_genres = detect_mood_genres(mood_query)
        if mood_genres:
            st.info(f"🎭 Based on your mood, suggesting: {', '.join(mood_genres)}")
            mood_movies = filtered_df[filtered_df['genres'].apply(lambda g: any(genre in g for genre in mood_genres))]
            mood_movies = mood_movies.sort_values('rating_5', ascending=False)
            st.write(f"**💝 Movies for you ({len(mood_movies)} found):**")
            rows = list(mood_movies.iterrows())
            for i in range(0, len(rows), 4):
                row_chunk = rows[i:i + 4]
                cols = st.columns(4)
                for j, (_, m) in enumerate(row_chunk):
                    with cols[j]:
                        show_search_result_card(m)
        else:
            st.warning("Couldn't detect a mood, try words like 'happy', 'sad', 'love', 'scary', 'stressed', 'adventure' etc.")
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
            st.write(f"**🔎 {len(combined_df)} results found — click to view:**")
            rows = list(combined_df.iterrows())
            for i in range(0, len(rows), 4):
                row_chunk = rows[i:i + 4]
                cols = st.columns(4)
                for j, (_, m) in enumerate(row_chunk):
                    with cols[j]:
                        show_search_result_card(m)
        else:
            st.warning("No movie found, try a different word")
        st.divider()

    else:
        st.write(f"### 📋 Total Movies: {len(filtered_df)}")
        PER_PAGE = 12
        total_pages = max((len(filtered_df) - 1) // PER_PAGE + 1, 1)
        start = st.session_state.page_num * PER_PAGE
        end = start + PER_PAGE
        page_movies = filtered_df.iloc[start:end]

        rows = list(page_movies.iterrows())
        for i in range(0, len(rows), 4):
            row_chunk = rows[i:i + 4]
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