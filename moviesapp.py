import streamlit as st
import pandas as pd
import requests
import difflib
import time
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


def fetch_with_retry(url, retries=5, timeout=15):
    headers = {"Connection": "close"}
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            return response.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1)
            continue
    return None


@st.cache_data(ttl=1800)
def get_trending_now():
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return []
    results = []
    for m in data.get('results', [])[:10]:
        results.append({
            "id": m.get("id"), "title": m.get("title", "Unknown"),
            "poster_path": m.get("poster_path"), "rating_5": round(m.get("vote_average", 0) / 2, 1),
        })
    return results


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
        return {"budget": 0, "revenue": 0, "status": "No data", "imdb_id": None, "runtime": 0, "countries": "N/A"}
    budget = data.get('budget', 0)
    revenue = data.get('revenue', 0)
    imdb_id = data.get('imdb_id', None)
    runtime = data.get('runtime', 0)
    countries = ", ".join([c['name'] for c in data.get('production_countries', [])]) or "N/A"
    if budget == 0 or revenue == 0:
        status = "No data"
    elif revenue >= budget * 2:
        status = "🔥 Blockbuster"
    elif revenue > budget:
        status = "✅ Hit"
    else:
        status = "❌ Flop"
    return {"budget": budget, "revenue": revenue, "status": status, "imdb_id": imdb_id, "runtime": runtime, "countries": countries}


@st.cache_data(ttl=3600)
def get_movie_full_info(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return None
    genre_names = [g['name'] for g in data.get('genres', [])]
    return {
        "id": data.get("id"), "title": data.get("title", "Unknown"),
        "poster_path": data.get("poster_path"), "release_date": data.get("release_date", ""),
        "language": data.get("original_language", "en").upper(),
        "genres": genre_names if genre_names else ["N/A"],
        "overview": data.get("overview", "No overview available."),
        "rating_5": round(data.get("vote_average", 0) / 2, 1),
    }


@st.cache_data(ttl=3600)
def get_crew_full(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return {}
    crew = data.get('crew', [])
    roles = {"Director": [], "Writer": [], "Producer": [], "Music": [], "Editor": [], "Cinematography": []}
    job_map = {
        "Director": "Director", "Writer": "Writer", "Screenplay": "Writer",
        "Producer": "Producer", "Original Music Composer": "Music",
        "Editor": "Editor", "Director of Photography": "Cinematography",
    }
    for c in crew:
        job = c.get('job', '')
        if job in job_map:
            roles[job_map[job]].append(c['name'])
    return {k: ", ".join(sorted(set(v))) if v else "N/A" for k, v in roles.items()}


@st.cache_data(ttl=3600)
def get_director(movie_id):
    crew = get_crew_full(movie_id)
    return crew.get("Director", "N/A")


@st.cache_data(ttl=3600)
def get_cast(movie_id, top_n=5):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return [], 0
    cast_full = data.get('cast', [])
    cast = cast_full[:top_n]
    return [{"name": c['name'], "photo": c.get('profile_path'), "character": c.get('character', '')} for c in cast], len(cast_full)


@st.cache_data(ttl=3600)
def get_all_cast(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return []
    cast_full = data.get('cast', [])
    return [{"name": c['name'], "photo": c.get('profile_path'), "character": c.get('character', '')} for c in cast_full]


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
        return "N/A", "N/A", "N/A"
    url = f"https://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
    data = fetch_with_retry(url)
    if not data:
        return "N/A", "N/A", "N/A"
    rating = data.get('imdbRating', 'N/A')
    awards = data.get('Awards', 'N/A')
    votes = data.get('imdbVotes', 'N/A')
    return rating, awards, votes


@st.cache_data(ttl=3600)
def get_all_details(movie_id):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f_details = executor.submit(get_movie_details, movie_id)
        f_crew = executor.submit(get_crew_full, movie_id)
        f_platforms = executor.submit(get_watch_info, movie_id)
        f_trailer = executor.submit(get_trailer, movie_id)
        details = f_details.result()
        crew = f_crew.result()
        platforms = f_platforms.result()
        trailer = f_trailer.result()
    imdb_rating, awards, votes = get_imdb_full(details['imdb_id'])
    return details, crew, platforms, imdb_rating, trailer, awards, votes


def format_money(amount):
    if amount == 0:
        return "N/A"
    return f"${amount:,}"


def recommend(movie_title, top_n=8):
    matches = df[df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return None, []
    idx = matches.index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    results = [df.iloc[i] for i, score in distances]
    return df.iloc[idx], results


if 'favorites' not in st.session_state:
    st.session_state.favorites = {}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = {}


def info_card(icon, label, value):
    return f'<div style="background:#151515; border:1px solid #2a2a2a; border-radius:14px; padding:14px; text-align:center; min-width:150px;"><div style="font-size:20px;">{icon}</div><div style="font-weight:700; font-size:15px; color:#fff; margin-top:4px;">{value}</div><div style="color:#888; font-size:12px;">{label}</div></div>'


def show_search_result_card(row, prefix=""):
    rating_10 = round(row['rating_5'] * 2, 1)
    if pd.notna(row['poster_path']):
        html = f'<div style="position:relative;"><img class="poster-img" src="https://image.tmdb.org/t/p/w300{row["poster_path"]}" style="width:100%; border-radius:8px;"><div style="position:absolute; top:8px; left:8px; background:#000000cc; color:#FFD700; padding:3px 8px; border-radius:6px; font-size:13px; font-weight:bold;">⭐ {rating_10}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.write("🎬 (No poster)")

    if st.button(row['title'], key=f"pick_{prefix}_{row['title']}_{row['id']}", use_container_width=True):
        st.session_state.selected_movie = row['title']
        st.session_state.external_movie_id = None
        st.session_state.page_num = 0
        st.rerun()
    st.caption(f"{row['language']} · {', '.join(row['genres'][:2])}")


def show_main_movie_panel(movie, is_external=False):
    with st.spinner("Loading details..."):
        details, crew, platforms, imdb_rating, trailer, awards, votes = get_all_details(movie['id'])
        cast, total_cast = get_cast(movie['id'])

    movie_year = movie['release_date'][:4] if movie.get('release_date') else "N/A"
    runtime = details.get('runtime', 0)
    runtime_str = f"{runtime // 60}h {runtime % 60}m" if runtime else "N/A"

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            if pd.notna(movie['poster_path']):
                st.image(f"https://image.tmdb.org/t/p/w400{movie['poster_path']}")
        with c2:
            title_col, heart_col, wl_col = st.columns([4, 1, 1])
            with title_col:
                st.markdown(f"# {movie['title']}")
            with heart_col:
                is_fav = movie['id'] in st.session_state.favorites
                if st.button("❤️" if is_fav else "🤍", key=f"heart_{movie['id']}_{is_external}"):
                    if is_fav:
                        del st.session_state.favorites[movie['id']]
                    else:
                        st.session_state.favorites[movie['id']] = {"id": movie['id'], "title": movie['title'], "poster_path": movie['poster_path'], "rating_5": movie['rating_5'], "language": movie['language'], "genres": movie['genres']}
                    st.rerun()
            with wl_col:
                in_wl = movie['id'] in st.session_state.watchlist
                if st.button("✅" if in_wl else "➕", key=f"wl_{movie['id']}_{is_external}"):
                    if in_wl:
                        del st.session_state.watchlist[movie['id']]
                    else:
                        st.session_state.watchlist[movie['id']] = {"id": movie['id'], "title": movie['title'], "poster_path": movie['poster_path'], "rating_5": movie['rating_5'], "language": movie['language'], "genres": movie['genres']}
                    st.rerun()

            st.caption(f"{movie_year} · {runtime_str} · {movie['language']}")

            genre_pills = " ".join([f'<span style="background:#2a2a2a; color:#ddd; padding:5px 12px; border-radius:6px; margin-right:6px; font-size:13px;">{g}</span>' for g in movie['genres']])
            st.markdown(f"<div style='margin-bottom:14px;'>{genre_pills}</div>", unsafe_allow_html=True)

            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                st.markdown(info_card("⭐", "IMDb Rating", f"{imdb_rating}/10"), unsafe_allow_html=True)
            with ic2:
                st.markdown(info_card("👥", "Votes", votes), unsafe_allow_html=True)
            with ic3:
                st.markdown(info_card("📅", "Release Date", movie.get('release_date') or "N/A"), unsafe_allow_html=True)

            st.markdown("")
            btn_cols = st.columns(2)
            with btn_cols[0]:
                if trailer:
                    st.link_button("▶️ Watch Trailer", trailer, use_container_width=True)
                else:
                    yt_search = f"https://www.youtube.com/results?search_query={movie['title'].replace(' ', '+')}+trailer"
                    st.link_button("▶️ Watch Trailer", yt_search, use_container_width=True)
            with btn_cols[1]:
                yt_songs = f"https://www.youtube.com/results?search_query={movie['title'].replace(' ', '+')}+songs"
                st.link_button("🎵 Search Songs", yt_songs, use_container_width=True)

    with st.container(border=True):
        st.markdown("### 📖 Story")
        st.write(movie['overview'])

    with st.container(border=True):
        st.markdown("### 🎬 Movie Information")
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            st.markdown(info_card("🎭", "Genre", ", ".join(movie['genres'][:2])), unsafe_allow_html=True)
        with i2:
            st.markdown(info_card("🌍", "Language", movie['language']), unsafe_allow_html=True)
        with i3:
            st.markdown(info_card("⏱", "Runtime", runtime_str), unsafe_allow_html=True)
        with i4:
            st.markdown(info_card("🌎", "Country", details.get('countries', 'N/A')), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"### 👥 Top Cast")
        if cast:
            cast_cols = st.columns(len(cast))
            for i, actor in enumerate(cast):
                with cast_cols[i]:
                    if actor['photo']:
                        st.image(f"https://image.tmdb.org/t/p/w185{actor['photo']}", width=90)
                    wiki_url = f"https://en.wikipedia.org/wiki/{actor['name'].replace(' ', '_')}"
                    st.markdown(f'<a href="{wiki_url}" target="_blank" style="color:#E50914; font-weight:600; text-decoration:none;">{actor["name"]}</a>', unsafe_allow_html=True)
                    st.caption(actor['character'])

            if total_cast > len(cast):
                with st.expander(f"View All Cast ({total_cast})"):
                    all_cast = get_all_cast(movie['id'])
                    for actor in all_cast:
                        ac1, ac2 = st.columns([1, 4])
                        with ac1:
                            if actor['photo']:
                                st.image(f"https://image.tmdb.org/t/p/w92{actor['photo']}", width=50)
                        with ac2:
                            wiki_url = f"https://en.wikipedia.org/wiki/{actor['name'].replace(' ', '_')}"
                            st.markdown(f'<a href="{wiki_url}" target="_blank" style="color:#E50914;">{actor["name"]}</a> — <span style="color:#888;">{actor["character"]}</span>', unsafe_allow_html=True)
        else:
            st.caption("Cast information not available (network issue — try again).")

    cr1, cr2, cr3 = st.columns(3)
    with cr1:
        with st.container(border=True):
            st.markdown("### 🎬 Crew")
            st.write(f"**Director:** {crew.get('Director', 'N/A')}")
            st.write(f"**Writer:** {crew.get('Writer', 'N/A')}")
            st.write(f"**Producer:** {crew.get('Producer', 'N/A')}")
            st.write(f"**Music:** {crew.get('Music', 'N/A')}")
            st.write(f"**Editor:** {crew.get('Editor', 'N/A')}")
            st.write(f"**Cinematography:** {crew.get('Cinematography', 'N/A')}")
    with cr2:
        with st.container(border=True):
            st.markdown("### ⭐ Ratings")
            st.write(f"**IMDb:** {imdb_rating}/10")
            st.write(f"**TMDB:** {movie['rating_5']}/5")
            st.caption("Rotten Tomatoes / Metacritic not available")
            st.write(f"**🏆 Awards:** {awards}")
    with cr3:
        with st.container(border=True):
            st.markdown("### 💰 Box Office")
            st.write(f"**Budget:** {format_money(details['budget'])}")
            st.write(f"**Worldwide Collection:** {format_money(details['revenue'])}")
            st.write(f"**Status:** {details['status']}")

    with st.container(border=True):
        st.markdown("### 📺 Available On OTT")
        if platforms:
            p_cols = st.columns(len(platforms))
            for i, p in enumerate(platforms):
                _, base_link = PLATFORM_LINKS.get(p, ("🔗", "https://www.google.com/search?q="))
                domain = DOMAIN_MAP.get(p, "google.com")
                favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
                full_link = base_link + movie['title'].replace(" ", "+")
                with p_cols[i]:
                    card_html = f'<a href="{full_link}" target="_blank" style="text-decoration:none;"><div style="background:#1A1A1A; border:1px solid #333; border-radius:8px; padding:10px; text-align:center;"><img src="{favicon}" width="24" style="margin-bottom:4px;"><br><span style="color:#ddd; font-size:12px;">{p}</span></div></a>'
                    st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.write("Not available on OTT")

    with st.container(border=True):
        st.markdown("### Your Review")
        star_rating = st.feedback("stars", key=f"stars_{movie['id']}")
        review = st.text_area("Write your review...", key=f"review_main_{movie['id']}", label_visibility="collapsed")
        if st.button("Submit Review", key=f"save_main_{movie['id']}"):
            st.success("Review saved!")

    if is_external:
        st.info("ℹ️ ఇది మన movie database లో లేదు కాబట్టి, 'Similar Movies' చూపించలేము.")


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
                st.session_state.external_movie_id = None
                st.rerun()


def open_movie_from_dict(item):
    match = df[df['title'].str.lower() == item['title'].lower()]
    if not match.empty:
        st.session_state.selected_movie = match.iloc[0]['title']
        st.session_state.external_movie_id = None
    else:
        st.session_state.external_movie_id = item['id']
        st.session_state.selected_movie = None
    st.session_state.nav_page = "Home"
    st.rerun()


# ------------------ UI ------------------
st.markdown("""
<style>
input[type="text"] { font-size: 18px !important; padding: 10px !important; }
.stButton button { border-radius: 6px !important; font-weight: 600 !important; }
div[data-testid="stImage"] img { border-radius: 8px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
.hero-box { padding: 30px 20px; border-radius: 12px; background: linear-gradient(135deg, #1a1a1a 0%, #0d0d0d 100%); margin-bottom: 20px; border: 1px solid #E50914; }
.stat-card { background-color: #1A1A1A; border-radius: 10px; padding: 14px; text-align: center; border: 1px solid #333; }

@media (max-width: 992px) {
    input[type="text"] { font-size: 16px !important; }
    h1 { font-size: 26px !important; }
    h2 { font-size: 20px !important; }
    .hero-box { padding: 20px 14px !important; }
    .stat-card { padding: 10px !important; }
}

/* ===== MOBILE ONLY (<768px): 2-column compact movie grid ===== */
/* Everything in this block is scoped strictly to max-width:767px and does
   NOT affect desktop/tablet layout in any way. */
@media (max-width: 767px) {
    /* Force movie-card grids (built with st.columns) into a tight 2-column layout.
       Streamlit has used different data-testid names across versions
       (stHorizontalBlock/stColumn in older builds, stColumn/column and
       [data-testid="stHorizontalBlock"] descendants in newer ones), so every
       selector below is repeated for each known variant to stay version-safe. */
    div[data-testid="stHorizontalBlock"],
    div[data-testid="column"],
    div[data-testid="stHorizontalBlock"] > div {
        gap: 10px !important;
        row-gap: 18px !important;
        flex-wrap: wrap !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
    div[data-testid="stHorizontalBlock"] div[data-testid="column"],
    div[data-testid="stHorizontalBlock"] > div[class*="stColumn"],
    div[data-testid="stHorizontalBlock"] > div[class*="column"] {
        min-width: 47% !important;
        max-width: 48% !important;
        flex: 1 1 47% !important;
        width: 47% !important;
    }

    /* Movie poster: professional 2:3 poster ratio, fills card width, no overflow */
    .poster-img {
        width: 100% !important;
        height: auto !important;
        aspect-ratio: 2 / 3 !important;
        object-fit: cover !important;
        border-radius: 10px !important;
        display: block !important;
    }

    /* Movie title (rendered as a button) wraps naturally instead of truncating */
    .stButton button {
        font-size: 12px !important;
        padding: 4px 8px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.25 !important;
        text-align: left !important;
    }

    input[type="text"] { font-size: 14px !important; padding: 8px !important; }
    h1 { font-size: 20px !important; }
    h2 { font-size: 16px !important; }
    h3 { font-size: 14px !important; }
    .hero-box { padding: 16px 10px !important; }
    .hero-box p { font-size: 13px !important; }
    .stat-card { padding: 8px !important; }
    .stat-card div:first-child { font-size: 15px !important; }
    div[data-testid="stMetricValue"] { font-size: 16px !important; }
    div[data-testid="stMetricLabel"] { font-size: 11px !important; }

    /* Prevent any accidental horizontal scroll on small screens */
    .block-container { overflow-x: hidden !important; padding-left: 12px !important; padding-right: 12px !important; }
}
</style>
""", unsafe_allow_html=True)

if 'nav_page' not in st.session_state:
    st.session_state.nav_page = "Home"
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'external_movie_id' not in st.session_state:
    st.session_state.external_movie_id = None
if 'page_num' not in st.session_state:
    st.session_state.page_num = 0
if 'show_more_trending' not in st.session_state:
    st.session_state.show_more_trending = False

with st.sidebar:
    st.markdown("## 🎬 Cine**Match**")
    st.caption("Find your perfect movie")
    st.markdown("")

    nav_items = ["🏠 Home", "🔥 Trending", "🎞️ Movies", "❤️ Favorites", "🔖 Watchlist", "🤖 AI Recommend", "ℹ️ About"]
    for item in nav_items:
        page_name = item.split(" ", 1)[1]
        is_active = st.session_state.nav_page == page_name
        if st.button(item, key=f"nav_{page_name}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.nav_page = page_name
            st.session_state.selected_movie = None
            st.session_state.external_movie_id = None
            st.rerun()

    st.divider()
    st.markdown("### 👨‍💻 About")
    st.write("**Galla Ramakrishna**")
    st.caption("Data Science / CSE Student")
    st.divider()
    st.caption("Made with ❤️ using Streamlit")
    st.caption("Powered by TMDB · OMDb · YouTube")

if st.session_state.external_movie_id:
    if st.button("🔙 Back"):
        st.session_state.external_movie_id = None
        st.rerun()
    st.divider()
    ext_movie = get_movie_full_info(st.session_state.external_movie_id)
    if ext_movie is None:
        st.error("Could not load movie details — please try again (network issue).")
    else:
        show_main_movie_panel(ext_movie, is_external=True)

elif st.session_state.selected_movie:
    if st.button("🔙 Back"):
        st.session_state.selected_movie = None
        st.rerun()
    st.divider()
    selected, results = recommend(st.session_state.selected_movie)
    if selected is None:
        st.error("Movie not found")
    else:
        show_main_movie_panel(selected)
        st.divider()
        st.subheader("🎯 Similar Movies")
        for movie in results:
            show_movie_card(movie)

elif st.session_state.nav_page == "About":
    st.title("ℹ️ About CineMatch")
    st.write("CineMatch is an AI-powered movie recommendation platform that helps you discover the perfect movie based on your mood, preferences and interests.")
    st.write("- 🤖 AI Recommendations\n- 📺 OTT Availability\n- ▶️ Trailer Search\n- 💭 Mood Search\n- 🔍 Smart Filtering")
    st.divider()
    st.write("**Developer:** Galla Ramakrishna — Data Science / CSE Student")

elif st.session_state.nav_page == "Favorites":
    st.title("❤️ Your Favorites")
    if not st.session_state.favorites:
        st.info("నువ్వు ఇంకా ఏ movie ని ❤️ చేయలేదు.")
    else:
        favs = list(st.session_state.favorites.values())
        for i in range(0, len(favs), 4):
            row_chunk = favs[i:i + 4]
            cols = st.columns(4)
            for j, item in enumerate(row_chunk):
                with cols[j]:
                    if item['poster_path']:
                        st.markdown(f'<img class="poster-img" src="https://image.tmdb.org/t/p/w300{item["poster_path"]}" style="width:100%; border-radius:8px;">', unsafe_allow_html=True)
                    if st.button(item['title'], key=f"fav_open_{item['id']}", use_container_width=True):
                        open_movie_from_dict(item)
                    st.caption(f"⭐ {item['rating_5']}/5 · {item['language']}")

elif st.session_state.nav_page == "Watchlist":
    st.title("🔖 Your Watchlist")
    if not st.session_state.watchlist:
        st.info("నువ్వు ఇంకా ఏ movie ని Watchlist లో add చేయలేదు.")
    else:
        wl = list(st.session_state.watchlist.values())
        for i in range(0, len(wl), 4):
            row_chunk = wl[i:i + 4]
            cols = st.columns(4)
            for j, item in enumerate(row_chunk):
                with cols[j]:
                    if item['poster_path']:
                        st.markdown(f'<img class="poster-img" src="https://image.tmdb.org/t/p/w300{item["poster_path"]}" style="width:100%; border-radius:8px;">', unsafe_allow_html=True)
                    if st.button(item['title'], key=f"wl_open_{item['id']}", use_container_width=True):
                        open_movie_from_dict(item)
                    st.caption(f"⭐ {item['rating_5']}/5 · {item['language']}")

elif st.session_state.nav_page == "AI Recommend":
    st.title("🤖 AI Recommend")
    st.write("Describe your mood or situation, and CineMatch will suggest movies for you.")
    mood_query = st.text_input("💭 How are you feeling?", placeholder="e.g. 'feeling stressed', 'want a fun family movie', 'just had a breakup'")
    if mood_query:
        mood_genres = detect_mood_genres(mood_query)
        if mood_genres:
            st.info(f"🎭 Based on your mood, suggesting: {', '.join(mood_genres)}")
            mood_movies = df[df['genres'].apply(lambda g: any(genre in g for genre in mood_genres))]
            mood_movies = mood_movies.sort_values('rating_5', ascending=False)
            st.write(f"**💝 Movies for you ({len(mood_movies)} found):**")
            rows = list(mood_movies.head(20).iterrows())
            for i in range(0, len(rows), 4):
                row_chunk = rows[i:i + 4]
                cols = st.columns(4)
                for j, (_, m) in enumerate(row_chunk):
                    with cols[j]:
                        show_search_result_card(m, prefix="airec")
        else:
            st.warning("Couldn't detect a mood, try words like 'happy', 'sad', 'love', 'scary', 'stressed', 'adventure' etc.")

elif st.session_state.nav_page == "Trending":
    st.title("🔥 Trending Now")
    trending_list = get_trending_now()
    if trending_list:
        rows = trending_list
        for i in range(0, len(rows), 4):
            row_chunk = rows[i:i + 4]
            cols = st.columns(4)
            for j, item in enumerate(row_chunk):
                with cols[j]:
                    if item['poster_path']:
                        html = f'<div style="position:relative;"><img class="poster-img" src="https://image.tmdb.org/t/p/w300{item["poster_path"]}" style="width:100%; border-radius:8px;"><div style="position:absolute; top:8px; left:8px; background:#000000cc; color:#FFD700; padding:3px 8px; border-radius:6px; font-size:13px; font-weight:bold;">⭐ {item["rating_5"]}</div></div>'
                        st.markdown(html, unsafe_allow_html=True)
                    if st.button(item['title'], key=f"trendpg_{item['id']}", use_container_width=True):
                        open_movie_from_dict(item)
    else:
        st.warning("Trending movies could not be loaded — this can happen with an unstable network connection. Try refreshing.")

elif st.session_state.nav_page == "Movies":
    st.title("🎞️ All Movies")
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1, 1, 1.3, 1.5])
        with f1:
            lang_filter = st.multiselect("Language", options=df['language'].unique(), default=list(df['language'].unique()), key="movies_lang")
        with f2:
            all_genres = sorted(set(g for genres in df['genres'] for g in genres if g))
            genre_filter = st.multiselect("Genre", options=all_genres, key="movies_genre")
        with f3:
            min_yr, max_yr = int(df['year'].min()), int(df['year'].max())
            year_range = st.slider("Release Year", min_yr, max_yr, (min_yr, max_yr), key="movies_year")
        with f4:
            quick_search = st.text_input("Search Movie", placeholder="Search for movies...", key="movies_search")

    filtered_df = df[df['language'].isin(lang_filter)]
    if genre_filter:
        filtered_df = filtered_df[filtered_df['genres'].apply(lambda g: any(genre in g for genre in genre_filter))]
    filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
    if quick_search:
        filtered_df = filtered_df[filtered_df['title'].str.contains(quick_search, case=False, na=False)]

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
                show_search_result_card(m, prefix="moviespage")

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

else:
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
            card_html = f'<div class="stat-card"><div style="font-size:22px;">{icon} <b>{num}</b></div><div style="color:#888; font-size:13px;">{label}</div></div>'
            st.markdown(card_html, unsafe_allow_html=True)

    with st.container(border=True):
        f1, f2, f3, f4, f5 = st.columns([1, 1, 1.3, 1.5, 1.5])
        with f1:
            lang_filter = st.multiselect("Language", options=df['language'].unique(), default=list(df['language'].unique()))
        with f2:
            all_genres = sorted(set(g for genres in df['genres'] for g in genres if g))
            genre_filter = st.multiselect("Genre", options=all_genres)
        with f3:
            min_yr, max_yr = int(df['year'].min()), int(df['year'].max())
            year_range = st.slider("Release Year", min_yr, max_yr, (min_yr, max_yr))
        with f4:
            quick_search = st.text_input("Search Movie", placeholder="Search for movies...")
        with f5:
            quick_mood = st.text_input("Mood Search (AI)", placeholder="How are you feeling?")

    filtered_df = df[df['language'].isin(lang_filter)]
    if genre_filter:
        filtered_df = filtered_df[filtered_df['genres'].apply(lambda g: any(genre in g for genre in genre_filter))]
    filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]

    search_query = quick_search
    mood_query = quick_mood

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
                        show_search_result_card(m, prefix="mood")
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
                        show_search_result_card(m, prefix="search")
        else:
            st.warning("No movie found, try a different word")
        st.divider()

    else:
        st.markdown("### 🔥 Trending Now")
        trending_list = get_trending_now()
        if trending_list:
            show_count = 10 if st.session_state.show_more_trending else 5
            t_cols = st.columns(5)
            for i, item in enumerate(trending_list[:show_count]):
                col_idx = i % 5
                if i > 0 and col_idx == 0:
                    t_cols = st.columns(5)
                with t_cols[col_idx]:
                    if item['poster_path']:
                        html = f'<div style="position:relative;"><img class="poster-img" src="https://image.tmdb.org/t/p/w300{item["poster_path"]}" style="width:100%; border-radius:8px;"><div style="position:absolute; top:8px; left:8px; background:#000000cc; color:#FFD700; padding:3px 8px; border-radius:6px; font-size:13px; font-weight:bold;">⭐ {item["rating_5"]}</div></div>'
                        st.markdown(html, unsafe_allow_html=True)
                    if st.button(item['title'], key=f"pick_trend_{item['id']}", use_container_width=True):
                        open_movie_from_dict(item)

            if not st.session_state.show_more_trending and len(trending_list) > 5:
                if st.button("⬇️ Show More Trending"):
                    st.session_state.show_more_trending = True
                    st.rerun()
        else:
            st.warning("Trending movies could not be loaded — this can happen with an unstable network connection. Try refreshing the page.")
        st.divider()

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
                    show_search_result_card(m, prefix="grid")

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