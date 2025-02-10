import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load dataset
tracks = pd.read_csv(r"C:\Users\admin\Python_YT\Gdsc\track_ds.csv")

# ✅ Fix column names
tracks.columns = tracks.columns.str.strip()
tracks.rename(columns={'names': 'track_name'}, inplace=True)

# ✅ Check if required columns exist
if 'artist_genres' in tracks.columns and 'artist_names' in tracks.columns:
    # 🎯 TF-IDF Vectorization on Genres
    tfidf_genre = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf_genre.fit_transform(tracks['artist_genres'].fillna(''))

    # 🎯 Compute Cosine Similarity
    genre_similarity = cosine_similarity(genre_matrix, genre_matrix)
else:
    genre_similarity = None

# 🎯 Recommendation Function (Only Genre-Based, Different Artists)
def genre_based_recommendations(track_name, top_n=5):
    if genre_similarity is None:
        return pd.DataFrame({"Error": ["Genre data unavailable!"]})

    if track_name not in tracks['track_name'].values:
        return pd.DataFrame({"Error": ["Track not found!"]})

    # Find the index of the selected track
    idx = tracks[tracks['track_name'] == track_name].index
    if len(idx) == 0:
        return pd.DataFrame({"Error": ["Track title not found!"]})

    idx = idx[0]
    input_artist = tracks.loc[idx, 'artist_names']  # Get the artist of selected track

    # Get similarity scores
    sim_scores = list(enumerate(genre_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filter out tracks by the same artist
    filtered_tracks = []
    seen_artists = {input_artist}  # Keep track of seen artists

    for i, score in sim_scores:
        if len(filtered_tracks) >= top_n:
            break
        artist = tracks.loc[i, 'artist_names']
        if artist not in seen_artists:  # Ensure different artists
            filtered_tracks.append(i)
            seen_artists.add(artist)

    return tracks.iloc[filtered_tracks][['track_name', 'artist_names', 'artist_genres', 'albums']]

# 🎨 Streamlit UI
st.title("🎵 Genre-Based Track Recommender (Different Artists)")
track_name = st.selectbox("Select a Track:", tracks['track_name'].unique())

if st.button("Get Recommendations"):
    recommendations = genre_based_recommendations(track_name, top_n=5)
    
    if "Error" in recommendations.columns:
        st.warning(recommendations["Error"].iloc[0])
    else:
        st.write("### Recommended Tracks (Different Artists, Same Genre):")
        for _, row in recommendations.iterrows():
            st.write(f"🎶 **{row['track_name']}** by {row['artist_names']} (Album: {row['albums']})")


