import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset 
tracks = pd.read_csv(r"C:\Users\admin\Python_YT\Gdsc\track_ds.csv")

# Just cleaning up column names 
tracks.columns = tracks.columns.str.strip()
tracks.rename(columns={'names': 'track_name'}, inplace=True)

# Check if necessary columns exist before proceeding 
if 'artist_genres' in tracks.columns and 'artist_names' in tracks.columns:
    # Convert the 'artist_genres' column into a numerical format using TF-IDF
    tfidf_genre = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf_genre.fit_transform(tracks['artist_genres'].fillna(''))
    
    # Compute similarity scores between all tracks based on genre
    genre_similarity = cosine_similarity(genre_matrix, genre_matrix)
else:
    genre_similarity = None  # If data is missing, set similarity to None

# Function to recommend tracks based on genre similarity but ensuring different artists

def genre_based_recommendations(track_name, top_n=5):
    if genre_similarity is None:
        return pd.DataFrame({"Error": ["Genre data unavailable!"]})  # Handle missing genre data
    
    if track_name not in tracks['track_name'].values:
        return pd.DataFrame({"Error": ["Track not found!"]})  # Handle invalid track name input
    
    # Find the index of the selected track
    idx = tracks[tracks['track_name'] == track_name].index
    if len(idx) == 0:
        return pd.DataFrame({"Error": ["Track title not found!"]})
    
    idx = idx[0]  # Get the first match
    input_artist = tracks.loc[idx, 'artist_names']  # Store the artist of the selected track
    
    # Get similarity scores for all tracks
    sim_scores = list(enumerate(genre_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
    
    filtered_tracks = []  # Store recommended tracks
    seen_artists = {input_artist}  # Keep track of artists to avoid recommending the same one
    
    for i, score in sim_scores:
        if len(filtered_tracks) >= top_n:
            break  # Stop when we have enough recommendations
        
        artist = tracks.loc[i, 'artist_names']
        if artist not in seen_artists:  # Only add tracks from different artists
            filtered_tracks.append(i)
            seen_artists.add(artist)  # Mark this artist as seen
    
    return tracks.iloc[filtered_tracks][['track_name', 'artist_names', 'artist_genres', 'albums']]

# Streamlit UI
st.title("🎵 Genre-Based Track Recommender (Different Artists)")

# Dropdown to select a track
track_name = st.selectbox("Select a Track:", tracks['track_name'].unique())

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = genre_based_recommendations(track_name, top_n=5)
    
    if "Error" in recommendations.columns:
        st.warning(recommendations["Error"].iloc[0])  # Show warning if something went wrong
    else:
        st.write("### Recommended Tracks (Different Artists, Same Genre):")
        for _, row in recommendations.iterrows():
            # Display each recommended track nicely
            st.write(f"🎶 **{row['track_name']}** by {row['artist_names']} (Album: {row['albums']})")
