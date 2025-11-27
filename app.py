from flask import Flask , render_template , request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import StandardScaler
# app
app= Flask(__name__)

# load save data
df=pd.read_csv("clustered_df.csv")




def recommend_songs(song_name, df, num_recommendations=5, use_cluster=True):
    """
    Recommend songs similar to song_name using cosine similarity on selected audio features.
    - Ensures the index used for similarity is the positional index inside the same_cluster_songs
      DataFrame (0..n-1) so it never goes out of bounds.
    - Normalizes features before computing cosine similarity.
    """

    # Validate song presence
    if song_name not in df['name'].values:
        raise ValueError(f"'{song_name}' not found in df['name']. Sample names: {df['name'].sample(5).tolist()}")

    # Choose features (tweak as needed)
    numerical_features = [
        'danceability','energy','loudness','speechiness','acousticness',
        'instrumentalness','liveness','valence','tempo','duration_ms'
    ]
    missing = [c for c in numerical_features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    # Narrow by cluster if present and requested
    if use_cluster and 'Cluster' in df.columns:
        song_cluster = df.loc[df['name'] == song_name, 'Cluster'].values[0]
        same_cluster_songs = df[df['Cluster'] == song_cluster].copy()
    else:
        same_cluster_songs = df.copy()

    # Reset index so positions align with similarity matrix rows (0 .. n-1)
    same_cluster_songs = same_cluster_songs.reset_index(drop=True)

    # Find positional index of the input song _within_ same_cluster_songs
    matches = np.where(same_cluster_songs['name'].values == song_name)[0]
    if len(matches) == 0:
        # unlikely, but safe guard
        raise ValueError("Song not found inside the selected cluster/subset.")
    song_pos = int(matches[0])   # positional index (0..n-1)

    # Build feature matrix and normalize
    X = same_cluster_songs[numerical_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute cosine similarity matrix
    similarity = cosine_similarity(X_scaled, X_scaled)  # shape (n, n)

    # Sanity check: ensure song_pos is within bounds
    n = similarity.shape[0]
    if not (0 <= song_pos < n):
        raise IndexError(f"Positional song index {song_pos} out of bounds for similarity matrix with size {n}.")

    # Get similarity scores (highest first) and exclude the song itself
    sim_idx_desc = np.argsort(similarity[song_pos])[::-1]  # descending order
    sim_idx_desc = [i for i in sim_idx_desc if i != song_pos]

    # Adjust requested number if dataset smaller
    max_possible = len(sim_idx_desc)
    if num_recommendations > max_possible:
        num_recommendations = max_possible

    top_idx = sim_idx_desc[:num_recommendations]

    recommendations = same_cluster_songs.iloc[top_idx][['name','artists','year']]

    return recommendations

#routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend",methods=['GET','POST'])
def recommend():
    recommendations=[]

    if request.method=="POST":
        song_name = request.form.get("song_name")
        try:
            rec_df = recommend_songs(song_name, df)
         # convert DataFrame -> list of dicts for Jinja
            recommendations = rec_df.to_dict(orient='records') if hasattr(rec_df, "to_dict") else rec_df
        except Exception as e:
            recommendations= [{'name':'error','artists':'invalid','year':2000}]
    return render_template('index.html', recommendations= recommendations)
#python app call
if __name__=="__main__":
    app.run(debug=True)