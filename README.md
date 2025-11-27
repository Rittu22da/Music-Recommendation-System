<img width="1911" height="906" alt="image" src="https://github.com/user-attachments/assets/26967260-c08e-4da7-a09b-7eb5e264650e" />
# Music-Recommendation-System
A machine-learning powered Song Recommendation System built using Python, Flask, and K-Means clustering. The app takes a song name as input and recommends similar songs based on audio feature similarity.
🚀 Features

Input any song name from the dataset

Automatically extracts audio feature vectors

Uses K-Means Clustering to group similar songs

Computes similarity using Euclidean distance

Shows Top 5 Recommended Songs

Clean and responsive UI with Bootstrap

Fully functional Flask backend

🧠 How It Works

Dataset (clustered_df.csv) contains audio features + metadata

Features are scaled using StandardScaler

K-Means model assigns each song to a cluster

When a user enters a song:

The model retrieves its feature vector

Calculates similarity with other songs

Sorts results by distance

Flask renders the recommendations to the UI

🛠️ Tech Stack

Backend: Python, Flask
Machine Learning: Scikit-learn, Pandas, NumPy
Frontend: HTML, CSS, Bootstrap, Jinja2
Data: CSV dataset with audio features

📂 Project Structure
📁 Music-Recommender/
│── app.py                # Flask backend
│── index.html            # Frontend UI (templates)
│── clustered_df.csv      # Dataset with audio features
│── static/               # CSS, JS, assets (optional)
│── README.md             # Project documentation


▶️ Running the Project Locally
1️⃣ Install Dependencies
pip install flask pandas numpy scikit-learn

2️⃣ Run the Flask App
python app.py

3️⃣ Open in Browser
http://127.0.0.1:5000/


🧪 Example Output

When you enter a song, the system returns a list like:

Ordinary People — John Legend (2005)
The Island — Paul Brady (1985)
That’s the Way It Is — Jimmy Giuffre (1957)


📈 Future Improvements

Add Spotify API Integration

Add album cover images

Add genre-based similarity

Deploy online (Render/Heroku/AWS)


📜 License

This project is open-source under the MIT License.


