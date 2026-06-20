# 🎬 CineMatch — Movie Recommendation System

A **Content-Based Movie Recommendation System** built with Python, Scikit-learn, and Streamlit featuring a Netflix-inspired dark UI.

## 🚀 How It Works

The model uses **Content-Based Filtering** with **Cosine Similarity**:
1. Extracts features from each movie — Genre, Cast, Director, Keywords, Overview
2. Converts them into a combined "tags" feature
3. Vectorizes using `CountVectorizer` (Bag of Words)
4. Computes cosine similarity between all movie vectors
5. Returns the top 5 most similar movies for any selected title

## 📁 Project Structure

```
movie-recommendation-system/
│
├── movies.csv                    # Dataset (2000 movies)
├── movie_recommendation.ipynb   # Data cleaning + model training notebook
├── movies_list.pkl              # Saved movie metadata (model artifact)
├── similarity.pkl               # Saved cosine similarity matrix
├── app.py                       # Streamlit frontend (Netflix dark theme)
├── requirements.txt             # Dependencies
└── README.md
```

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data cleaning & processing |
| Scikit-learn | CountVectorizer + Cosine Similarity |
| Pickle | Model serialization |
| Streamlit | Web app frontend |

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook first to generate pkl files
jupyter notebook movie_recommendation.ipynb

# 4. Launch the Streamlit app
streamlit run app.py
```

## 📊 Dataset

- **2,000 movies** across genres: Action, Drama, Thriller, Sci-Fi, Romance, Horror, Animation and more
- **Features used:** Genre, Director, Top 3 Cast, Keywords, Overview
- **Columns:** MovieID, Title, Year, Genre, Director, Cast, Language, Country, Runtime, Rating, Votes, Budget, Revenue, Keywords, Overview, Popularity

## 🎯 Features

- 🔍 Search from 2,000+ movies
- 🎬 Top 5 personalized recommendations
- 🌑 Netflix-style dark UI
- ⭐ Shows rating, year, director, cast & genre for each recommendation

---

Built as part of Data Science Summer Training · Ducat India
