import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import os

# Load and preprocess data
def load_data(path="books.csv"):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)  # fill NaNs in all text columns
    df["content"] = df["title"] + " " + df["author"] + " " + df["genre"] + " " + df["description"]
    return df

def build_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["content"])
    return tfidf, matrix

def compute_similarity(matrix):
    return cosine_similarity(matrix)

def recommend_books(title, df, sim_matrix, top_n=5):
    if title not in df["title"].values:
        return []
    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    indices = [i for i, _ in scores[1: top_n+1]]
    return df.iloc[indices][["title", "author", "genre"]].to_dict("records")

# Flask app setup
app = Flask(__name__)

books_df = load_data()
tfidf_matrix_builder, tfidf_matrix = build_tfidf_matrix(books_df)
sim_matrix = compute_similarity(tfidf_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/titles')
def titles():
    titles = books_df["title"].tolist()
    return jsonify(titles)

@app.route('/library')
def library():
    # Convert books_df to a list of dicts for Jinja
    books = books_df[["title","author","genre","description"]].to_dict(orient="records")
    return render_template('library.html', books=books)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    recs = recommend_books(data.get('title',''), books_df, sim_matrix)
    return jsonify(recs)

@app.route('/custom_recommend', methods=['POST'])
def custom_recommend():
    data = request.get_json()

    new_content = (
        data.get('title', '') + ' ' +
        data.get('author', '') + ' ' +
        data.get('genre', '') + ' ' +
        data.get('description', '')
    )

    if not new_content.strip():
        return jsonify([])

    new_vector = tfidf_matrix_builder.transform([new_content])
    similarities = cosine_similarity(new_vector, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-5:][::-1]
    recommendations = books_df.iloc[top_indices][["title", "author", "genre"]].to_dict("records")

    return jsonify(recommendations)

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

if __name__ == '__main__':
    app.run(debug=True)