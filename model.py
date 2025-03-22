import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import os
from fuzzywuzzy import fuzz, process

# Load data
books = pd.read_csv("cleaned_books.csv")
books['features'] = (
    books['title'].fillna('') + " " +
    books['author'].fillna('') + " " +
    books['genres'].fillna('') + " " +
    books['description'].fillna('')
)

# Precompute the TF-IDF matrix and cosine similarity
precomputed_files = ["cosine_sim.pkl", "tfidf_vectorizer.pkl", "svd.pkl"]
all_files_exist = all(os.path.exists(f) for f in precomputed_files)

if not all_files_exist:
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books['features'])
    svd = TruncatedSVD(n_components=100)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    cosine_sim = cosine_similarity(reduced_matrix, reduced_matrix)
    with open("cosine_sim.pkl", "wb") as f:
        pickle.dump(cosine_sim, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("svd.pkl", "wb") as f:
        pickle.dump(svd, f)
else:
    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("svd.pkl", "rb") as f:
        svd = pickle.load(f)
        
def recommend_books(book_titles, ratings, top_n=5):
    """
    Recommend books based on two input titles and their user-provided ratings.
    
    Args:
        book_titles (list): List of two book titles input by the user.
        ratings (list): List of two ratings (1-5) corresponding to the book titles.
        top_n (int): Number of recommendations to return.
    
    Returns:
        list: List of recommended book titles tailored to user ratings.
    """
    
    # Input Validation
    # Validate inputs
    if len(book_titles) != 2 or len(ratings) != 2:
        return {"error": "Please provide exactly two book titles and two ratings."}
    if not all(isinstance(r, (int, float)) and 1 <= r <= 5 for r in ratings):
        return {"error": "Ratings must be numbers between 1 and 5."}
    
    indices = []
    missing_books = []
    
    # Find indices of input books (partial matching and case-insensitive)
    for title in book_titles:
        # First try exact partial matching
        matches = books[books['title'].str.lower().str.contains(title.lower(), na=False)]
        if not matches.empty:
            indices.append(matches.index[0])  # Take first match
        else:
            # Use fuzzywuzzy to find closest match
            best_match = process.extractOne(title, books['title'], scorer=fuzz.token_sort_ratio)
            if best_match and best_match[1] >= similarity_threshold:
                matched_title = best_match[0]
                indices.append(books[books['title'] == matched_title].index[0])
            else:
                missing_books.append(title)
    
    # Handle missing titles with genre-based fallback
    if not indices:
        # If both titles are missing, find genres from input (approximate)
        genres_to_match = []
        for title in book_titles:
            best_match = process.extractOne(title, books['title'], scorer=fuzz.token_sort_ratio)
            if best_match and best_match[1] >= 50:  # Lower threshold for genre guess
                matched_title = best_match[0]
                genre = books[books['title'] == matched_title]['genres'].iloc[0]
                genres_to_match.extend(genre.split(', '))
        
        if genres_to_match:
            # Filter books by any matching genre, sort by rating, take top_n
            genre_filter = books['genres'].apply(lambda x: any(g in x for g in genres_to_match if g))
            fallback_titles = books[genre_filter].sort_values('rating', ascending=False)['title'].head(top_n).tolist()
            return {
                "warning": f"No matches found for: {', '.join(missing_books)}. Showing popular books in similar genres.",
                "titles": fallback_titles
            }
        else:
            # Ultimate fallback: top popular books
            fallback_titles = books.sort_values('rating', ascending=False)['title'].head(top_n).tolist()
            return {
                "warning": f"No matches found for: {', '.join(missing_books)}. Showing popular books instead.",
                "titles": fallback_titles
            }
        
    # Normalize ratings to use as weights (e.g., 5 -> 1.0, 1 -> 0.2)
    weights = [r / 5.0 for r in ratings]
    
    # Compute weighted similarity scores based on user ratings
    # If user rates a book higher, its similarity contributes more
    sim_scores = sum(cosine_sim[idx] * w for idx, w in zip(indices, weights)) / sum(weights)
    
    # Get top N recommendations, excluding the input books if they're in the top results
    sim_indices = sim_scores.argsort()[-top_n-2:-1][::-1]  # Exclude top 2 (likely inputs)
    # Convert input titles to lowercase for case-insensitive comparison
    input_titles_lower = [t.lower() for t in book_titles]
    
    # Filter out input books and take top_n recommendations
    recommended_titles = []
    for idx in sim_indices:
        title = books['title'].iloc[idx]
        if title.lower() not in input_titles_lower:  # Exclude input books
            recommended_titles.append(title)
        if len(recommended_titles) >= top_n:  # Stop once we have enough
            break
    
    return recommended_titles

# Testing the function locally
if __name__ == "__main__":
    test_recommendations = recommend_books(
        book_titles=["the mark of athena", "Lord of the rings"],
        ratings=[4, 4],  # User likes The Hobbit more than 1984
        top_n=5
    )
    print(test_recommendations)
