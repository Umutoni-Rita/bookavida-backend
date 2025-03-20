import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv("cleaned_books.csv")
books['features'] = books['title'] + " " + books['authors']  # We combine them for similarity

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) # Computes the similarity matrix

def recommend_books(book_titles, top_n=3):
    indices = [books[books['title'].str.lower() == title.lower()].index[0] for title in book_titles]
    sim_scores = cosine_sim[indices].mean(axis=0) #Computes the average similarity for both books
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1] # Top N similar books
    return books['title'].iloc[sim_indices].tolist()

# Let's test it
print(recommend_books(["The Great Gatsby", "To Kill a Mockingbird"])) 