from fastapi import FastAPI
import requests # to fetch from Google Books API
from model import recommend_books
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/recommend")
def get_recommendations(book1: str, book2: str):
    # Call the function to get recommendations
    recommended_titles = recommend_books([book1, book2])
    
    api_key = os.getenv("GOOGLE_API_KEY")  # Load from environment
    if not api_key:
        return {"error": "Google API key not configured"}
    
    results=[]
    for title in recommended_titles:
        url = f"https://www.googleapis.com/books/v1/volumes?q={title}&key={api_key}"
        response = requests.get(url).json()
        summary = response['items'][0]['volumeInfo'].get('description', 'No summary available')
        results.append({"title": title, "summary": summary})
        
    return {"Recommendations" : results}
    