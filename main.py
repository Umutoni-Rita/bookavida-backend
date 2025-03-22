from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from model import recommend_books
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/recommend")
def get_recommendations(book1: str, book2: str, rating1: int, rating2: int):
    # Get recommendations
    recommended = recommend_books([book1, book2], [rating1, rating2], top_n=5)
    
    # Handle error or warning cases from the model
    if isinstance(recommended, dict):
        if "error" in recommended:
            raise HTTPException(status_code=400, detail=recommended["error"])
        if "warning" in recommended:
            return recommended  # Return warning with fallback titles
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    results = []
    for title in recommended:
        url = f"https://www.googleapis.com/books/v1/volumes?q={title}&key={api_key}"
        try:
            response = requests.get(url).json()
            item = response['items'][0]['volumeInfo']
            summary = item.get('description', 'No summary available')
            cover = item.get('imageLinks', {}).get('thumbnail', '')
            authors = item.get('authors', ['Author information unavailable'])
            results.append({"title": title, "summary": summary, "cover": cover, "authors": authors}, )
        except Exception as e:
            results.append({"title": title, "summary": f"Error fetching summary: {str(e)}"})
            
    
    return {"recommendations": results}