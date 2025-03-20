import pandas as pd

books = pd.read_csv('book_details.csv')
# print(books.head())
# print(books.shape) #(10000, 23)
# print(books.columns)

# Clean up the dataset
books = books[['title', 'author', 'genres', 'description']]
books.dropna(inplace=True)
books = books.rename(columns={'original_title':'title'})
# print(books.columns)
books.to_csv("cleaned_books.csv", index=False)