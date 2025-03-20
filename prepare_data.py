import pandas as pd

books = pd.read_csv('books.csv')
print(books.head())
# print(books.shape) #(10000, 23)

# Clean up the dataset
books = books[['book_id', 'title', 'authors']]
books.dropna(inplace=True)
books.to_csv("cleaned_books.csv", index=False)