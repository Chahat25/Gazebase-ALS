import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')

# Input text
text = "Deep Learning Lab Manual on text data using NLTK."

# Tokenization
tokens = word_tokenize(text)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

print("Tokens:", tokens)
print("Stemmed Words:", stemmed_words)
