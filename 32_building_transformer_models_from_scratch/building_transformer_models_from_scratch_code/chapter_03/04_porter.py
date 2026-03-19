import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# download the necessary resources if haven't done so
nltk.download('punkt_tab')

text = "These models may become unstable quickly if not initialized."
stemmer = PorterStemmer()
words = word_tokenize(text)
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)
