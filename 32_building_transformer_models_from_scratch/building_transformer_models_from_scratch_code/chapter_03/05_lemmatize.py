import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# download the necessary resources if haven't done so
nltk.download('wordnet')

text = "These models may become unstable quickly if not initialized."
lemmatizer = WordNetLemmatizer()
words = word_tokenize(text)
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized_words)
