from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Prepare your text data
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps over the lazy fox",
    # ... more sentences
]

# Preprocess the sentences
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Train the model
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,  # dimension of the word vectors
    window=5,         # context window size
    min_count=1,      # ignore words with frequency < min_count
    workers=4,        # number of CPU cores to use
    sg=0              # 0 for CBOW, 1 for Skip-gram
)

# Save the model
model.save("word2vec.model")

# Use the model
model = Word2Vec.load("word2vec.model")
vector = model.wv['quick']  # get the vector for a word
similar_words = model.wv.most_similar('quick')
print(similar_words)
