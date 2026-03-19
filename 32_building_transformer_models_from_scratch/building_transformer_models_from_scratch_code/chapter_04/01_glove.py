from gensim.models import KeyedVectors

# Load pretrained GloVe embeddings
model = KeyedVectors.load_word2vec_format('glove.6B.50d.txt', binary=False,
                                          no_header=True)
# Find similar words
similar_words = model.most_similar('king')
print(similar_words)
print()

# Word analogies
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result)
