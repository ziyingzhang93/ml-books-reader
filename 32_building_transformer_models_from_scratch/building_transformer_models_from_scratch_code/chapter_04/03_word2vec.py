import torch
import torch.nn as nn
import torch.optim as optim

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

# Prepare your text data
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps over the lazy fox",
    # ... more sentences
]

# Create a dataset for training
skipgram_size = 2
dataset = []
vocab = set()
for sentence in sentences:
    tokens = sentence.split()
    vocab.update(tokens)
    for i in range(len(tokens)):
        context = tokens[i-skipgram_size:i] + tokens[i+1:i+skipgram_size+1]
        target = tokens[i]
        dataset.append((context, target))

vocab_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
vocab_size = len(vocab)

# Training setup
embedding_dim = 50
model = Word2VecModel(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for context, target in dataset:
        context_idx = [vocab_to_idx[word] for word in context]
        target_idx = [vocab_to_idx[target]] * len(context)
        optimizer.zero_grad()
        output = model(torch.tensor(target_idx))
        loss = criterion(output, torch.tensor(context_idx))
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), "word2vec.pt")
