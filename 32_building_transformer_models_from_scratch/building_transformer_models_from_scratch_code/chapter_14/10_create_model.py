import torch
import torch.nn as nn
import tokenizers

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        batch_size, target_len = target_seq.shape
        outputs = []
        _enc_out, hidden, cell = self.encoder(input_seq)
        dec_in = target_seq[:, :1]
        for t in range(target_len-1):
            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            pred = pred[:, -1:, :]
            outputs.append(pred)
            dec_in = torch.cat([dec_in, pred.argmax(dim=2)], dim=1)
        outputs = torch.cat(outputs, dim=1)
        return outputs

en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emb_dim = 256
hidden_dim = 256
num_layers = 1
enc_vocab = en_tokenizer.get_vocab_size()
dec_vocab = fr_tokenizer.get_vocab_size()


encoder = EncoderLSTM(enc_vocab, emb_dim, hidden_dim, num_layers).to(device)
decoder = DecoderLSTM(dec_vocab, emb_dim, hidden_dim, num_layers).to(device)
model = Seq2SeqLSTM(encoder, decoder).to(device)
print(model)
