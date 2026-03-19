import unicodedata
import zipfile

import tokenizers

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

text_pairs = []
with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    for line in zip_ref.read("fra.txt").decode("utf-8").splitlines():
        eng, fra = normalize(line)
        text_pairs.append((eng, fra))

en_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
fr_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

# Configure pre-tokenizer to split on whitespace and punctuation, add space at beginning
# of the sentence
en_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
fr_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)

# Configure decoder: So that word boundary symbol "Ġ" will be removed
en_tokenizer.decoder = tokenizers.decoders.ByteLevel()
fr_tokenizer.decoder = tokenizers.decoders.ByteLevel()

# Train BPE for English and French using the same trainer
VOCAB_SIZE = 8000
trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[start]", "[end]", "[pad]"],
    show_progress=True
)
en_tokenizer.train_from_iterator([x[0] for x in text_pairs], trainer=trainer)
fr_tokenizer.train_from_iterator([x[1] for x in text_pairs], trainer=trainer)

en_tokenizer.enable_padding(pad_id=en_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
fr_tokenizer.enable_padding(pad_id=fr_tokenizer.token_to_id("[pad]"), pad_token="[pad]")

# Save the trained tokenizers
en_tokenizer.save("en_tokenizer.json", pretty=True)
fr_tokenizer.save("fr_tokenizer.json", pretty=True)
