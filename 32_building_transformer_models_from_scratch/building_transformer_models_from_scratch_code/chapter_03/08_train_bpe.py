from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
print(ds)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
print(tokenizer)

tokenizer.train_from_iterator(ds["train"]["text"], trainer)
print(tokenizer)
tokenizer.save("my-tokenizer.json")

# reload the trained tokenizer
tokenizer = Tokenizer.from_file("my-tokenizer.json")
