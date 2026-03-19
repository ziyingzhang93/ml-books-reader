from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
tokenizer = SentencePieceUnigramTokenizer()

tokenizer.train_from_iterator(ds["train"]["text"])
tokenizer.save("my-tokenizer.json")
