from functools import lru_cache
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class AutoComplete:
    def __init__(self, model_name="gpt2"):
        """Initialize the auto-complete system."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def get_completion(self, text, max_length=50):
        """Generate completion for the input text."""
        print("**** Completion:", text)
        # Encode the input text
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attn_masks = inputs["attention_mask"].to(self.device)

        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attn_masks,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

        # Decode and extract completion
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = full_text[len(text):]

        return completion

class CachedAutoComplete(AutoComplete):
    def __init__(self, cache_size=1000, **kwargs):
        """Initialize with caching support."""
        super().__init__(**kwargs)
        self.get_completion = lru_cache(maxsize=cache_size)(self.get_completion)

class OptimizedAutoComplete(CachedAutoComplete):
    def __init__(self, **kwargs):
        """Initialize with optimizations."""
        super().__init__(**kwargs)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            self.model = self.model.half()  # Use FP16 on GPU

        # use eval mode and cuda graphs
        self.model.eval()

    def preprocess_batch(self, texts):
        """Efficiently process multiple texts."""
        # Tokenize all texts at once
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return inputs.to(self.device)

    def generate_batch(self, texts, max_length=50):
        """Generate completions for multiple texts."""
        # Preprocess batch
        inputs = self.preprocess_batch(texts)

        # Generate completions
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

        # Decode completions
        completions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract new text
        results = []
        for text, completion in zip(texts, completions):
            results.append(completion[len(text):])

        return results

# Example: Optimized batch completion
optimized_complete = OptimizedAutoComplete()
texts = [
    "Machine learning is",
    "Deep neural networks can",
    "The training process involves"
]
completions = optimized_complete.generate_batch(texts)
for text, completion in zip(texts, completions):
    print(f"\nInput: {text}")
    print(f"Completion: {completion}")
