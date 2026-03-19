from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class AutoComplete:
    def __init__(self, model_name="gpt2"):
        """Initialize the auto-complete system."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def get_completion(self, text, max_length=50):
        """Generate completion for the input text."""
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

# using autocomplete to see what we get
auto_complete = AutoComplete()
text = "The future of artificial"
completion = auto_complete.get_completion(text)
print(f"Input: {text}")
print(f"Completion: {completion}")
