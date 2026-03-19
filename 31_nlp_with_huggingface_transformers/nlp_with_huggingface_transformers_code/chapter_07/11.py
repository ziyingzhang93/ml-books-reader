import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BatchGenerator:
    def __init__(self, model_name="gpt2"):
        """Initialize the text generator with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use.
                              Any of: "gpt2", "gpt2-medium", "gpt2-large"
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_batch(self, prompts, **kwargs):
        """Generate text for multiple prompts efficiently.

        Args:
            prompts (list): List of input prompts
            batch_size (int): Number of prompts to process at once
            **kwargs: Additional generation parameters

        Returns:
            list: Generated texts for each prompt
        """
        inputs = self.tokenizer(prompts, padding=True, padding_side="left",
                                return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            **kwargs
        )
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return results

# Example usage of batch generation
batch_generator = BatchGenerator()
prompts = [
    "The future of AI",
    "Space exploration will",
    "In the next decade",
    "Climate change has"
]

generated_texts = batch_generator.generate_batch(
    prompts,
    max_length=100,
    temperature=0.7,
    do_sample=True,
)

for prompt, text in zip(prompts, generated_texts):
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {text}")
