import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        """Initialize the text generator with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use.
                              Any of: 'gpt2', 'gpt2-medium', 'gpt2-large'
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def generate_text(self, prompt, max_length=100, temperature=0.7,
                      top_k=50, top_p=0.95):
        """Generate text based on the input prompt.

        Args:
            prompt (str): Input text to continue from
            max_length (int): Maximum length of generated text
            temperature (float): Controls randomness in generation
            top_k (int): Number of highest probability tokens to consider
            top_p (float): Cumulative probability threshold for token filtering

        Returns:
            str: Generated text including the prompt
        """
        try:
            # Encode the input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Configure generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 2,
                "do_sample": True,
            }

            # Generate text
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

            # Decode and return the generated text
            generated_text = self.tokenizer.decode(
                output_sequences[0],
                skip_special_tokens=True
            )
            return generated_text
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            return prompt
