import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class MultilingualTranslator:
    def __init__(self, model_name="t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def translate(self, text, source_lang, target_lang):
        """Translate text from source language to target language"""
        # Make sure the source and target languages are supported
        supported_lang = ["English", "French", "German", "Spanish"]
        if source_lang not in supported_lang:
            raise ValueError(f"Unsupported source language: {source_lang}")
        if target_lang not in supported_lang:
            raise ValueError(f"Unsupported target language: {target_lang}")
        # Prepare the input text
        task_prefix = f"translate {source_lang} to {target_lang}"
        input_text = f"{task_prefix}: {text}"
        # Tokenize and generate translation
        inputs = self.tokenizer(input_text, return_tensors="pt",
                                max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        outputs = self.model.generate(**inputs, max_length=512, num_beams=4,
                                      length_penalty=0.6, early_stopping=True)
        # Decode and return translation
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

en_text = "Hello, how are you today?"
es_text = "¿Cómo estás hoy?"
translator = MultilingualTranslator("t5-base")

translation = translator.translate(en_text, "English", "French")
print(f"English: {en_text}")
print(f"French: {translation}")
print()

translation = translator.translate(en_text, "English", "German")
print(f"English: {en_text}")
print(f"German: {translation}")
print()

translation = translator.translate(es_text, "Spanish", "English")
print(f"Spanish: {es_text}")
print(f"English: {translation}")
