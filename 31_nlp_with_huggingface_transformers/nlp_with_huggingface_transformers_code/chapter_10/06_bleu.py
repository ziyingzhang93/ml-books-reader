import torch
import sacrebleu
from transformers import T5ForConditionalGeneration, T5Tokenizer

class MultilingualTranslator:
    def __init__(self, model_name="t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def translate(self, text, source_lang, target_lang):
        """Translate text and report the beam search scores"""
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
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=512, num_beams=4*4,
                                          num_beam_groups=4, num_return_sequences=4,
                                          diversity_penalty=0.8, length_penalty=0.6,
                                          early_stopping=True, output_scores=True,
                                          return_dict_in_generate=True)
        # Decode and return translation
        translation = [self.tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs.sequences]
        return {
            "translation": translation,
            "score": [float(score) for score in outputs.sequences_scores],
        }

sample_document = """
Machine translation has evolved significantly over the years. Early systems used
rule-based approaches that defined grammatical rules for languages.  Statistical
machine translation later emerged, using large corpora of translated texts to learn
translation patterns automatically.
"""
reference_translation = """
La traduction automatique a considérablement évolué au fil des ans. Les premiers
systèmes utilisaient des approches basées sur des règles définissant les règles
grammaticales des langues. La traduction automatique statistique est apparue plus
tard, utilisant de vastes corpus de textes traduits pour apprendre automatiquement
des modèles de traduction.
"""

translator = MultilingualTranslator("t5-base")
output = translator.translate(sample_document, "English", "French")
print(f"English: {sample_document}")
print("French:")
for text, score in zip(output["translation"], output["score"]):
    bleu = sacrebleu.corpus_bleu([text], [[reference_translation]])
    print(f"- (score: {score:.2f}, bleu: {bleu.score:.2f}) {text}")
