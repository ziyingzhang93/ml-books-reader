import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name="t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

input_text = "This is an important message that needs accurate translation."
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
inputs = inputs.to(device)

outputs = model.generate(**inputs, max_length=512, num_beams=4*4, num_beam_groups=4,
                         num_return_sequences=4, diversity_penalty=0.8,
                         length_penalty=0.6, early_stopping=True, output_scores=True,
                         return_dict_in_generate=True)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
)
for idx, (out_tok, out_score) in enumerate(zip(outputs.sequences, transition_scores)):
    translation = tokenizer.decode(out_tok, skip_special_tokens=True)
    print(f"Translation: {translation}")
    print("token | token string   | logits  | probability")
    for tok, score in zip(out_tok[1:], out_score.cpu()):
        print(f"| {tok:5d} | {tokenizer.decode(tok):14s} | {score.numpy():.4f} "
              f"| {np.exp(score.numpy()):.2%}")
