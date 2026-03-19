from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

def reformulate_query(query, n=2):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=10,
        num_return_sequences=n,
        temperature=1.5,  # High temperature for diversity
        top_k=50,
        do_sample=True
    )
    # Decode the outputs one by one
    reformulations = [tokenizer.decode(output, skip_special_tokens=True)
                      for output in outputs]
    all_queries = [query] + reformulations
    return all_queries

# Generate reformulations from an example query
query = "How do transformer-based systems process natural language?"
reformulated_queries = reformulate_query(query)
print(f"Original Query: {query}")
print("Reformulated Queries:")
for i, q in enumerate(reformulated_queries[1:], 1):
    print(f"{i}. {q}")
