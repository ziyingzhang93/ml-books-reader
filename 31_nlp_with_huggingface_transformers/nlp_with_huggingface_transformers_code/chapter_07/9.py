...
generator = TextGenerator()

# Example of sampling effects
prompt = "The scientist discovered"

# Using top-k sampling
top_k_text = generator.generate_text(
    prompt,
    top_k=10,
    top_p=1.0,
    max_length=50
)
print(f"Top-k sampling (k=10):\n{top_k_text}\n")

# Using nucleus (top-p) sampling
nucleus_text = generator.generate_text(
    prompt,
    top_k=0,
    top_p=0.9,
    max_length=50
)
print(f"Nucleus sampling (p=0.9):\n{nucleus_text}\n")

# Combining both
combined_text = generator.generate_text(
    prompt,
    top_k=50,
    top_p=0.95,
    max_length=50
)
print(f"Combined sampling:\n{combined_text}\n")
