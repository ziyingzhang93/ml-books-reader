...
# Create a text generator instance
generator = TextGenerator()

# Example 1: Basic text generation
prompt = "The future of artificial intelligence will"
generated_text = generator.generate_text(prompt)
print(f"Generated text:\n{generated_text}\n")

# Example 2: More creative generation with higher temperature
creative_text = generator.generate_text(
    prompt="Once upon a time",
    temperature=0.9,
    max_length=200
)
print(f"Creative generation:\n{creative_text}\n")

# Example 3: More focused generation with lower temperature
focused_text = generator.generate_text(
    prompt="The benefits of machine learning include",
    temperature=0.5,
    max_length=150
)
print(f"Focused generation:\n{focused_text}\n")
