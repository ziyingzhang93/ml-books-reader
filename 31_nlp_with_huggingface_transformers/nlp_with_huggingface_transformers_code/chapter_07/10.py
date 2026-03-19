...
generator = TextGenerator()

# Example of temperature effects
prompt = "The robot carefully"

# Low temperature (more focused)
focused = generator.generate_text(
    prompt,
    temperature=0.3,
    max_length=50
)
print(f"Low temperature (0.3):\n{focused}\n")

# Medium temperature (balanced)
balanced = generator.generate_text(
    prompt,
    temperature=0.7,
    max_length=50
)
print(f"Medium temperature (0.7):\n{balanced}\n")

# High temperature (more creative)
creative = generator.generate_text(
    prompt,
    temperature=1.0,
    max_length=50
)
print(f"High temperature (1.0):\n{creative}\n")
