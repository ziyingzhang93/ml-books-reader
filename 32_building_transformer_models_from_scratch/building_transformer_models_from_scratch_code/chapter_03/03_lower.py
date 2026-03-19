import re

text = "Hello, world! This is a test."
tokens = re.findall(r'\w+|[^\w\s]', text.lower())
print(f"Tokens: {tokens}")
