import re

text = "Hello, world! This is a test."
tokens = re.findall(r'\w+|[^\w\s]', text)
print(f"Tokens: {tokens}")
