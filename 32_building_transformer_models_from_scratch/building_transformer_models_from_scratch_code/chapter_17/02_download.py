import os
import requests

# Download novels from Project Gutenberg
DATASOURCE = {
    "moby_dick": "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "dracula": "https://www.gutenberg.org/ebooks/345.txt.utf-8",
    "little_women": "https://www.gutenberg.org/ebooks/37106.txt.utf-8",
    "pride_and_prejudice": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
    "alice_in_wonderland": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
    "crime_and_punishment": "https://www.gutenberg.org/ebooks/2554.txt.utf-8",
    "tom_sawyer": "https://www.gutenberg.org/ebooks/74.txt.utf-8",
    "tale_of_two_cities": "https://www.gutenberg.org/ebooks/98.txt.utf-8",
    "sherlock_holmes": "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
    "war_and_peace": "https://www.gutenberg.org/ebooks/2600.txt.utf-8",
}
for filename, url in DATASOURCE.items():
    if not os.path.exists(f"{filename}.txt"):
        response = requests.get(url)
        with open(f"{filename}.txt", "wb") as f:
            f.write(response.content)
