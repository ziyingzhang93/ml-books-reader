import os
import requests

if not os.path.exists("fra-eng.zip"):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
    response = requests.get(url)
    with open("fra-eng.zip", "wb") as f:
        f.write(response.content)
