import requests
from bs4 import BeautifulSoup

def scrape_eurlex(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n")
    with open("data/multimodal_data/eurlex_doc_1.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text

# Example usage
# scrape_eurlex("https://eur-lex.europa.eu/example")