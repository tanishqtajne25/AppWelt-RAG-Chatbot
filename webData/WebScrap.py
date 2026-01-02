from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

# ================= CONFIG =================
BASE_URL = "https://vgold.co.in"
MAX_PAGES = 20
OUTPUT_DIR = "output"
# =========================================

visited = set()
documents = []


def normalize(url):
    parsed = urlparse(url)
    return parsed.scheme + "://" + parsed.netloc + parsed.path.rstrip("/")


def is_valid_page(url):
    blocked_keywords = [
        "/author/",
        "/admin",
        "/wp-",
        "/tag/",
        "/category/",
        "/feed",
        "/page/",
        "#"
    ]
    for key in blocked_keywords:
        if key in url:
            return False
    return True


def clean_text(text):
    noise_phrases = [
        "Skip to content",
        "LOG IN",
        "Chatbot",
        "WhatsApp",
        "Go to Top",
        "Submit",
        "Thank you for your message",
        "There was an error trying to send your message",
        "Page load link",
        "$(document).ready",
        "Mobile App",
        "Email",
        "Phone"
    ]

    for phrase in noise_phrases:
        text = text.replace(phrase, "")

    # collapse whitespace
    text = " ".join(text.split())
    return text


def crawl():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        )

        queue = [BASE_URL]

        while queue and len(visited) < MAX_PAGES:
            url = normalize(queue.pop(0))

            if url in visited:
                continue
            if not is_valid_page(url):
                continue

            visited.add(url)
            print(f"Scraping ({len(visited)}): {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(3000)

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                # remove boilerplate elements
                for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                    tag.decompose()

                text = soup.get_text(separator=" ", strip=True)
                text = clean_text(text)

                if len(text) > 300:
                    documents.append({
                        "source": url,
                        "content": text
                    })

                # discover internal links
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/") or BASE_URL in href:
                        full = normalize(urljoin(BASE_URL, href))
                        if BASE_URL in full and full not in visited:
                            queue.append(full)

            except Exception as e:
                print(f"Failed: {url} -> {e}")

        browser.close()


def save_to_txt():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, doc in enumerate(documents, start=1):
        path = f"{OUTPUT_DIR}/page_{i}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE URL:\n{doc['source']}\n\n")
            f.write("CONTENT:\n")
            f.write(doc["content"])


# ================ RUN =====================
crawl()
save_to_txt()

print("\nDONE")
print("Pages visited:", len(visited))
print("Pages saved:", len(documents))
print(f"Files stored in: {OUTPUT_DIR}/")
