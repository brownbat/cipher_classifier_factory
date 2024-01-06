import requests
import os
import random
import re
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Constants
GUTENBERG_TOP_100_URL = "https://www.gutenberg.org/browse/scores/top"
LOCAL_LIBRARY_PATH = "local_library"
BOOK_IDS_FILE = "data/book_ids.json"
FAILED_BOOK_IDS_FILE = "data/failed_book_ids.json"


def save_book_ids(book_ids):
    """
    Saves book IDs and the current date to a local JSON file.
    """
    data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'book_ids': book_ids
    }
    with open(BOOK_IDS_FILE, 'w', encoding='utf-8') as file:
        json.dump(data, file)


def load_book_ids():
    """
    Loads book IDs from the local JSON file if available and not outdated.
    """
    if os.path.exists(BOOK_IDS_FILE):
        with open(BOOK_IDS_FILE, 'r', encoding='utf-8') as file:
            data = json.load(file)
            saved_date = datetime.strptime(data['date'], "%Y-%m-%d")
            if saved_date.date() >= (datetime.now() - timedelta(days=1)).date():
                return data['book_ids']
    return None


def save_failed_book_ids(failed_book_ids):
    with open(FAILED_BOOK_IDS_FILE, 'w', encoding='utf-8') as file:
        json.dump(failed_book_ids, file)


def load_failed_book_ids():
    if os.path.exists(FAILED_BOOK_IDS_FILE):
        with open(FAILED_BOOK_IDS_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    return []


def get_top_books_ids(limit=10):
    """
    Fetches IDs of top books from Project Gutenberg's 'Top 100 EBooks last 30 days' section.
    """
    response = requests.get(GUTENBERG_TOP_100_URL)
    soup = BeautifulSoup(response.content, 'html.parser')

    top_30_section = soup.find('h2', id='books-last30').find_next('ol')
    book_links = top_30_section.find_all('a', href=True)

    book_ids = [link['href'].split('/')[-1] for link in book_links]

    return book_ids[:limit]


def fetch_and_store_top_books_ids():
    """
    Fetches and stores the top book IDs
    if they are not already stored or outdated.
    """
    book_ids = load_book_ids()
    if not book_ids:
        book_ids = get_top_books_ids()
        save_book_ids(book_ids)
    return book_ids


def download_text(url):
    """
    Downloads and returns the text from a given Project Gutenberg URL.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def preprocess_and_save_text(text, book_id):
    """
    Preprocesses the text by extracting the main content,
    converting to lowercase, and keeping alphabetic characters only.
    Then, saves it to a file.
    """
    main_text = extract_main_content(text)
    processed_text = re.sub('[^a-z]+', '', main_text.lower())

    local_path = os.path.join(LOCAL_LIBRARY_PATH, f"{book_id}_processed.txt")
    with open(local_path, 'w', encoding='utf-8') as file:
        file.write(processed_text)


def extract_main_content(text):
    """
    Extracts the main content of the book from the given text.
    """
    start_pattern = r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK"
    end_pattern = r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK"

    start_match = re.search(start_pattern, text)
    end_match = re.search(end_pattern, text)

    if start_match and end_match:
        start_index = start_match.end()
        end_index = end_match.start()
        return text[start_index:end_index].strip()

    # Handle error or return a default value if markers are not found
    print("Could not find Project Gutenberg markers, returning full text")
    return text
    

def download_and_store_book(book_id, failed_book_ids):
    """
    Downloads book based on ID, preprocesses, and stores locally.
    """
    if not os.path.exists(LOCAL_LIBRARY_PATH):
        os.makedirs(LOCAL_LIBRARY_PATH)

    canonical_url = (
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt")
    processed_local_path = os.path.join(
        LOCAL_LIBRARY_PATH, f"{book_id}_processed.txt")

    if not os.path.exists(processed_local_path):
        try:
            text = download_text(canonical_url)
            preprocess_and_save_text(text, book_id)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download book ID {book_id}: {e}")
            failed_book_ids.append(book_id)
            save_failed_book_ids(failed_book_ids)
            return False
    return True


def get_random_text_passage(length):
    """
    Selects a random book from the list of top 100 IDs,
    downloads and preprocesses it if necessary,
    and extracts a random text passage.
    """
    book_ids = fetch_and_store_top_books_ids()
    while True:
        random_book_id = random.choice(book_ids)
        failed_book_ids = load_failed_book_ids()
        if random_book_id in failed_book_ids:
            continue
        processed_file_path = os.path.join(
            LOCAL_LIBRARY_PATH, f"{random_book_id}_processed.txt")

        if not os.path.exists(processed_file_path):
            if not download_and_store_book(random_book_id, failed_book_ids):
                continue

        with open(processed_file_path, 'r', encoding='utf-8') as file:
            text = file.read()

            if len(text) < length:
                raise ValueError(
                    "The extracted text is shorter than the requested length.")
            else:
                start = random.randint(0, len(text) - length)
                return text[start:start + length]

# Example Usage
if __name__ == "__main__":
    failed_book_ids = load_failed_book_ids()
    samples = []
    # Specify the length of the text passage in characters
    text_length = 500
    random_passage = get_random_text_passage(text_length)
    print(random_passage)
    save_failed_book_ids(failed_book_ids)
