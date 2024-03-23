import json
import time
from datetime import datetime
from io import BytesIO
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fp.fp import FreeProxy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader

# Proxy init
def get_proxy():
    print("Starting proxy ...")
    proxy_url = FreeProxy(
        country_id=[
            "US",
            "CA",
            "FR",
            "NZ",
            "SE",
            "PT",
            "CZ",
            "NL",
            "ES",
            "SK",
            "UK",
            "PL",
            "IT",
            "DE",
            "AT",
            "JP",
        ],
        https=True,
        rand=True,
        timeout=3,
    ).get()
    proxy_obj = {"server": proxy_url, "username": "", "password": ""}

    print(f"Proxy generated: {proxy_url}")

    return proxy_obj


def save_to_db(text, url):
    timestamp = datetime.now().isoformat()
    # Load existing data from db.json
    try:
        with open("db.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    # Create a new entry with the domain name as key
    website = {"date": timestamp, "text": text}
    new_entry = {"start_url": url, "data": website}

    # Append new entry to the data list
    data.append(new_entry)

    # Write data back to db.json
    with open("db.json", "w") as f:
        json.dump(data, f, indent=4)


def scrape_webpages(urls, proxy):
    print("Scraping text from webpages from each of the links ...")
    scraped_texts = []
    for url in urls:
        try:
            if url.endswith(".pdf"):
                response = requests.get(url, proxies=proxy)
                reader = PdfReader(BytesIO(response.content))
                number_of_pages = len(reader.pages)

                for p in range(number_of_pages):

                    page = reader.pages[p]
                    text = page.extract_text()
                    scraped_texts.append(text)
            else:
                page = requests.get(url, proxies=proxy)
                soup = BeautifulSoup(page.content, "html.parser")
                text = " ".join([p.get_text() for p in soup.find_all("p")])
                scraped_texts.append(text)

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    all_scraped_text = "\n".join(scraped_texts)
    print("Finished scraping the text from webpages!")
    return all_scraped_text


def get_domain(url):
    return urlparse(url).netloc


def get_robots_file(url, proxy):
    robots_url = urljoin(url, "/robots.txt")
    try:
        response = requests.get(robots_url, proxies=proxy)
        return response.text
    except Exception as e:
        print(f"Error fetching robots.txt: {e}")
        return None


def parse_robots(content):
    # This function assumes simple rules without wildcards, comments, etc.
    # For a full parser, consider using a library like robotparser.
    disallowed = []
    for line in content.splitlines():
        if line.startswith("Disallow:"):
            path = line[len("Disallow:") :].strip()
            disallowed.append(path)
    return disallowed


def is_allowed(url, disallowed_paths, base_domain):
    parsed_url = urlparse(url)
    if parsed_url.netloc != base_domain:
        return False
    for path in disallowed_paths:
        if parsed_url.path.startswith(path):
            return False
    return True


def scrape_site_links(url, proxy):
    visited_links = set()
    not_visited_links = set()
    to_visit = [url]
    base_domain = get_domain(url)
    disallowed_paths = parse_robots(get_robots_file(url, proxy))
    last_found_time = time.time()  # Track the last time a link was found

    while to_visit:
        # Break the loop if  30 seconds have passed without finding a new link
        if time.time() - last_found_time > 15:
            print("FINISHED scraping the links")
            break

        current_url = to_visit.pop(0)
        if current_url not in visited_links and is_allowed(
            current_url, disallowed_paths, base_domain
        ):
            visited_links.add(current_url)
            try:
                print(f"{current_url}")
                response = requests.get(current_url, proxies=proxy)
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    new_url = urljoin(current_url, link["href"])
                    if new_url not in visited_links:
                        to_visit.append(new_url)
                        last_found_time = time.time()  # Update the last found time
            except Exception as e:
                print(f" !!! COULD NOT VISIT: {current_url}")
                not_visited_links.add(current_url)

    return visited_links


class WebProcesser:
    def __init__(self) -> None:
        self.chunk_size = (500,)
        self.chunk_overlap = (100,)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None
        self.retriever = None
        self.db_path = "db.json"
        db_file = json.dumps([])
        with open(self.db_path, "w") as outfile:
            outfile.write(db_file)

    def init_db_website(self, url):
        web_text = ""
        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
                for entry in data:
                    if (
                        url in entry["start_url"]
                    ):  # ADD check for today's scraped website data, not longer
                        print("Website is already scraped today!")
                    web_text = entry["data"]["text"]
        except FileNotFoundError:
            data = []
        # Check if website already in the db
        if not web_text:
            proxy = get_proxy()
            # Scrape all the links from the given start URL using the proxy
            all_links = scrape_site_links(url, proxy)

            # Scrape the content from all the links obtained, using the proxy
            web_text = scrape_webpages(all_links, proxy)
            save_to_db(web_text, url)

        documents = self.text_splitter.split_text(str(web_text))
        self.db = Chroma.from_texts(documents, embedding=self.embedding)
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
        return True

    def get_context(self, question, chunk_size=500, chunk_overlap=100):
        """Get context from question and txt file"""
        print("Embedding model started ...")
        context = self.retriever.get_relevant_documents(question)
        print(f"Emdeggind Model returned: {context}")

        return context
