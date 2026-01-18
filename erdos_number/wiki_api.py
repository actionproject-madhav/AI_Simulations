import requests
from typing import List, Set, Optional

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WikipediaChainFinder/1.0 (https://github.com/google/antigravity; contact: user@example.com)"

from database import get_cached_links, save_links

def get_links_for_page(title: str) -> Set[str]:
    """Fetches all internal links (Namespace 0) with caching."""
    cached = get_cached_links(title, incoming=False)
    if cached is not None:
        return cached

    links = set()
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "links",
        "plnamespace": 0,
        "pllimit": "max"
    }
    headers = {"User-Agent": USER_AGENT}

    while True:
        try:
            response = requests.get(WIKI_API_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if "links" in page_data:
                    for link in page_data["links"]:
                        links.add(link["title"])
            if "continue" in data:
                params.update(data["continue"])
            else:
                break
        except Exception as e:
            print(f"Error fetching links for {title}: {e}")
            break
            
    save_links(title, links, incoming=False)
    return links

def get_incoming_links(title: str) -> Set[str]:
    """Fetches pages that link TO the given title with caching."""
    cached = get_cached_links(title, incoming=True)
    if cached is not None:
        return cached

    links = set()
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "linkshere",
        "lhnamespace": 0,
        "lhlimit": "max"
    }
    headers = {"User-Agent": USER_AGENT}
    
    while True:
        try:
            response = requests.get(WIKI_API_URL, params=params, headers=headers)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if "linkshere" in page_data:
                    for link in page_data["linkshere"]:
                        links.add(link["title"])
            if "continue" in data:
                params.update(data["continue"])
            else:
                break
        except Exception as e:
            print(f"Error fetching incoming links for {title}: {e}")
            break
            
    save_links(title, links, incoming=True)
    return links

if __name__ == "__main__":
    # Quick test
    test_title = "Python (programming language)"
    print(f"Fetching links for '{test_title}'...")
    links = get_links_for_page(test_title)
    print(f"Found {len(links)} links.")
    # Show first 10
    for l in sorted(list(links))[:10]:
        print(f" - {l}")
