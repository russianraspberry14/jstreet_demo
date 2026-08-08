"""
Middle-ground scraper for building a J Street knowledge base corpus.
Run this LOCALLY on your own machine (not in a sandboxed/restricted network).

pip install requests beautifulsoup4

Strategy:
1. Try the sitemap first (fast, complete page list).
2. Filter to relevant sections (about, policy, mission, staff, faq, issue, leadership).
3. Cap at ~20 pages — enough for a solid demo corpus without heavy cleanup overhead.
4. If no sitemap is found, fall back to a short curated list of common page paths.
5. Respects robots.txt, strips nav/footer noise, skips near-empty pages.
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

BASE_URL = "https://jstreet.org"
OUTPUT_DIR = "jstreet_corpus"
MAX_PAGES = 20
CRAWL_DELAY = 1.0

HEADERS = {"User-Agent": "Mozilla/5.0 (educational RAG demo project; respectful crawler)"}

RELEVANT_KEYWORDS = ["about", "policy", "mission", "staff", "faq", "issue", "leadership", "principles"]

SKIP_PATTERNS = [
    "/wp-admin", "/wp-login", "/cart", "/checkout", "/donate", "/search",
    "/tag/", "/author/", "/feed", ".pdf", ".jpg", ".png", ".zip", "mailto:",
    "tel:", "#", "/wp-json"
]

# Backup list used only if no sitemap is found — adjust/add real paths you find by browsing the site
FALLBACK_PATHS = [
    "/about-us/",
    "/about-us/mission-principles/",
    "/about-us/staff/",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def can_fetch(url):
    try:
        rp = RobotFileParser()
        rp.set_url(urljoin(BASE_URL, "/robots.txt"))
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


def get_sitemap_urls():
    candidates = ["/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml"]
    for path in candidates:
        try:
            resp = requests.get(urljoin(BASE_URL, path), headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "xml")
            locs = [loc.text for loc in soup.find_all("loc")]

            sub_sitemaps = [l for l in locs if l.endswith(".xml")]
            page_urls = set(l for l in locs if not l.endswith(".xml"))

            for sm in sub_sitemaps[:10]:
                try:
                    r2 = requests.get(sm, headers=HEADERS, timeout=10)
                    s2 = BeautifulSoup(r2.text, "xml")
                    page_urls.update([loc.text for loc in s2.find_all("loc")])
                except Exception:
                    continue

            if page_urls:
                print(f"Found {len(page_urls)} URLs via sitemap: {path}")
                return list(page_urls)
        except Exception:
            continue
    return []


def should_skip(url):
    if not url.startswith(BASE_URL):
        return True
    return any(p in url.lower() for p in SKIP_PATTERNS)


def clean_and_save(url, index, manifest):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "form", "svg", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) < 300:
            print(f"Skipping (too short): {url}")
            return

        filename = os.path.join(OUTPUT_DIR, f"doc_{index:02d}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Source: {url}\nTitle: {title}\n\n{text}")

        manifest.append((filename, url, title))
        print(f"[{index}] Saved: {title} -> {filename}")
        time.sleep(CRAWL_DELAY)

    except Exception as e:
        print(f"Failed on {url}: {e}")


def main():
    print("Discovering pages via sitemap...")
    urls = get_sitemap_urls()

    if urls:
        filtered = [u for u in urls if any(k in u.lower() for k in RELEVANT_KEYWORDS)]
        urls = filtered if filtered else urls
    else:
        print("No sitemap found — using fallback path list.")
        urls = [urljoin(BASE_URL, p) for p in FALLBACK_PATHS]

    urls = [u for u in urls if not should_skip(u)][:MAX_PAGES]
    print(f"Will attempt to fetch {len(urls)} pages (capped at {MAX_PAGES}).")

    manifest = []
    for i, url in enumerate(urls):
        if not can_fetch(url):
            print(f"Skipping (blocked by robots.txt): {url}")
            continue
        clean_and_save(url, i, manifest)

    with open(os.path.join(OUTPUT_DIR, "_manifest.txt"), "w", encoding="utf-8") as f:
        for filename, url, title in manifest:
            f.write(f"{filename}\t{url}\t{title}\n")

    print(f"\nDone. {len(manifest)} documents saved to '{OUTPUT_DIR}/'.")
    print("Skim each file quickly — trim any leftover menu/button text before using as your RAG corpus.")


if __name__ == "__main__":
    main()