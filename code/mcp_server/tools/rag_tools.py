import json
import logging
from bs4 import BeautifulSoup
import httpx

from mcp_server.config import settings # Import the settings object

logger = logging.getLogger(__name__)

async def google_search(query: str) -> str:
    """
    Performs a Google search using the Serper API and returns a formatted string.
    """
    api_key = settings.SERPER_API_KEY
    if not api_key:
        logger.error("SERPER_API_KEY not found in config. Cannot perform search.")
        raise ValueError("SERPER_API_KEY is not configured.")

    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=15.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            search_results = resp.json()

        output = ""
        if "organic" in search_results:
            for item in search_results["organic"][:5]:
                output += f"{item['link']} {item['title']}\n"
        return output.strip()
    except Exception as e:
        logger.error(f"Error during Google Search: {e}")
        return "Error performing search."

async def view_text_website(url: str) -> str:
    """
    Fetches the content of a website and returns it as plain text.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=20.0), follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html_text = resp.text

        soup = BeautifulSoup(html_text, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())
    except Exception as e:
        logger.error(f"Error viewing website {url}: {e}")
        return f"Error retrieving content from {url}."
