import os
import requests
import json
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

async def google_search(query: str) -> str:
    """
    Performs a Google search using the Serper API and returns a formatted string.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        logger.warning("SERPER_API_KEY not found. Returning mock search results.")
        return "http://mock.url/result1 Mock Title 1\nhttp://mock.url/result2 Mock Title 2"

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        search_results = response.json()

        # Format the results into a string
        output = ""
        if "organic" in search_results:
            for item in search_results["organic"][:5]: # Top 5 results
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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split()) # Normalize whitespace
    except Exception as e:
        logger.error(f"Error viewing website {url}: {e}")
        return f"Error retrieving content from {url}."

if __name__ == '__main__':
    import asyncio

    async def test_tools():
        # To run this test, set the SERPER_API_KEY environment variable
        # export SERPER_API_KEY='...'

        print("--- Testing google_search ---")
        search_results = await google_search("What is the SIM-ONE Framework?")
        print(search_results)

        print("\n--- Testing view_text_website ---")
        # Use a URL from the search results if available
        first_url = search_results.split('\n')[0].split(' ')[0] if search_results else None
        if first_url and first_url.startswith('http'):
            content = await view_text_website(first_url)
            print(content[:500] + "...")
        else:
            print("No valid URL to test.")

    asyncio.run(test_tools())
