import logging
import asyncio
from typing import List, Set
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from mcp_server.protocols.hip.hip import HIPProtocol

logger = logging.getLogger(__name__)

class RAGManager:
    """
    A manager to handle recursive, governed Retrieval-Augmented Generation.
    """

    def __init__(self):
        self.hip_protocol = HIPProtocol()

    async def perform_research(self, topic: str, num_sources: int = 2, depth: int = 1) -> str:
        """
        Performs research on a given topic and returns a consolidated context.

        Args:
            topic: The topic to research.
            num_sources: The number of top search results to use as seed URLs.
            depth: The recursion depth for following links.

        Returns:
            A string containing the consolidated research material.
        """
        logger.info(f"RAGManager: Starting research for topic: '{topic}' at depth {depth}.")

        # --- Initial Seed URL Search ---
        try:
            search_results_str = await google_search(f"in-depth article about {topic}")
            seed_urls = [line.split(' ')[0] for line in search_results_str.strip().split('\n') if line.startswith('http')]
        except Exception as e:
            logger.error(f"RAGManager: Initial google_search failed: {e}")
            return "Could not perform initial search."

        if not seed_urls:
            logger.warning("RAGManager: No seed URLs found.")
            return "No relevant information found during web research."

        # --- Recursive Content Retrieval ---
        visited_urls: Set[str] = set()
        research_context = await self._recursive_fetch(seed_urls[:num_sources], depth, visited_urls)

        if not research_context:
            return "Could not retrieve content from any web sources."

        return research_context

    async def _recursive_fetch(self, urls_to_visit: List[str], depth: int, visited_urls: Set[str]) -> str:
        """
        Recursively fetches content from URLs, governed by the HIP protocol.
        """
        if depth < 0 or not urls_to_visit:
            return ""

        # Fetch content from the current list of URLs
        tasks = {url: asyncio.create_task(view_text_website(url)) for url in urls_to_visit if url not in visited_urls}
        if not tasks:
            return ""

        done, _ = await asyncio.wait(tasks.values())

        current_level_context = ""
        next_level_urls: List[str] = []

        for url, task in tasks.items():
            visited_urls.add(url)
            try:
                html_content = task.result()
                soup = BeautifulSoup(html_content, 'html.parser')

                # Extract text content
                text_content = soup.get_text(separator=' ', strip=True)
                current_level_context += f"--- Source: {url} ---\n"
                current_level_context += text_content[:2000] # Truncate
                current_level_context += "\n\n"

                # Extract links for next level of recursion
                if depth > 0:
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(url, link['href'])
                        # Use HIP to govern link following
                        if self.hip_protocol.execute({"url": full_url})["action"] == "follow":
                            if full_url not in visited_urls:
                                next_level_urls.append(full_url)
            except Exception as e:
                logger.warning(f"RAGManager: Failed to process URL {url}: {e}")

        # Recursively fetch content from the approved links
        deeper_context = await self._recursive_fetch(next_level_urls, depth - 1, visited_urls)

        return current_level_context + deeper_context

if __name__ == '__main__':
    async def main():
        logging.basicConfig(level=logging.INFO)
        rag_manager = RAGManager()

        topic = "the SIM-ONE framework for AI governance"
        print(f"--- Performing recursive research on: '{topic}' ---")

        try:
            # Mock the tools for local testing
            async def mock_google_search(query):
                return "https://www.mocksite.com/page1\nhttps://www.anothersite.com/articleA"

            async def mock_view_text_website(url):
                if url == "https://www.mocksite.com/page1":
                    return "<html><body>Page 1 content. <a href='/page2'>Link to Page 2</a></body></html>"
                if url == "https://www.mocksite.com/page2":
                    return "<html><body>Page 2 content. Deeper research here.</body></html>"
                if url == "https://www.anothersite.com/articleA":
                    return "<html><body>Article A content. <a href='https://example.com/login'>Ignore this link</a></body></html>"
                return ""

            # Replace the global tools with mocks for the test
            global google_search, view_text_website
            google_search = mock_google_search
            view_text_website = mock_view_text_website

            context = await rag_manager.perform_research(topic, depth=1)
            print("\n--- Research Context ---")
            print(context)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

    asyncio.run(main())
