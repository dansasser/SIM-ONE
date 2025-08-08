import logging
import asyncio
from typing import List

# The tools are globally available in the agent's environment
# from mcp_server.tools import google_search, view_text_website

logger = logging.getLogger(__name__)

class RAGManager:
    """
    A manager to handle Retrieval-Augmented Generation by searching the web.
    """

    async def perform_research(self, topic: str, num_sources: int = 2) -> str:
        """
        Performs research on a given topic and returns a consolidated context.

        Args:
            topic: The topic to research.
            num_sources: The number of top search results to use as sources.

        Returns:
            A string containing the consolidated research material.
        """
        logger.info(f"RAGManager: Starting research for topic: '{topic}'")

        search_queries = [
            f"in-depth analysis of {topic}",
            f"key aspects of {topic}",
            f"criticism and viewpoints on {topic}"
        ]

        all_urls: List[str] = []
        for query in search_queries:
            try:
                search_results_str = await google_search(query)
                urls = [line.split(' ')[0] for line in search_results_str.strip().split('\n') if line.startswith('http')]
                all_urls.extend(urls)
            except Exception as e:
                logger.warning(f"RAGManager: Search query '{query}' failed: {e}")

        # Remove duplicate URLs while preserving order
        unique_urls = list(dict.fromkeys(all_urls))

        if not unique_urls:
            logger.warning("RAGManager: No URLs found for any search query.")
            return "No relevant information found during web research."

        # Fetch content from the top URLs
        tasks = [view_text_website(url) for url in unique_urls[:num_sources]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        research_context = ""
        for i, res in enumerate(results):
            url = unique_urls[i]
            if isinstance(res, Exception):
                logger.warning(f"RAGManager: Failed to retrieve content from {url}: {res}")
            else:
                logger.info(f"RAGManager: Successfully retrieved content from {url}")
                research_context += f"--- Source: {url} ---\n"
                research_context += res[:2000] # Truncate each source to keep context manageable
                research_context += "\n\n"

        if not research_context:
            return "Could not retrieve content from any web sources."

        return research_context

if __name__ == '__main__':
    async def main():
        logging.basicConfig(level=logging.INFO)
        rag_manager = RAGManager()

        topic = "the impact of AI on modern software development"
        print(f"--- Performing research on: '{topic}' ---")

        try:
            context = await rag_manager.perform_research(topic)
            print("\n--- Research Context ---")
            print(context[:500] + "...")
        except NameError as e:
            print(f"\nCaught expected error because tools are not in local scope: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

    asyncio.run(main())
