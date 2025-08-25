import logging
from typing import Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class HIPProtocol:
    """
    A simple implementation of the Hyperlink Interpretation Protocol (HIP).
    It decides whether a given URL should be followed based on a set of rules.
    """

    def __init__(self):
        self.banned_domains = {"example.com", "somesite.org"}
        self.ignore_keywords = {"login", "signup", "about", "contact", "privacy"}
        self.allowed_schemes = {"http", "https"}

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes a URL and decides on an action.

        Args:
            data: The input data, expected to contain a 'url'.

        Returns:
            A dictionary with the decision ('action') and a 'reason'.
        """
        url = data.get("url")
        if not url:
            return {"action": "ignore", "reason": "No URL provided."}

        logger.info(f"HIP: Evaluating URL: {url}")

        try:
            parsed_url = urlparse(url)

            if parsed_url.scheme not in self.allowed_schemes:
                return {"action": "ignore", "reason": f"Scheme '{parsed_url.scheme}' is not allowed."}

            if parsed_url.netloc in self.banned_domains:
                return {"action": "ignore", "reason": f"Domain '{parsed_url.netloc}' is on the ban list."}

            for keyword in self.ignore_keywords:
                if keyword in parsed_url.path.lower():
                    return {"action": "ignore", "reason": f"URL path contains ignored keyword: '{keyword}'."}

        except Exception as e:
            logger.warning(f"HIP: Could not parse URL '{url}': {e}")
            return {"action": "ignore", "reason": f"URL could not be parsed."}

        logger.info(f"HIP: Approving URL for follow: {url}")
        return {"action": "follow", "reason": "URL passed all checks."}

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    hip = HIPProtocol()

    print("--- Testing HIP Protocol ---")

    test_urls = [
        "https://www.good-source.com/article/123",
        "https://example.com/some-page",
        "ftp://unsupported.com/file",
        "https://www.good-source.com/about-us",
        "not-a-url"
    ]

    for url in test_urls:
        result = hip.execute({"url": url})
        print(f"URL: '{url}' -> Action: {result['action']}, Reason: {result['reason']}")

    assert hip.execute({"url": "https://www.good-source.com/article/123"})['action'] == 'follow'
    assert hip.execute({"url": "https://example.com/some-page"})['action'] == 'ignore'
    assert hip.execute({"url": "https://www.good-source.com/about-us"})['action'] == 'ignore'
