import os
import json
import importlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ProtocolManager:
    """
    Manages the discovery, loading, and lifecycle of cognitive protocols.
    """

    def __init__(self, protocol_dir: str = "mcp_server/protocols"):
        self.protocol_dir = protocol_dir
        self.protocols: Dict[str, Any] = {}
        self.scan_protocols()

    def scan_protocols(self):
        """
        Scans the protocol directory to find and register available protocols.
        """
        logger.info(f"Scanning for protocols in {self.protocol_dir}...")
        for root, _, files in os.walk(self.protocol_dir):
            if "protocol.json" in files:
                manifest_path = os.path.join(root, "protocol.json")
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    protocol_name = manifest.get("name")
                    if protocol_name:
                        self.protocols[protocol_name] = manifest
                        logger.info(f"  - Found protocol: {protocol_name}")

    def get_protocol(self, name: str) -> Any:
        """
        Loads and returns an instance of the specified protocol.

        Args:
            name: The name of the protocol to load.

        Returns:
            An instance of the protocol class, or None if not found.
        """
        manifest = self.protocols.get(name)
        if not manifest:
            logger.error(f"Protocol '{name}' not found.")
            return None

        entry_point = manifest.get("entryPoint")
        if not entry_point:
            logger.error(f"No entryPoint specified for protocol '{name}'.")
            return None

        try:
            module_path, class_name = entry_point.rsplit('.', 1)
            module = importlib.import_module(module_path)
            protocol_class = getattr(module, class_name)
            logger.info(f"Loading protocol '{name}' from {entry_point}")
            return protocol_class()
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading protocol '{name}': {e}")
            return None

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    pm = ProtocolManager()

    logger.info("\nAvailable protocols:")
    for name in pm.protocols:
        logger.info(f"- {name}")

    logger.info("\nLoading REP protocol...")
    rep_protocol = pm.get_protocol("ReasoningAndExplanationProtocol")

    if rep_protocol:
        logger.info("REP protocol loaded successfully.")
        # Example of using the loaded protocol
        facts = ["has_feathers", "flies", "lays_eggs"]
        rules = [
            (["has_feathers"], "is_bird"),
            (["flies", "is_bird"], "is_flying_bird"),
            (["is_bird", "lays_eggs"], "is_oviparous_bird")
        ]
        result = rep_protocol.execute(facts, rules)
        logger.info("\nREP execution result:")
        logger.info(f"Conclusions: {result['conclusions']}")
        logger.info("Explanation:")
        for step in result['explanation']:
            logger.info(f"- {step}")
