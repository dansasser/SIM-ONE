import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WorkflowTemplateManager:
    """
    Manages loading and retrieving workflow templates.
    """

    def __init__(self, template_file: str = "mcp_server/workflow_templates.json"):
        self.template_file = template_file
        self.templates: Dict[str, Any] = self._load_templates()

    def _load_templates(self) -> Dict[str, Any]:
        """
        Loads the workflow templates from the JSON file.
        """
        try:
            with open(self.template_file, 'r') as f:
                templates = json.load(f)
                logger.info(f"Loaded {len(templates)} workflow templates from {self.template_file}")
                return templates
        except FileNotFoundError:
            logger.error(f"Workflow template file not found: {self.template_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding workflow template file: {e}")
            return {}

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a workflow template by name.
        """
        template = self.templates.get(name)
        if not template:
            logger.warning(f"Workflow template '{name}' not found.")
        return template

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    wtm = WorkflowTemplateManager()

    print("\nAvailable Templates:")
    for name, template in wtm.templates.items():
        print(f"- {name}: {template['description']}")

    print("\nGetting 'full_reasoning' template:")
    full_reasoning_template = wtm.get_template("full_reasoning")
    print(json.dumps(full_reasoning_template, indent=2))

    print("\nGetting a non-existent template:")
    non_existent = wtm.get_template("non_existent_template")
    print(non_existent)
