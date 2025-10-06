"""
Synthetic Dataset Generator for GitHub UI Components
Creates augmented datasets from search result embeddings
"""

import json
import numpy as np
import random
from typing import Dict, List, Any
from datetime import datetime

class SyntheticDatasetGenerator:
    def __init__(self, base_search_results: List[Dict]):
        self.base_data = base_search_results
        self.generated_count = 0
        
    def augment_ui_template(self, template_text: str) -> Dict[str, str]:
        """Generate variations of UI templates"""
        base_variations = {
            'original': template_text,
            'dark_mode': template_text.replace('#ffffff', '#1a1a1a').replace('#f6f8fa', '#0d1117'),
            'light_mode': template_text.replace('#1a1a1a', '#ffffff').replace('#0d1117', '#f6f8fa'),
            'mobile_size': template_text.replace('width: 100%', 'width: 320px').replace('height: auto', 'height: 568px'),
            'desktop_size': template_text.replace('width: 320px', 'width: 100%').replace('height: 568px', 'height: auto'),
            'minimalist': self._remove_decorative_elements(template_text),
            'enhanced': self._add_animations_and_effects(template_text)
        }
        
        return base_variations
    
    def _remove_decorative_elements(self, text: str) -> str:
        """Create minimalist version by removing decorative elements"""
        # Remove complex gradients, shadows, animations
        simplified = text
        simplified = simplified.replace('gradient', 'solid')
        simplified = simplified.replace('shadow', '')
        simplified = simplified.replace('animation', '')
        simplified = simplified.replace('transition', '')
        return simplified
    
    def _add_animations_and_effects(self, text: str) -> str:
        """Add enhanced visual effects"""
        enhanced = text
        if 'hover' not in enhanced:
            enhanced = enhanced.replace('}', '  transition: all 0.3s ease;\\n  hover: { transform: scale(1.05); }\\n}')
        return enhanced
    
    def generate_embedding_variations(self, base_embedding: List[float], count: int = 5) -> List[List[float]]:
        """Generate synthetic embeddings with controlled noise"""
        variations = []
        base_array = np.array(base_embedding)
        
        for i in range(count):
            # Add Gaussian noise with different intensities
            noise_level = random.uniform(0.01, 0.1)
            noisy_embedding = base_array + np.random.normal(0, noise_level, 1024)
            variations.append(noisy_embedding.tolist())
        
        return variations
    
    def create_synthetic_entry(self, base_entry: Dict, variation_type: str, 
                             template_variant: str, embedding_variant: List[float]) -> Dict:
        """Create a complete synthetic dataset entry"""
        self.generated_count += 1
        
        synthetic_id = f"synth_{self.generated_count:06d}"
        timestamp = datetime.utcnow().isoformat()
        
        return {
            "_index": "synthetic-github-ui",
            "_id": synthetic_id,
            "_score": random.uniform(0.1, 0.9),  # Synthetic relevance score
            "_source": {
                "text": template_variant,
                "embeddings": embedding_variant,
                "metadata": {
                    "synthetic": True,
                    "synthetic_id": synthetic_id,
                    "base_source": base_entry["_source"]["metadata"]["source"],
                    "original_file_path": base_entry["_source"]["metadata"]["file_path"],
                    "variation_type": variation_type,
                    "generation_timestamp": timestamp,
                    "chunk_category": "synthetic_template",
                    "embedding_model": "gte-large-en-v1.5",
                    "augmentation_method": "embedding_noise+template_variation"
                }
            }
        }
    
    def generate_dataset(self, target_size: int = 10000) -> List[Dict]:
        """Generate synthetic dataset of specified size"""
        synthetic_dataset = []
        
        while len(synthetic_dataset) < target_size:
            # Select random base entry
            base_entry = random.choice(self.base_data)
            
            # Generate template variations
            template_variations = self.augment_ui_template(base_entry["_source"]["text"])
            
            # Generate embedding variations
            embedding_variations = self.generate_embedding_variations(
                base_entry["_source"]["embeddings"], count=3
            )
            
            # Create synthetic combinations
            for template_key, template_variant in template_variations.items():
                for embedding_variant in embedding_variations:
                    synthetic_entry = self.create_synthetic_entry(
                        base_entry, 
                        f"template_{template_key}", 
                        template_variant, 
                        embedding_variant
                    )
                    synthetic_dataset.append(synthetic_entry)
                    
                    if len(synthetic_dataset) >= target_size:
                        break
                if len(synthetic_dataset) >= target_size:
                    break
        
        return synthetic_dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save synthetic dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "total_entries": len(dataset),
                    "base_source_count": len(self.base_data),
                    "synthetic_ratio": len(dataset) / len(self.base_data)
                },
                "dataset": dataset
            }, f, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    # Load your search results
    with open('search_results.json', 'r') as f:
        search_results = json.load(f)
    
    # Initialize generator
    generator = SyntheticDatasetGenerator(search_results)
    
    # Generate synthetic dataset
    synthetic_data = generator.generate_dataset(target_size=5000)
    
    # Save to file
    generator.save_dataset(synthetic_data, 'synthetic_github_ui_dataset.json')
    
    print(f"Generated {len(synthetic_data)} synthetic entries")
    print(f"Based on {len(search_results)} original entries")
    print(f"Synthetic ratio: {len(synthetic_data) / len(search_results):.2f}x")