"""
Template Variation Generator for GitHub UI Components
Specialized generator for creating realistic UI template variations
"""

import re
import random
from typing import Dict, List, Set
from synthetic_dataset_generator import SyntheticDatasetGenerator

class TemplateVariationGenerator(SyntheticDatasetGenerator):
    def __init__(self, base_search_results: List[Dict]):
        super().__init__(base_search_results)
        self.color_palettes = {
            'github_original': ['#ffffff', '#f6f8fa', '#1a1a1a', '#0366d6', '#28a745'],
            'github_dark': ['#0d1117', '#161b22', '#f0f6fc', '#58a6ff', '#3fb950'],
            'monochromatic': ['#2d3748', '#4a5568', '#718096', '#a0aec0', '#e2e8f0'],
            'warm_colors': ['#fed7d7', '#feb2b2', '#fc8181', '#e53e3e', '#c53030'],
            'cool_colors': ['#bee3f8', '#90cdf4', '#63b3ed', '#4299e1', '#3182ce']
        }
        
        self.component_types = {
            'button': ['primary', 'secondary', 'danger', 'success', 'outline'],
            'icon': ['small', 'medium', 'large', 'x-large', 'custom'],
            'modal': ['small', 'medium', 'large', 'fullscreen', 'floating'],
            'card': ['default', 'elevated', 'bordered', 'transparent', 'interactive']
        }
    
    def extract_ui_patterns(self, template_text: str) -> Dict[str, List[str]]:
        """Extract UI patterns from template text"""
        patterns = {
            'colors': re.findall(r'#[0-9a-fA-F]{6}', template_text),
            'sizes': re.findall(r'(width|height):\\s*([\\d.]+)(px|%|rem|em)', template_text),
            'classes': re.findall(r'class=\"([^\"]+)\"', template_text),
            'ids': re.findall(r'id=\"([^\"]+)\"', template_text),
            'svg_elements': re.findall(r'<svg[^>]*>.*?</svg>', template_text, re.DOTALL)
        }
        return patterns
    
    def generate_color_variations(self, template_text: str) -> Dict[str, str]:
        """Generate color theme variations"""
        original_colors = set(re.findall(r'#[0-9a-fA-F]{6}', template_text))
        variations = {'original': template_text}
        
        for palette_name, palette_colors in self.color_palettes.items():
            if palette_name == 'github_original':
                continue
                
            variant_text = template_text
            for i, original_color in enumerate(list(original_colors)[:5]):
                if i < len(palette_colors):
                    variant_text = variant_text.replace(original_color, palette_colors[i])
            
            variations[palette_name] = variant_text
        
        return variations
    
    def generate_size_variations(self, template_text: str) -> Dict[str, str]:
        """Generate size and layout variations"""
        variations = {'original': template_text}
        
        # Mobile-first variations
        mobile_text = re.sub(r'width:\\s*([\\d.]+)(px|%|rem)', 
                           lambda m: f'width: {min(320, float(m.group(1)))}px' if m.group(2) == 'px' else m.group(0), 
                           template_text)
        variations['mobile'] = mobile_text
        
        # Desktop variations
        desktop_text = re.sub(r'width:\\s*([\\d.]+)px', 
                            lambda m: f'width: {max(1024, float(m.group(1)))}px', 
                            template_text)
        variations['desktop'] = desktop_text
        
        # Responsive variations
        responsive_text = re.sub(r'width:\\s*([\\d.]+)px', 
                               'width: 100%', template_text)
        variations['responsive'] = responsive_text
        
        return variations
    
    def generate_component_variations(self, template_text: str) -> Dict[str, str]:
        """Generate component type variations"""
        variations = {'original': template_text}
        
        # Button variations
        if 'button' in template_text.lower():
            for button_type in self.component_types['button']:
                variant = template_text.replace('button', f'button {button_type}')
                variations[f'button_{button_type}'] = variant
        
        # Icon size variations
        svg_matches = re.findall(r'<svg[^>]*>', template_text)
        if svg_matches:
            for size in self.component_types['icon']:
                size_map = {'small': '16', 'medium': '24', 'large': '32', 'x-large': '48'}
                if size in size_map:
                    variant = re.sub(r'width=\"[^\"]*\"', f'width=\"{size_map[size]}\"', template_text)
                    variant = re.sub(r'height=\"[^\"]*\"', f'height=\"{size_map[size]}\"', variant)
                    variations[f'icon_{size}'] = variant
        
        return variations
    
    def generate_accessibility_variations(self, template_text: str) -> Dict[str, str]:
        """Generate accessibility-enhanced variations"""
        variations = {'original': template_text}
        
        # Add ARIA labels
        if '<svg' in template_text and 'aria-label' not in template_text:
            aria_variant = template_text.replace('<svg', '<svg aria-label=\"descriptive icon\"')
            variations['aria_enhanced'] = aria_variant
        
        # Add focus states
        if ':hover' in template_text and ':focus' not in template_text:
            focus_variant = template_text.replace(':hover', ':hover, :focus')
            variations['focus_enhanced'] = focus_variant
        
        # Add high contrast
        high_contrast = template_text.replace('#f6f8fa', '#000000').replace('#ffffff', '#000000')
        high_contrast = high_contrast.replace('#1a1a1a', '#ffffff')
        variations['high_contrast'] = high_contrast
        
        return variations
    
    def advanced_augment_ui_template(self, template_text: str) -> Dict[str, str]:
        """Advanced template augmentation with multiple variation types"""
        variations = {}
        
        # Combine all variation types
        color_variations = self.generate_color_variations(template_text)
        size_variations = self.generate_size_variations(template_text)
        component_variations = self.generate_component_variations(template_text)
        accessibility_variations = self.generate_accessibility_variations(template_text)
        
        # Merge all variations
        variations.update(color_variations)
        variations.update(size_variations)
        variations.update(component_variations)
        variations.update(accessibility_variations)
        
        # Generate combined variations
        for color_key, color_val in color_variations.items():
            for size_key, size_val in size_variations.items():
                if color_key != 'original' or size_key != 'original':
                    # Apply size variations to color variants
                    combined = self.generate_size_variations(color_val).get(size_key, color_val)
                    variations[f'{color_key}_{size_key}'] = combined
        
        return variations
    
    def generate_specialized_dataset(self, target_size: int = 10000) -> List[Dict]:
        """Generate specialized synthetic dataset with template variations"""
        synthetic_dataset = []
        
        while len(synthetic_dataset) < target_size:
            base_entry = random.choice(self.base_data)
            template_text = base_entry['_source']['text']
            
            # Generate advanced template variations
            template_variations = self.advanced_augment_ui_template(template_text)
            
            # Generate embedding variations
            embedding_variations = self.generate_embedding_variations(
                base_entry['_source']['embeddings'], count=2
            )
            
            # Create synthetic entries
            for template_key, template_variant in template_variations.items():
                for embedding_variant in embedding_variations:
                    synthetic_entry = self.create_synthetic_entry(
                        base_entry,
                        f'advanced_{template_key}',
                        template_variant,
                        embedding_variant
                    )
                    synthetic_dataset.append(synthetic_entry)
                    
                    if len(synthetic_dataset) >= target_size:
                        break
                if len(synthetic_dataset) >= target_size:
                    break
        
        return synthetic_dataset

# Example usage
if __name__ == "__main__":
    # Load search results
    with open('search_results.json', 'r') as f:
        search_data = json.load(f)
    
    # Initialize specialized generator
    specialized_generator = TemplateVariationGenerator(search_data)
    
    # Generate specialized dataset
    specialized_data = specialized_generator.generate_specialized_dataset(8000)
    
    # Save results
    specialized_generator.save_dataset(specialized_data, 'specialized_ui_template_dataset.json')
    
    print(f"Generated {len(specialized_data)} specialized template variations")
    print("Variation types include:")
    print("- Color theme variations (dark, monochromatic, warm, cool)")
    print("- Size variations (mobile, desktop, responsive)")
    print("- Component type variations (button types, icon sizes)")
    print("- Accessibility enhancements (ARIA, focus states, high contrast)")
    print("- Combined variations (color + size, etc.)")