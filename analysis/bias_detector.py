"""Bias detection using transformer models and custom heuristics."""
import re
import numpy as np
from typing import Dict, List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')


class BiasDetector:
    """Detect gender bias and discriminatory language in text."""

    # Keyword dictionaries per language
    GENDER_BIAS_KEYWORDS = {
        'english': {
            'male_biased': [
                'he', 'his', 'him', 'man', 'men', 'boy', 'male',
                'leader', 'manager', 'engineer', 'scientist',
                'strong', 'aggressive', 'ambitious', 'logical'
            ],
            'female_biased': [
                'she', 'her', 'hers', 'woman', 'women', 'girl', 'female',
                'nurse', 'secretary', 'assistant', 'support',
                'beautiful', 'emotional', 'nurturing', 'caring'
            ]
        },
        'spanish': {
            'male_biased': [
                'él', 'su', 'hombre', 'hombres', 'chico', 'masculino', 'varón',
                'líder', 'gerente', 'ingeniero', 'científico',
                'fuerte', 'agresivo', 'ambicioso', 'lógico'
            ],
            'female_biased': [
                'ella', 'su', 'mujer', 'mujeres', 'chica', 'femenina', 'femenino',
                'enfermera', 'secretaria', 'asistenta', 'apoyo',
                'hermosa', 'emocional', 'maternal', 'cariñosa'
            ]
        }
    }

    DISCRIMINATORY_KEYWORDS = {
        'english': {
            'age': ['young', 'old', 'elderly', 'millennial', 'boomer'],
            'race': ['diverse', 'minority', 'immigrant'],
            'disability': ['disabled', 'handicapped', 'special needs'],
            'appearance': ['attractive', 'overweight', 'skinny']
        },
        'spanish': {
            'age': ['joven', 'viejo', 'anciano', 'millennial', 'boomer'],
            'race': ['diverso', 'minoría', 'inmigrante'],
            'disability': ['discapacitado', 'minusválido', 'necesidades especiales'],
            'appearance': ['atractivo', 'sobrepeso', 'delgado']
        }
    }

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cuda",
                 language: str = "english"):
        """Initialize bias detector with transformer model."""
        self.model_name = model_name
        self.device = 0 if device == "cuda" else -1
        self.language = language
        
        # Initialize zero-shot classification for bias detection
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        
        # Select keywords for the configured language
        self.gender_bias_keywords = self.GENDER_BIAS_KEYWORDS.get(
            language, self.GENDER_BIAS_KEYWORDS['english']
        )
        self.discriminatory_keywords = self.DISCRIMINATORY_KEYWORDS.get(
            language, self.DISCRIMINATORY_KEYWORDS['english']
        )

    def detect_gender_bias(self, text: str) -> Dict:
        """Detect gender bias in text."""
        text_lower = text.lower()
        
        # Collect actual keywords found using exact word matching
        male_keywords = []
        for keyword in self.gender_bias_keywords['male_biased']:
            # Use word boundaries to match whole words only
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                male_keywords.append(keyword)
        
        female_keywords = []
        for keyword in self.gender_bias_keywords['female_biased']:
            # Use word boundaries to match whole words only
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                female_keywords.append(keyword)
        
        male_count = len(male_keywords)
        female_count = len(female_keywords)
        total = male_count + female_count
        
        if total == 0:
            return {
                'has_gender_bias': False,
                'male_bias_score': 0.0,
                'female_bias_score': 0.0,
                'bias_direction': 'neutral',
                'confidence': 1.0,
                'male_keywords_found': [],
                'female_keywords_found': []
            }
        
        male_ratio = male_count / total
        female_ratio = female_count / total
        
        bias_threshold = 0.6
        has_bias = True if male_ratio > bias_threshold or female_ratio > bias_threshold else False
        
        if male_ratio > female_ratio:
            bias_direction = 'male'
        elif female_ratio > male_ratio:
            bias_direction = 'female'
        else:
            bias_direction = 'balanced'
        
        return {
            'has_gender_bias': has_bias,
            'male_bias_score': male_ratio,
            'female_bias_score': female_ratio,
            'bias_direction': bias_direction,
            'male_keywords_found': male_keywords,
            'female_keywords_found': female_keywords,
            'confidence': max(male_ratio, female_ratio)
        }

    def detect_discriminatory_language(self, text: str) -> Dict:
        """Detect discriminatory language by category."""
        text_lower = text.lower()
        results = {}
        
        for category, keywords in self.discriminatory_keywords.items():
            found_keywords = []
            for kw in keywords:
                # Use word boundaries to match whole words only
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    found_keywords.append(kw)
            has_discriminatory = len(found_keywords) > 0
            results[category] = {
                'has_discriminatory_language': has_discriminatory,
                'keywords_found': found_keywords,
                'count': len(found_keywords)
            }
        
        return results

    def ml_based_bias_detection(self, text: str, categories: List[str]) -> Dict:
        """Use zero-shot classification for bias detection."""
        try:
            result = self.classifier(text[:512], categories)
            
            return {
                'labels': result['labels'],
                'scores': result['scores'],
                'top_category': result['labels'][0],
                'top_score': result['scores'][0]
            }
        except Exception as e:
            print(f"Error in ML-based bias detection: {e}")
            return {'labels': categories, 'scores': [0.0] * len(categories)}

    def comprehensive_bias_analysis(self, text: str) -> Dict:
        """Comprehensive bias analysis combining multiple techniques."""
        analysis = {
            'text': text,
            'gender_bias': self.detect_gender_bias(text),
            'discriminatory_language': self.detect_discriminatory_language(text)
        }
        
        # Check if any discriminatory language is present
        has_discriminatory = any(
            v['has_discriminatory_language'] 
            for v in analysis['discriminatory_language'].values()
        )
        
        # Calculate overall bias score based on actual bias detection
        # Only count gender score if gender bias is actually detected
        if analysis['gender_bias']['has_gender_bias']:
            gender_score = analysis['gender_bias']['confidence']
        else:
            gender_score = 0.0
        
        # Discriminatory language score: 1.0 if present, 0.0 if absent
        discriminatory_score = 1.0 if has_discriminatory else 0.0
        
        overall_bias_score = (
            gender_score * 0.6 +
            discriminatory_score * 0.4
        )
        
        analysis['overall_bias_score'] = min(overall_bias_score, 1.0)
        # A text is biased only if it has gender bias or discriminatory language
        analysis['is_biased'] = analysis['gender_bias']['has_gender_bias'] or has_discriminatory
        
        return analysis

    def batch_analysis(self, texts: List[str]) -> List[Dict]:
        """Perform bias analysis on multiple texts."""
        return [self.comprehensive_bias_analysis(text) for text in texts]

    def generate_bias_report(self, analysis: Dict) -> str:
        """Generate a human-readable bias report."""
        report = []
        report.append("=" * 60)
        report.append("BIAS DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"\nText: {analysis['text']}...")
        report.append(f"\nOverall Bias Score: {analysis['overall_bias_score']:.2%}")
        report.append(f"Is Biased: {analysis['is_biased']}")
        
        # Gender bias
        gb = analysis['gender_bias']
        report.append(f"\nGender Bias Analysis:")
        report.append(f"  - Direction: {gb['bias_direction'].upper()}")
        if gb['male_keywords_found']:
            report.append(f"  - Male Keywords: {', '.join(gb['male_keywords_found'])}")
        if gb['female_keywords_found']:
            report.append(f"  - Female Keywords: {', '.join(gb['female_keywords_found'])}")
        
        # Discriminatory language
        report.append(f"\nDiscriminatory Language:")
        for category, data in analysis['discriminatory_language'].items():
            if data['count'] > 0:
                report.append(f"  - {category.upper()}: {', '.join(data['keywords_found'])}")
        
        return '\n'.join(report)
