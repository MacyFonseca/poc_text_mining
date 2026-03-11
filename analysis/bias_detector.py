"""Bias detection using transformer models and custom heuristics."""
import numpy as np
from typing import Dict, List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')


class BiasDetector:
    """Detect gender bias and discriminatory language in text."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cuda"):
        """Initialize bias detector with transformer model."""
        self.model_name = model_name
        self.device = 0 if device == "cuda" else -1
        
        # Initialize zero-shot classification for bias detection
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        
        # Bias detection keywords
        self.gender_bias_keywords = {
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
        }
        
        self.discriminatory_keywords = {
            'age': ['young', 'old', 'elderly', 'millennial', 'boomer'],
            'race': ['diverse', 'minority', 'immigrant'],
            'disability': ['disabled', 'handicapped', 'special needs'],
            'appearance': ['attractive', 'overweight', 'skinny']
        }

    def detect_gender_bias(self, text: str) -> Dict:
        """Detect gender bias in text."""
        text_lower = text.lower()
        
        male_count = sum(1 for keyword in self.gender_bias_keywords['male_biased'] 
                        if keyword in text_lower)
        female_count = sum(1 for keyword in self.gender_bias_keywords['female_biased'] 
                          if keyword in text_lower)
        
        total = male_count + female_count
        if total == 0:
            return {
                'has_gender_bias': False,
                'male_bias_score': 0.0,
                'female_bias_score': 0.0,
                'bias_direction': 'neutral',
                'confidence': 1.0
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
            'male_keywords_found': male_count,
            'female_keywords_found': female_count,
            'confidence': max(male_ratio, female_ratio)
        }

    def detect_discriminatory_language(self, text: str) -> Dict:
        """Detect discriminatory language by category."""
        text_lower = text.lower()
        results = {}
        
        for category, keywords in self.discriminatory_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
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
            'text': text[:200],  # Store first 200 chars
            'gender_bias': self.detect_gender_bias(text),
            'discriminatory_language': self.detect_discriminatory_language(text)
        }
        
        # Calculate overall bias score
        gender_score = analysis['gender_bias']['confidence']
        discriminatory_scores = [
            v['count'] for v in analysis['discriminatory_language'].values()
        ]
        overall_bias_score = (
            gender_score * 0.6 +
            (min(sum(discriminatory_scores), 5) / 5) * 0.4  # Normalize discriminatory
        )
        
        analysis['overall_bias_score'] = min(overall_bias_score, 1.0)
        analysis['is_biased'] = analysis['overall_bias_score'] > 0.5
        
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
        report.append(f"  - Male Keywords: {gb['male_keywords_found']}")
        report.append(f"  - Female Keywords: {gb['female_keywords_found']}")
        
        # Discriminatory language
        report.append(f"\nDiscriminatory Language:")
        for category, data in analysis['discriminatory_language'].items():
            if data['count'] > 0:
                report.append(f"  - {category.upper()}: {', '.join(data['keywords_found'])}")
        
        return '\n'.join(report)
