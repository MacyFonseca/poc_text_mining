"""Bias detection using fine-tuned transformer models and zero-shot classification."""
import re
from typing import Dict, List
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


class MLBiasDetector:
    """Detect bias using a fine-tuned bias detection model combined with
    zero-shot classification for bias categorization."""

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

    BIAS_CATEGORIES = [
        'gender bias', 'racial bias', 'age bias',
        'disability bias', 'appearance bias', 'neutral'
    ]

    def __init__(self, device: str = "cuda", language: str = "english"):
        """Initialize ML bias detector with fine-tuned and zero-shot models."""
        self.device = 0 if device == "cuda" else -1
        self.language = language

        # Fine-tuned binary bias classifier (Biased / Non-biased)
        # Uses valurank/distilroberta-bias which ships PyTorch weights.
        self.bias_classifier = pipeline(
            "text-classification",
            model="valurank/distilroberta-bias",
            device=self.device
        )

        # Zero-shot classifier for bias categorization
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )

        self.gender_bias_keywords = self.GENDER_BIAS_KEYWORDS.get(
            language, self.GENDER_BIAS_KEYWORDS['english']
        )
        self.discriminatory_keywords = self.DISCRIMINATORY_KEYWORDS.get(
            language, self.DISCRIMINATORY_KEYWORDS['english']
        )

    def detect_bias(self, text: str) -> Dict:
        """Binary bias detection using the fine-tuned valurank model."""
        try:
            result = self.bias_classifier(text[:512])[0]
            # valurank/distilroberta-bias uses LABEL_1=Biased, LABEL_0=Non-biased
            is_biased = result['label'] == 'LABEL_1'
            label = 'Biased' if is_biased else 'Non-biased'
            return {
                'is_biased': is_biased,
                'label': label,
                'confidence': result['score']
            }
        except Exception as e:
            print(f"Error in fine-tuned bias detection: {e}")
            return {'is_biased': False, 'label': 'Non-biased', 'confidence': 0.0}

    def categorize_bias(self, text: str, categories: List[str] = None) -> Dict:
        """Categorize bias type using zero-shot classification."""
        if categories is None:
            categories = self.BIAS_CATEGORIES
        try:
            result = self.zero_shot_classifier(text[:512], categories)
            return {
                'labels': result['labels'],
                'scores': result['scores'],
                'top_category': result['labels'][0],
                'top_score': result['scores'][0]
            }
        except Exception as e:
            print(f"Error in zero-shot bias categorization: {e}")
            return {
                'labels': categories,
                'scores': [0.0] * len(categories),
                'top_category': 'neutral',
                'top_score': 0.0
            }

    def detect_gender_bias(self, text: str) -> Dict:
        """Detect gender bias in text using keyword matching."""
        text_lower = text.lower()

        male_keywords = [
            kw for kw in self.gender_bias_keywords['male_biased']
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        ]
        female_keywords = [
            kw for kw in self.gender_bias_keywords['female_biased']
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        ]

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
        has_bias = male_ratio > bias_threshold or female_ratio > bias_threshold

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
            found_keywords = [
                kw for kw in keywords
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
            ]
            results[category] = {
                'has_discriminatory_language': len(found_keywords) > 0,
                'keywords_found': found_keywords,
                'count': len(found_keywords)
            }

        return results

    def comprehensive_bias_analysis(self, text: str) -> Dict:
        """Comprehensive bias analysis combining fine-tuned model, zero-shot
        categorization, and keyword-based detection."""
        # Fine-tuned binary detection
        ml_detection = self.detect_bias(text)

        # Always run zero-shot categorization to identify bias type
        ml_categorization = self.categorize_bias(text)

        # Keyword-based detection
        gender_bias = self.detect_gender_bias(text)
        discriminatory_language = self.detect_discriminatory_language(text)

        has_discriminatory = any(
            v['has_discriminatory_language']
            for v in discriminatory_language.values()
        )

        # Check if zero-shot thinks the text is biased (top category != neutral)
        zs_is_biased = ml_categorization['top_category'] != 'neutral'
        zs_score = ml_categorization['top_score'] if zs_is_biased else 0.0

        # Overall score: combine all three signal sources
        ml_score = ml_detection['confidence'] if ml_detection['is_biased'] else 0.0
        keyword_gender_score = gender_bias['confidence'] if gender_bias['has_gender_bias'] else 0.0
        keyword_discrim_score = 1.0 if has_discriminatory else 0.0

        overall_bias_score = (
            zs_score * 0.35 +
            ml_score * 0.15 +
            keyword_gender_score * 0.3 +
            keyword_discrim_score * 0.2
        )

        is_biased = (
            ml_detection['is_biased']
            or zs_is_biased
            or gender_bias['has_gender_bias']
            or has_discriminatory
        )

        return {
            'text': text,
            'ml_detection': ml_detection,
            'ml_categorization': ml_categorization,
            'gender_bias': gender_bias,
            'discriminatory_language': discriminatory_language,
            'overall_bias_score': min(overall_bias_score, 1.0),
            'is_biased': is_biased
        }

    def batch_analysis(self, texts: List[str]) -> List[Dict]:
        """Perform bias analysis on multiple texts."""
        return [self.comprehensive_bias_analysis(text) for text in texts]

    def generate_bias_report(self, analysis: Dict) -> str:
        """Generate a human-readable bias report."""
        report = []
        report.append("=" * 60)
        report.append("ML BIAS DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"\nText: {analysis['text']}...")
        report.append(f"\nOverall Bias Score: {analysis['overall_bias_score']:.2%}")
        report.append(f"Is Biased: {analysis['is_biased']}")

        # Fine-tuned model result
        ml = analysis['ml_detection']
        report.append(f"\nFine-Tuned Model (valurank/distilroberta-bias):")
        report.append(f"  - Label: {ml['label']}")
        report.append(f"  - Confidence: {ml['confidence']:.2%}")

        # Zero-shot categorization
        cat = analysis['ml_categorization']
        report.append(f"\nBias Categorization (zero-shot):")
        report.append(f"  - Top Category: {cat['top_category']}")
        report.append(f"  - Top Score: {cat['top_score']:.2%}")
        for label, score in zip(cat['labels'], cat['scores']):
            report.append(f"    {label}: {score:.2%}")

        # Gender bias keywords
        gb = analysis['gender_bias']
        report.append(f"\nGender Bias (keyword-based):")
        report.append(f"  - Direction: {gb['bias_direction'].upper()}")
        if gb['male_keywords_found']:
            report.append(f"  - Male Keywords: {', '.join(gb['male_keywords_found'])}")
        if gb['female_keywords_found']:
            report.append(f"  - Female Keywords: {', '.join(gb['female_keywords_found'])}")

        # Discriminatory language
        report.append(f"\nDiscriminatory Language (keyword-based):")
        for category, data in analysis['discriminatory_language'].items():
            if data['count'] > 0:
                report.append(f"  - {category.upper()}: {', '.join(data['keywords_found'])}")

        return '\n'.join(report)
