"""Bias detection using fine-tuned transformer models and zero-shot classification.

Combines three signal sources:
1. Fine-tuned binary bias classifier (``valurank/distilroberta-bias`` – English only).
2. Zero-shot categorisation (BART for English, XLM-R for Spanish).
3. Keyword / pattern analysis (shared ``bias_keywords`` module).

The ``is_biased`` decision is now threshold-based rather than OR-gated to
reduce false positives on texts that merely *discuss* gender.
"""
import logging
import re
from typing import Dict, List

from transformers import pipeline
import warnings

from analysis.bias_keywords import (
    detect_discriminatory_language,
    detect_gender_bias as kw_detect_gender_bias,
    score_to_severity,
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Model selection per language
_ZS_MODELS = {
    'english': 'facebook/bart-large-mnli',
    'spanish': 'joeddav/xlm-roberta-large-xnli',
}

# The fine-tuned bias model only supports English.
_FINETUNED_MODEL = 'valurank/distilroberta-bias'


def _split_text(text: str, max_len: int = 512) -> List[str]:
    """Split *text* into chunks of at most *max_len* characters on sentence
    boundaries when possible."""
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_len
        if end >= len(text):
            chunks.append(text[start:])
            break
        boundary = text.rfind('. ', start, end)
        if boundary > start:
            end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


class MLBiasDetector:
    """Detect bias using a fine-tuned bias detection model combined with
    zero-shot classification for bias categorisation."""

    BIAS_CATEGORIES = [
        'gender bias', 'racial bias', 'age bias',
        'disability bias', 'appearance bias', 'neutral',
    ]

    def __init__(self, device: str = "cuda", language: str = "english"):
        """Initialise ML bias detector.

        Parameters
        ----------
        device : str
            ``"cuda"`` or ``"cpu"``.
        language : str
            ``"english"`` or ``"spanish"``.  When *spanish* is selected the
            fine-tuned model is skipped (English-only) and only the
            multilingual zero-shot model + keywords are used.
        """
        self.device = 0 if device == "cuda" else -1
        self.language = language

        # Fine-tuned binary bias classifier – English only
        if language == 'english':
            self.bias_classifier = pipeline(
                "text-classification",
                model=_FINETUNED_MODEL,
                device=self.device,
            )
        else:
            self.bias_classifier = None
            logger.warning(
                "Fine-tuned bias model (%s) only supports English. "
                "Skipping for language='%s'.", _FINETUNED_MODEL, language,
            )

        # Zero-shot classifier – language-aware
        zs_model = _ZS_MODELS.get(language, _ZS_MODELS['english'])
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=zs_model,
            device=self.device,
        )

    # -- fine-tuned model ---------------------------------------------------

    def detect_bias(self, text: str) -> Dict:
        """Binary bias detection using the fine-tuned valurank model.

        Returns a neutral result when the model is unavailable (non-English).
        Long texts are chunked and scores averaged.
        """
        if self.bias_classifier is None:
            return {
                'is_biased': False,
                'label': 'Unavailable',
                'confidence': 0.0,
            }
        try:
            chunks = _split_text(text, max_len=512)
            biased_scores: List[float] = []
            for chunk in chunks:
                result = self.bias_classifier(chunk)[0]
                is_biased = result['label'] == 'LABEL_1'
                score = result['score'] if is_biased else (1.0 - result['score'])
                biased_scores.append(score)

            avg_score = sum(biased_scores) / len(biased_scores)
            is_biased = avg_score > 0.5
            label = 'Biased' if is_biased else 'Non-biased'
            return {
                'is_biased': is_biased,
                'label': label,
                'confidence': round(avg_score, 4),
            }
        except Exception as e:
            logger.error("Error in fine-tuned bias detection: %s", e)
            return {'is_biased': False, 'label': 'Non-biased', 'confidence': 0.0}

    # -- zero-shot categorisation -------------------------------------------

    def categorize_bias(self, text: str,
                        categories: List[str] = None) -> Dict:
        """Categorise bias type using zero-shot classification.

        Long texts are chunked; per-label scores are averaged.
        """
        if categories is None:
            categories = self.BIAS_CATEGORIES
        try:
            chunks = _split_text(text, max_len=512)
            all_scores: Dict[str, List[float]] = {c: [] for c in categories}

            for chunk in chunks:
                result = self.zero_shot_classifier(chunk, categories)
                for label, score in zip(result['labels'], result['scores']):
                    all_scores[label].append(score)

            avg_scores = {
                label: sum(s) / len(s) for label, s in all_scores.items()
            }
            sorted_labels = sorted(avg_scores, key=avg_scores.get, reverse=True)
            sorted_scores = [round(avg_scores[l], 4) for l in sorted_labels]

            return {
                'labels': sorted_labels,
                'scores': sorted_scores,
                'top_category': sorted_labels[0],
                'top_score': sorted_scores[0],
            }
        except Exception as e:
            logger.error("Error in zero-shot bias categorisation: %s", e)
            return {
                'labels': categories,
                'scores': [0.0] * len(categories),
                'top_category': 'neutral',
                'top_score': 0.0,
            }

    # -- keyword / pattern helpers (delegate to shared module) --------------

    def detect_gender_bias(self, text: str,
                           bias_threshold: float = 0.6) -> Dict:
        """Detect gender bias via keywords and explicit patterns."""
        return kw_detect_gender_bias(
            text, language=self.language,
            bias_threshold=bias_threshold,
        )

    def detect_discriminatory_language(self, text: str) -> Dict:
        """Detect discriminatory language by category."""
        return detect_discriminatory_language(text, language=self.language)

    # -- comprehensive analysis ---------------------------------------------

    def comprehensive_bias_analysis(
        self, text: str,
        overall_bias_threshold: float = 0.25,
        weights: tuple = (0.30, 0.30, 0.20, 0.20),
    ) -> Dict:
        """Comprehensive bias analysis combining all signal sources.

        Parameters
        ----------
        overall_bias_threshold : float
            Minimum ``overall_bias_score`` to mark the text as biased.
        weights : tuple
            ``(fine_tuned, zero_shot, gender_keyword, discriminatory_keyword)``.
        """
        ml_detection = self.detect_bias(text)
        ml_categorization = self.categorize_bias(text)
        gender_bias = self.detect_gender_bias(text)
        discriminatory_language = self.detect_discriminatory_language(text)

        has_discriminatory = any(
            v['has_discriminatory_language']
            for v in discriminatory_language.values()
        )

        # --- score each signal source (0.0 when not triggered) ---
        ml_score = ml_detection['confidence'] if ml_detection['is_biased'] else 0.0

        zs_is_biased = ml_categorization['top_category'] != 'neutral'
        zs_score = ml_categorization['top_score'] if zs_is_biased else 0.0

        kw_gender_score = gender_bias['confidence'] if gender_bias['has_gender_bias'] else 0.0
        kw_discrim_score = 1.0 if has_discriminatory else 0.0

        w_ft, w_zs, w_kg, w_kd = weights
        overall_bias_score = min(
            ml_score * w_ft
            + zs_score * w_zs
            + kw_gender_score * w_kg
            + kw_discrim_score * w_kd,
            1.0,
        )

        # Threshold-based decision instead of OR-logic
        is_biased = overall_bias_score >= overall_bias_threshold
        severity = score_to_severity(overall_bias_score)

        return {
            'text': text,
            'ml_detection': ml_detection,
            'ml_categorization': ml_categorization,
            'gender_bias': gender_bias,
            'discriminatory_language': discriminatory_language,
            'overall_bias_score': round(overall_bias_score, 4),
            'is_biased': is_biased,
            'severity': severity,
        }

    def batch_analysis(self, texts: List[str]) -> List[Dict]:
        """Perform bias analysis on multiple texts."""
        return [self.comprehensive_bias_analysis(text) for text in texts]

    def generate_bias_report(self, analysis: Dict) -> str:
        """Generate a human-readable bias report."""
        text_preview = analysis['text'][:200] + ('...' if len(analysis['text']) > 200 else '')
        severity = analysis.get('severity', 'unknown')

        report = [
            "=" * 60,
            "ML BIAS DETECTION REPORT",
            "=" * 60,
            f"\nText: {text_preview}",
            f"\nOverall Bias Score: {analysis['overall_bias_score']:.2%}",
            f"Severity: {severity.upper()}",
            f"Is Biased: {analysis['is_biased']}",
        ]

        # Fine-tuned model result
        ml = analysis['ml_detection']
        report.append(f"\nFine-Tuned Model ({_FINETUNED_MODEL}):")
        report.append(f"  - Label: {ml['label']}")
        report.append(f"  - Confidence: {ml['confidence']:.2%}")

        # Zero-shot categorisation
        cat = analysis['ml_categorization']
        report.append(f"\nBias Categorisation (zero-shot):")
        report.append(f"  - Top Category: {cat['top_category']}")
        report.append(f"  - Top Score: {cat['top_score']:.2%}")
        for label, score in zip(cat['labels'], cat['scores']):
            report.append(f"    {label}: {score:.2%}")

        # Gender bias keywords
        gb = analysis['gender_bias']
        report.append(f"\nGender Bias (keyword + pattern):")
        report.append(f"  - Direction: {gb['bias_direction'].upper()}")
        if gb['male_keywords_found']:
            report.append(f"  - Male Keywords: {', '.join(gb['male_keywords_found'])}")
        if gb['female_keywords_found']:
            report.append(f"  - Female Keywords: {', '.join(gb['female_keywords_found'])}")
        if gb.get('bias_patterns_matched'):
            report.append(f"  - Bias Patterns: {', '.join(gb['bias_patterns_matched'])}")
        if gb.get('positive_context_found'):
            report.append(f"  - Positive Context: {', '.join(gb['positive_context_found'])}")

        # Discriminatory language
        report.append(f"\nDiscriminatory Language (keyword-based):")
        for category, data in analysis['discriminatory_language'].items():
            if data['count'] > 0:
                report.append(f"  - {category.upper()}: {', '.join(data['keywords_found'])}")

        return '\n'.join(report)
