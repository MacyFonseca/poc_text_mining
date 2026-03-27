"""Lightweight bias detection using keyword and pattern heuristics only.

Uses the shared keyword/pattern dictionaries from ``bias_keywords``.
No transformer models are loaded — this class is fast, requires no GPU,
and has no heavy dependencies beyond the standard library.

For ML-powered detection (zero-shot + fine-tuned models), use
``MLBiasDetector`` from ``analysis.ml_bias_detector`` instead.
"""
import logging
from typing import Dict, List

from analysis.bias_keywords import (
    detect_discriminatory_language,
    detect_gender_bias as kw_detect_gender_bias,
    score_to_severity,
)

logger = logging.getLogger(__name__)


class BiasDetector:
    """Detect gender bias and discriminatory language in text using
    keyword matching and explicit bias patterns.

    This is the lightweight detector — no models are downloaded or loaded.
    """

    def __init__(self, model_name: str = "bert-base-uncased",
                 device: str = "cuda", language: str = "english"):
        """Initialise bias detector.

        Parameters
        ----------
        model_name : str
            Kept for backward-compatibility; not used.
        device : str
            Kept for backward-compatibility; not used.
        language : str
            ``"english"`` or ``"spanish"``.
        """
        self.language = language

    # -- keyword-based helpers (delegate to shared module) ------------------

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

    def comprehensive_bias_analysis(self, text: str,
                                    gender_bias_threshold: float = 0.6,
                                    overall_bias_threshold: float = 0.25,
                                    weights: tuple = (0.6, 0.4)) -> Dict:
        """Comprehensive bias analysis combining keyword + pattern detection.

        Parameters
        ----------
        gender_bias_threshold : float
            Male/female ratio above which keyword imbalance flags gender bias.
        overall_bias_threshold : float
            Minimum ``overall_bias_score`` to mark the text as biased.
        weights : tuple
            ``(gender_weight, discriminatory_weight)`` for score combination.
        """
        gender_bias = self.detect_gender_bias(
            text, bias_threshold=gender_bias_threshold,
        )
        discriminatory = self.detect_discriminatory_language(text)

        has_discriminatory = any(
            v['has_discriminatory_language'] for v in discriminatory.values()
        )

        gender_score = gender_bias['confidence'] if gender_bias['has_gender_bias'] else 0.0
        discriminatory_score = 1.0 if has_discriminatory else 0.0

        w_gender, w_discrim = weights
        overall_bias_score = min(
            gender_score * w_gender + discriminatory_score * w_discrim,
            1.0,
        )

        is_biased = overall_bias_score >= overall_bias_threshold
        severity = score_to_severity(overall_bias_score)

        return {
            'text': text,
            'gender_bias': gender_bias,
            'discriminatory_language': discriminatory,
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
            "BIAS DETECTION REPORT",
            "=" * 60,
            f"\nText: {text_preview}",
            f"\nOverall Bias Score: {analysis['overall_bias_score']:.2%}",
            f"Severity: {severity.upper()}",
            f"Is Biased: {analysis['is_biased']}",
        ]

        gb = analysis['gender_bias']
        report.append(f"\nGender Bias Analysis:")
        report.append(f"  - Direction: {gb['bias_direction'].upper()}")
        if gb['male_keywords_found']:
            report.append(f"  - Male Keywords: {', '.join(gb['male_keywords_found'])}")
        if gb['female_keywords_found']:
            report.append(f"  - Female Keywords: {', '.join(gb['female_keywords_found'])}")
        if gb.get('bias_patterns_matched'):
            report.append(f"  - Bias Patterns: {', '.join(gb['bias_patterns_matched'])}")
        if gb.get('positive_context_found'):
            report.append(f"  - Positive Context: {', '.join(gb['positive_context_found'])}")

        report.append(f"\nDiscriminatory Language:")
        for category, data in analysis['discriminatory_language'].items():
            if data['count'] > 0:
                report.append(f"  - {category.upper()}: {', '.join(data['keywords_found'])}")

        return '\n'.join(report)
