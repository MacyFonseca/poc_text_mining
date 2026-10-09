"""Bias detection using fine-tuned transformer models and zero-shot (NLI) classification.

Combines three signal sources:
1. Fine-tuned binary bias classifier (``himel7/bias-detector`` - English only,
   trained on news media bias; scored per sentence).
2. Zero-shot categorisation with an NLI model (BART for English, XLM-R for
   Spanish).  Each category is a hypothesis in the document's language
   ("Este texto expresa un estereotipo de género.") scored independently with
   the full 3-way entailment probability, over windows of a few sentences.
3. Keyword / pattern analysis (shared ``bias_keywords`` module).

The ``is_biased`` decision is threshold-based on a weighted sum.  When a
signal is unavailable (the fine-tuned model for Spanish) its weight is
redistributed over the remaining signals instead of counting as zero.
"""
import json
import logging
import os
from typing import Dict, List, Optional

import torch
import warnings
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from analysis.bias_keywords import (
    collect_findings,
    detect_discriminatory_language,
    detect_gender_bias as kw_detect_gender_bias,
    discriminatory_score as kw_discriminatory_score,
    score_to_severity,
)
from utils.nlp import get_nlp

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# NLI model per language.  Measured on data/gold: XLM-R-large gives usable
# entailment scores for Spanish; mDeBERTa-v3-base-xnli gives near-zero
# entailment for every sentence with these hypotheses (see ROADMAP P2.1).
_ZS_MODELS = {
    'english': 'facebook/bart-large-mnli',
    'spanish': 'joeddav/xlm-roberta-large-xnli',
}

# The fine-tuned bias model only supports English.
_FINETUNED_MODEL = 'himel7/bias-detector'

# One hypothesis per category, in the document's language.
_HYPOTHESES = {
    'english': {
        'gender bias': 'This text expresses a gender stereotype.',
        'racial bias': 'This text expresses prejudice against immigrants or people of another ethnic origin.',
        'age bias': 'This text expresses prejudice against people because of their age.',
        'disability bias': 'This text expresses prejudice against people with disabilities.',
        'appearance bias': 'This text judges people by their physical appearance.',
    },
    'spanish': {
        'gender bias': 'Este texto expresa un estereotipo de género.',
        'racial bias': 'Este texto expresa prejuicios contra personas inmigrantes o de otro origen étnico.',
        'age bias': 'Este texto expresa prejuicios contra las personas por su edad.',
        'disability bias': 'Este texto expresa prejuicios contra las personas con discapacidad.',
        'appearance bias': 'Este texto juzga a las personas por su aspecto físico.',
    },
}
ZS_CATEGORY_THRESHOLD = 0.5      # entailment probability needed to name a category
ZS_WINDOW_TOKENS = 200           # sentences are grouped into windows up to this size

DEFAULT_WEIGHTS = (0.30, 0.30, 0.20, 0.20)
WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'config', 'ml_weights.json')


def load_learned_weights(language: str) -> Optional[Dict]:
    """Weights fitted by ``eval/fit_weights.py`` for *language*, if present."""
    try:
        with open(WEIGHTS_FILE, encoding='utf-8') as f:
            return json.load(f).get(language)
    except (OSError, ValueError):
        return None


class MLBiasDetector:
    """Detect bias using a fine-tuned bias detection model combined with
    zero-shot classification for bias categorisation."""

    BIAS_CATEGORIES = list(_HYPOTHESES['english']) + ['neutral']

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
        self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1
        self.language = language if language in _HYPOTHESES else 'english'
        self.nlp = get_nlp(self.language)

        # Fine-tuned binary bias classifier - English only
        if self.language == 'english':
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

        # NLI model for zero-shot categorisation - language-aware
        zs_model = _ZS_MODELS[self.language]
        self.nli_tokenizer = AutoTokenizer.from_pretrained(zs_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(zs_model).eval()
        if self.device == 0:
            self.nli_model.to('cuda')
        self._entail_id = next(
            i for i, label in self.nli_model.config.id2label.items()
            if label.lower().startswith('entail')
        )
        self.learned = load_learned_weights(self.language)

    # -- text units -------------------------------------------------------

    def _sentences(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        return [{'text': s.text.strip(), 'start': s.start_char, 'end': s.end_char}
                for s in doc.sents if s.text.strip()]

    def _windows(self, sentences: List[Dict]) -> List[Dict]:
        """Group consecutive sentences into windows of <= ZS_WINDOW_TOKENS tokens."""
        windows, current, size = [], [], 0
        for s in sentences:
            n = len(self.nli_tokenizer.tokenize(s['text']))
            if current and size + n > ZS_WINDOW_TOKENS:
                windows.append(current)
                current, size = [], 0
            current.append(s)
            size += n
        if current:
            windows.append(current)
        return [{'text': ' '.join(s['text'] for s in w), 'start': w[0]['start'], 'end': w[-1]['end']}
                for w in windows]

    # -- fine-tuned model ---------------------------------------------------

    def detect_bias(self, text: str, sentences: Optional[List[Dict]] = None) -> Dict:
        """Binary bias detection with the fine-tuned model, per sentence.

        The page confidence is the highest sentence score, so one biased
        sentence is not averaged away.  Returns a neutral result when the
        model is unavailable (non-English).
        """
        if self.bias_classifier is None:
            return {
                'is_biased': False,
                'label': 'Unavailable',
                'confidence': 0.0,
                'available': False,
                'sentence_scores': [],
            }
        try:
            sentences = sentences or self._sentences(text)
            outputs = self.bias_classifier([s['text'] for s in sentences],
                                           batch_size=16, truncation=True)
            scored = []
            for s, out in zip(sentences, outputs):
                p = out['score'] if out['label'] == 'LABEL_1' else 1.0 - out['score']
                scored.append({**s, 'score': round(p, 4)})
            top = max((s['score'] for s in scored), default=0.0)
            is_biased = top > 0.5
            return {
                'is_biased': is_biased,
                'label': 'Biased' if is_biased else 'Non-biased',
                'confidence': round(top, 4),
                'available': True,
                'sentence_scores': [s for s in scored if s['score'] > 0.5],
            }
        except Exception as e:
            logger.error("Error in fine-tuned bias detection: %s", e)
            return {'is_biased': False, 'label': 'Non-biased', 'confidence': 0.0,
                    'available': True, 'sentence_scores': []}

    # -- zero-shot categorisation -------------------------------------------

    @torch.no_grad()
    def _entailment(self, premises: List[str], hypotheses: List[str]) -> List[List[float]]:
        """P(entailment) (3-way softmax) for every premise x hypothesis pair."""
        pairs = [(p, h) for p in premises for h in hypotheses]
        probs: List[float] = []
        for i in range(0, len(pairs), 16):
            batch = pairs[i:i + 16]
            enc = self.nli_tokenizer([p for p, _ in batch], [h for _, h in batch],
                                     return_tensors='pt', padding=True,
                                     truncation='only_first', max_length=512)
            if self.device == 0:
                enc = enc.to('cuda')
            logits = self.nli_model(**enc).logits
            probs.extend(logits.softmax(-1)[:, self._entail_id].tolist())
        n = len(hypotheses)
        return [probs[i * n:(i + 1) * n] for i in range(len(premises))]

    def categorize_bias(self, text: str, categories: List[str] = None,
                        sentences: Optional[List[Dict]] = None) -> Dict:
        """Categorise bias type with NLI, scoring each category independently.

        Each category's page score is its highest entailment probability over
        all windows.  ``top_category`` is ``'neutral'`` when no category
        reaches ``ZS_CATEGORY_THRESHOLD``.
        """
        hyps = _HYPOTHESES[self.language]
        if categories is None:
            categories = list(hyps)
        categories = [c for c in categories if c in hyps]
        try:
            windows = self._windows(sentences or self._sentences(text))
            scores = self._entailment([w['text'] for w in windows], [hyps[c] for c in categories])
            best = {c: max(row[j] for row in scores) for j, c in enumerate(categories)}
            sorted_labels = sorted(best, key=best.get, reverse=True)
            top = sorted_labels[0]
            flagged = [
                {**w, 'category': c, 'score': round(row[j], 4)}
                for w, row in zip(windows, scores) for j, c in enumerate(categories)
                if row[j] >= ZS_CATEGORY_THRESHOLD
            ]
            return {
                'labels': sorted_labels,
                'scores': [round(best[c], 4) for c in sorted_labels],
                'top_category': top if best[top] >= ZS_CATEGORY_THRESHOLD else 'neutral',
                'top_score': round(best[top], 4),
                'flagged_windows': flagged,
            }
        except Exception as e:
            logger.error("Error in zero-shot bias categorisation: %s", e)
            return {
                'labels': categories,
                'scores': [0.0] * len(categories),
                'top_category': 'neutral',
                'top_score': 0.0,
                'flagged_windows': [],
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

    def signal_scores(self, text: str) -> Dict:
        """Run every signal source once and return the raw parts and scores."""
        sentences = self._sentences(text)
        ml_detection = self.detect_bias(text, sentences)
        ml_categorization = self.categorize_bias(text, sentences=sentences)
        gender_bias = self.detect_gender_bias(text)
        discriminatory_language = self.detect_discriminatory_language(text)

        zs_is_biased = ml_categorization['top_category'] != 'neutral'
        return {
            'ml_detection': ml_detection,
            'ml_categorization': ml_categorization,
            'gender_bias': gender_bias,
            'discriminatory_language': discriminatory_language,
            'scores': {
                'fine_tuned': ml_detection['confidence'] if ml_detection['is_biased'] else 0.0,
                'zero_shot': ml_categorization['top_score'] if zs_is_biased else 0.0,
                'gender_keyword': gender_bias['confidence'],
                'discriminatory_keyword': kw_discriminatory_score(discriminatory_language),
            },
        }

    def comprehensive_bias_analysis(
        self, text: str,
        overall_bias_threshold: float = 0.25,
        weights: tuple = DEFAULT_WEIGHTS,
    ) -> Dict:
        """Comprehensive bias analysis combining all signal sources.

        Parameters
        ----------
        overall_bias_threshold : float
            Minimum ``overall_bias_score`` to mark the text as biased.
        weights : tuple
            ``(fine_tuned, zero_shot, gender_keyword, discriminatory_keyword)``.
            The weight of an unavailable signal is spread over the others.
            If ``config/ml_weights.json`` exists (``eval/fit_weights.py``),
            the learned weights and threshold replace the defaults.
        """
        parts = self.signal_scores(text)
        s = parts['scores']
        names = ['fine_tuned', 'zero_shot', 'gender_keyword', 'discriminatory_keyword']

        if self.learned and weights == DEFAULT_WEIGHTS:
            w = dict(zip(names, self.learned['weights']))
            overall_bias_threshold = self.learned['threshold']
        else:
            w = dict(zip(names, weights))
            if not parts['ml_detection'].get('available', True):
                missing = w.pop('fine_tuned')
                total = sum(w.values())
                w = {k: v + missing * v / total for k, v in w.items()}
                w['fine_tuned'] = 0.0

        overall_bias_score = min(sum(s[k] * w[k] for k in names), 1.0)
        is_biased = overall_bias_score >= overall_bias_threshold
        severity = score_to_severity(overall_bias_score)

        return {
            'text': text,
            'ml_detection': parts['ml_detection'],
            'ml_categorization': parts['ml_categorization'],
            'gender_bias': parts['gender_bias'],
            'discriminatory_language': parts['discriminatory_language'],
            'findings': collect_findings(text, self.language),
            'signal_scores': {k: round(v, 4) for k, v in s.items()},
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
        report.append(f"\nBias Categorisation (zero-shot NLI, {_ZS_MODELS[self.language]}):")
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

        # Sentence-level findings with suggestions
        if analysis.get('findings'):
            report.append("\nFindings:")
            for f in analysis['findings']:
                line = f"  - [{f['subtype']}] \"{f['match']}\": {f['explanation']}"
                if f.get('suggestion'):
                    line += f" Suggestion: {f['suggestion']}"
                report.append(line)

        return '\n'.join(report)
