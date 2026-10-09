"""Bias review with a local instruction-tuned LLM served by Ollama (ROADMAP P2.4).

The LLM reads a page, quotes each biased or non-inclusive passage, classifies
it, explains why, and proposes an inclusive rewrite in the document's
language.  This gives span-level detection for Spanish without a Spanish
fine-tuned model, and feeds the recommendation system.

Everything runs locally (no document leaves the machine), which matters for
unpublished research proposals.  Requires Ollama (https://ollama.com):

    ollama pull llama3.1:8b          # or any model; set OLLAMA_MODEL

Same interface as ``BiasDetector`` / ``MLBiasDetector``; adds ``llm_review``.
Quotes the model returns that do not occur in the text are discarded, so the
model cannot invent findings.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List

from analysis.bias_keywords import (
    collect_findings,
    detect_discriminatory_language,
    detect_gender_bias as kw_detect_gender_bias,
    discriminatory_score as kw_discriminatory_score,
    noisy_or,
    score_to_severity,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')
DEFAULT_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
SEVERITY_SCORES = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
CATEGORIES = ['gender_stereotype', 'explicit_discrimination', 'generic_masculine',
              'androcentric', 'gendered_term', 'age', 'race', 'disability', 'appearance']
CATEGORY_GROUP = {'age': 'age', 'race': 'race', 'disability': 'disability', 'appearance': 'appearance'}

_SYSTEM_PROMPT = """You review research documents for gender bias, discriminatory language and non-inclusive language.

Flag a passage when it:
- sets a requirement based on gender, age, appearance, origin or disability (explicit_discrimination, age, appearance);
- attributes a trait or ability to a gender (gender_stereotype) or a negative trait to a social group (race, age, disability);
- uses an outdated or pejorative term for a group (race, age, disability);
- uses generic masculine for mixed groups, e.g. "los investigadores" (generic_masculine), the masculine for all humanity, e.g. "el hombre", "mankind" (androcentric), or gendered job titles, e.g. "chairman" (gendered_term).

Do NOT flag passages that report data about a group, study or criticise bias, promote equality, or refer to a specific named person.

Return JSON: {"findings": [{"quote": "...", "category": "...", "severity": "low|medium|high", "explanation": "...", "suggestion": "..."}]}
- "quote" must be copied exactly from the text, as short as possible (a few words).
- "explanation" in English, one sentence.
- "suggestion" is an inclusive rewrite of the quote, in the same language as the text.
- severity: high = explicit discrimination or hostile generalisation; medium = stereotype; low = non-inclusive wording.
- If nothing should be flagged, return {"findings": []}."""

_EXAMPLES = [
    ("Los investigadores presentarán un informe. Las mujeres representan el 38% de la plantilla.",
     {"findings": [{"quote": "Los investigadores", "category": "generic_masculine", "severity": "low",
                    "explanation": "Generic masculine for a mixed group.",
                    "suggestion": "El personal investigador"}]}),
    ("Girls are naturally worse at maths. The study examines gender stereotypes in schools.",
     {"findings": [{"quote": "Girls are naturally worse at maths", "category": "gender_stereotype",
                    "severity": "medium", "explanation": "Attributes a lower ability to girls.",
                    "suggestion": "Remove the claim or cite evidence on attainment by gender."}]}),
]

_SCHEMA = {
    'type': 'object',
    'properties': {'findings': {'type': 'array', 'items': {
        'type': 'object',
        'properties': {
            'quote': {'type': 'string'},
            'category': {'type': 'string', 'enum': CATEGORIES},
            'severity': {'type': 'string', 'enum': list(SEVERITY_SCORES)},
            'explanation': {'type': 'string'},
            'suggestion': {'type': 'string'},
        },
        'required': ['quote', 'category', 'severity', 'explanation', 'suggestion'],
    }}},
    'required': ['findings'],
}


class LLMBiasDetector:
    """Detect bias with a local LLM (Ollama) plus the shared keyword signals."""

    def __init__(self, language: str = 'english', model: str = DEFAULT_MODEL,
                 host: str = DEFAULT_HOST, timeout: int = 600, device: str = 'cpu'):
        self.language = language
        self.model = model
        self.host = host.rstrip('/')
        self.timeout = timeout

    # -- availability -------------------------------------------------------

    def is_available(self) -> bool:
        """True when the Ollama server is reachable and the model is pulled."""
        try:
            with urllib.request.urlopen(f'{self.host}/api/tags', timeout=3) as r:
                names = {m['name'] for m in json.load(r).get('models', [])}
            return self.model in names or f'{self.model}:latest' in names
        except (urllib.error.URLError, OSError, ValueError):
            return False

    # -- LLM review ---------------------------------------------------------

    def _chat(self, text: str) -> Dict:
        messages = [{'role': 'system', 'content': _SYSTEM_PROMPT}]
        for example_text, example_answer in _EXAMPLES:
            messages.append({'role': 'user', 'content': example_text})
            messages.append({'role': 'assistant', 'content': json.dumps(example_answer, ensure_ascii=False)})
        messages.append({'role': 'user', 'content': text})
        body = json.dumps({
            'model': self.model, 'messages': messages, 'stream': False,
            'format': _SCHEMA, 'options': {'temperature': 0, 'seed': 0, 'num_ctx': 8192},
        }).encode()
        req = urllib.request.Request(f'{self.host}/api/chat', data=body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(json.load(r)['message']['content'])

    def review(self, text: str) -> Dict:
        """Ask the LLM for findings; keep only quotes that occur in *text*."""
        try:
            raw = self._chat(text).get('findings', [])
        except Exception as e:
            logger.error("LLM review failed: %s", e)
            return {'available': False, 'error': str(e), 'findings': [], 'score': 0.0}

        findings, seen = [], set()
        for f in raw:
            quote = (f.get('quote') or '').strip().strip('"')
            m = re.search(re.escape(quote), text, re.I) if quote else None
            if not m or (m.start(), m.end()) in seen or f.get('category') not in CATEGORIES:
                continue
            seen.add((m.start(), m.end()))
            score = SEVERITY_SCORES.get(f.get('severity'), 0.3)
            findings.append({
                'type': 'llm',
                'category': CATEGORY_GROUP.get(f['category'], 'gender'),
                'subtype': f['category'],
                'match': text[m.start():m.end()],
                'start': m.start(),
                'end': m.end(),
                'sentence': None,
                'score': score,
                'explanation': f.get('explanation', ''),
                'suggestion': f.get('suggestion') or None,
            })
        return {'available': True, 'model': self.model, 'findings': findings,
                'score': round(noisy_or([f['score'] for f in findings]), 4)}

    # -- keyword / pattern helpers (delegate to shared module) --------------

    def detect_gender_bias(self, text: str, bias_threshold: float = 0.6) -> Dict:
        return kw_detect_gender_bias(text, language=self.language, bias_threshold=bias_threshold)

    def detect_discriminatory_language(self, text: str) -> Dict:
        return detect_discriminatory_language(text, language=self.language)

    # -- comprehensive analysis ---------------------------------------------

    def comprehensive_bias_analysis(self, text: str,
                                    overall_bias_threshold: float = 0.25,
                                    weights: tuple = (0.6, 0.2, 0.2)) -> Dict:
        """Combine the LLM review with keyword signals.

        weights : ``(llm, gender_keyword, discriminatory_keyword)``.  When the
        LLM is unreachable its weight is spread over the keyword signals, so
        the result degrades to the keyword detector instead of failing.
        """
        llm = self.review(text)
        gender_bias = self.detect_gender_bias(text)
        discriminatory = self.detect_discriminatory_language(text)

        w_llm, w_g, w_d = weights
        if not llm['available']:
            w_g, w_d = w_g + w_llm * w_g / (w_g + w_d), w_d + w_llm * w_d / (w_g + w_d)
            w_llm = 0.0
        overall = min(llm['score'] * w_llm + gender_bias['confidence'] * w_g
                      + kw_discriminatory_score(discriminatory) * w_d, 1.0)

        keyword_findings = collect_findings(text, self.language)
        return {
            'text': text,
            'llm_review': llm,
            'gender_bias': gender_bias,
            'discriminatory_language': discriminatory,
            'findings': sorted(keyword_findings + llm['findings'], key=lambda f: f['start']),
            'overall_bias_score': round(overall, 4),
            'is_biased': overall >= overall_bias_threshold,
            'severity': score_to_severity(overall),
        }

    def batch_analysis(self, texts: List[str]) -> List[Dict]:
        return [self.comprehensive_bias_analysis(t) for t in texts]

    def generate_bias_report(self, analysis: Dict) -> str:
        lines = ['=' * 60, 'LLM BIAS REVIEW REPORT', '=' * 60,
                 f"\nOverall Bias Score: {analysis['overall_bias_score']:.2%}",
                 f"Severity: {analysis['severity'].upper()}",
                 f"Is Biased: {analysis['is_biased']}",
                 f"Model: {analysis['llm_review'].get('model', 'unavailable')}", '\nFindings:']
        for f in analysis['findings']:
            line = f"  - [{f['subtype']}] \"{f['match']}\": {f['explanation']}"
            if f.get('suggestion'):
                line += f" Suggestion: {f['suggestion']}"
            lines.append(line)
        return '\n'.join(lines)
