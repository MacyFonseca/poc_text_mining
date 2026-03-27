"""Shared keyword dictionaries, bias patterns, and keyword-based detection
logic used by both BiasDetector and MLBiasDetector.

Separates gendered *references* (neutral on their own) from *stereotypical
role associations* (biased when paired with gender) and adds explicit
discriminatory *patterns* that indicate real bias.  Also includes
positive-context phrases used to discount false positives in texts that
discuss gender equality rather than perpetuate bias.
"""
import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  GENDERED REFERENCES  (neutral – only used for ratio analysis)
# ---------------------------------------------------------------------------
GENDERED_REFERENCES = {
    'english': {
        'male': ['he', 'his', 'him', 'man', 'men', 'boy', 'male'],
        'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'female'],
    },
    'spanish': {
        'male': ['él', 'hombre', 'hombres', 'chico', 'masculino', 'varón'],
        'female': ['ella', 'mujer', 'mujeres', 'chica', 'femenina', 'femenino'],
    },
}

# ---------------------------------------------------------------------------
# 2.  STEREOTYPICAL ROLE ASSOCIATIONS
#     Roles or adjectives traditionally associated with one gender.
#     These only contribute to bias when they co-occur with gendered
#     references or explicit patterns.
# ---------------------------------------------------------------------------
STEREOTYPICAL_ASSOCIATIONS = {
    'english': {
        'male_coded': [
            'leader', 'manager', 'engineer', 'scientist', 'programmer',
            'strong', 'aggressive', 'ambitious', 'logical', 'assertive',
            'competitive', 'decisive',
        ],
        'female_coded': [
            'nurse', 'secretary', 'assistant', 'receptionist', 'support',
            'beautiful', 'emotional', 'nurturing', 'caring', 'charming',
            'supportive', 'helpful',
        ],
    },
    'spanish': {
        'male_coded': [
            'líder', 'gerente', 'ingeniero', 'científico', 'programador',
            'fuerte', 'agresivo', 'ambicioso', 'lógico', 'asertivo',
            'competitivo', 'decidido',
        ],
        'female_coded': [
            'enfermera', 'secretaria', 'asistenta', 'recepcionista', 'apoyo',
            'hermosa', 'emocional', 'maternal', 'cariñosa', 'encantadora',
            'servicial', 'amable',
        ],
    },
}

# ---------------------------------------------------------------------------
# 3.  EXPLICIT BIAS PATTERNS  (high-confidence discriminatory phrases)
#     Each pattern is a compiled regex matched against lowered text.
# ---------------------------------------------------------------------------
_EN_BIAS_PATTERNS = [
    # Gender + role requirements
    r'\b(male|female)\s+(engineer|scientist|manager|programmer|leader|nurse|secretary|assistant|receptionist|developer|architect)\b',
    r'\b(only\s+)?(men|women|males|females)\s+(should|may|can|need)\b',
    r'\bprefer\w*\s+(male|female|men|women)\b',
    r'\b(no|without)\s+(women|men|males|females)\b',
    r'\b(man|woman|boy|girl)\s+needed\b',
    r'\b(looking\s+for\s+a?\s*)(man|woman|male|female)\b',
    r'\b(must\s+be\s+)(male|female|a\s+man|a\s+woman)\b',
    # Appearance-linked gender
    r'\b(attractive|beautiful|handsome)\s+(woman|man|female|male|candidate)\b',
    # Age restrictions
    r'\bage\s*\d{1,2}\s*[-–]\s*\d{1,2}\b',
    r'\b(under|over)\s+\d{1,2}\s*(years)?\b',
    r'\b(young|old)\s+(man|woman|person|people|candidate|professional)\b',
]

_ES_BIAS_PATTERNS = [
    # Gender + role requirements
    r'\b(hombre|mujer|masculino|femenino|femenina)\s+(ingeniero|ingeniera|gerente|científico|científica|programador|programadora|enfermero|enfermera|secretario|secretaria)\b',
    r'\bse\s+busca\s+(hombre|mujer)\b',
    r'\bpreferiblemente\s+(hombre|mujer|masculino|femenino|femenina)\b',
    r'\bsolo\s+(hombres|mujeres)\b',
    r'\bexclusivamente\s+(hombres|mujeres|masculino|femenino)\b',
    # Age restrictions
    r'\bedad\s*\d{1,2}\s*[-–]\s*\d{1,2}\b',
    r'\b(menor|mayor)\s+de\s+\d{1,2}\s*(años)?\b',
    r'\b(joven|viejo|anciano)\s+(hombre|mujer|persona|profesional)\b',
]

BIAS_PATTERNS: Dict[str, List[re.Pattern]] = {
    'english': [re.compile(p, re.IGNORECASE) for p in _EN_BIAS_PATTERNS],
    'spanish': [re.compile(p, re.IGNORECASE) for p in _ES_BIAS_PATTERNS],
}

# ---------------------------------------------------------------------------
# 4.  DISCRIMINATORY KEYWORDS (by category)
# ---------------------------------------------------------------------------
DISCRIMINATORY_KEYWORDS = {
    'english': {
        'age': ['elderly', 'millennial', 'boomer', 'digital native',
                'outdated skills'],
        'race': ['minority', 'immigrant'],
        'disability': ['disabled', 'handicapped', 'special needs',
                       'health issues', 'physical limitations'],
        'appearance': ['attractive', 'overweight', 'skinny', 'handsome',
                       'good-looking'],
    },
    'spanish': {
        'age': ['anciano', 'millennial', 'boomer', 'nativo digital',
                'habilidades obsoletas'],
        'race': ['minoría', 'inmigrante'],
        'disability': ['discapacitado', 'minusválido', 'necesidades especiales',
                       'problemas de salud', 'limitaciones físicas'],
        'appearance': ['atractivo', 'atractiva', 'sobrepeso', 'delgado',
                       'delgada', 'guapo', 'guapa', 'buena presencia'],
    },
}

# ---------------------------------------------------------------------------
# 5.  POSITIVE / INCLUSIVE CONTEXT  (reduce false positives)
# ---------------------------------------------------------------------------
POSITIVE_CONTEXT_PHRASES = {
    'english': [
        'gender equality', 'equal opportunity', 'diversity and inclusion',
        'inclusive', 'inclusivity', 'empowerment', 'empower',
        'welcome applications from', 'underrepresented',
        'diverse backgrounds', 'all qualified candidates',
        'equal opportunities', 'regardless of gender',
        'perspective of gender', 'gender perspective',
    ],
    'spanish': [
        'igualdad de género', 'perspectiva de género', 'igualdad',
        'inclusión', 'inclusivo', 'inclusiva', 'empoderamiento',
        'empoderar', 'diversidad', 'equidad',
        'sin distinción de género', 'oportunidades iguales',
        'todas las personas', 'todos los candidatos',
        'candidatas y candidatos',
    ],
}

# ---------------------------------------------------------------------------
# 6.  SEVERITY THRESHOLDS  (used to map overall_bias_score → severity)
# ---------------------------------------------------------------------------
SEVERITY_LEVELS = {
    'none': (0.0, 0.10),
    'low': (0.10, 0.30),
    'medium': (0.30, 0.55),
    'high': (0.55, 0.80),
    'critical': (0.80, 1.01),
}


def score_to_severity(score: float) -> str:
    """Map a 0-1 bias score to a severity label."""
    for level, (lo, hi) in SEVERITY_LEVELS.items():
        if lo <= score < hi:
            return level
    return 'critical'


# ---------------------------------------------------------------------------
# 7.  SHARED DETECTION FUNCTIONS
# ---------------------------------------------------------------------------

def _find_keywords(keywords: List[str], text_lower: str) -> List[str]:
    """Return keywords that appear as whole words in *text_lower*."""
    return [
        kw for kw in keywords
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
    ]


def detect_gender_bias(text: str, language: str = 'english',
                       bias_threshold: float = 0.6) -> Dict:
    """Keyword- and pattern-based gender bias detection.

    Returns a dict with bias metrics, found keywords, matched patterns, and
    a positive-context discount.
    """
    text_lower = text.lower()

    refs = GENDERED_REFERENCES.get(language, GENDERED_REFERENCES['english'])
    stereo = STEREOTYPICAL_ASSOCIATIONS.get(language, STEREOTYPICAL_ASSOCIATIONS['english'])
    patterns = BIAS_PATTERNS.get(language, BIAS_PATTERNS['english'])
    positive = POSITIVE_CONTEXT_PHRASES.get(language, POSITIVE_CONTEXT_PHRASES['english'])

    # --- gendered references ---
    male_refs = _find_keywords(refs['male'], text_lower)
    female_refs = _find_keywords(refs['female'], text_lower)

    # --- stereotypical associations ---
    male_stereo = _find_keywords(stereo['male_coded'], text_lower)
    female_stereo = _find_keywords(stereo['female_coded'], text_lower)

    # --- explicit bias patterns ---
    matched_patterns = []
    for pat in patterns:
        m = pat.search(text_lower)
        if m:
            matched_patterns.append(m.group())

    # --- positive context ---
    positive_found = _find_keywords(positive, text_lower)
    positive_discount = min(len(positive_found) * 0.15, 0.6)

    # ----- scoring -----
    # Stereotype score: stereotypes that co-occur with the *opposite* gender
    # reference are more biased, but even standalone stereotypes contribute.
    male_keyword_count = len(male_refs) + len(male_stereo)
    female_keyword_count = len(female_refs) + len(female_stereo)
    total = male_keyword_count + female_keyword_count

    if total == 0 and not matched_patterns:
        return {
            'has_gender_bias': False,
            'male_bias_score': 0.0,
            'female_bias_score': 0.0,
            'bias_direction': 'neutral',
            'confidence': 0.0,
            'male_keywords_found': [],
            'female_keywords_found': [],
            'bias_patterns_matched': [],
            'positive_context_found': positive_found,
            'severity': 'none',
        }

    male_ratio = male_keyword_count / total if total else 0.0
    female_ratio = female_keyword_count / total if total else 0.0

    # Pattern matches are a strong signal
    pattern_score = min(len(matched_patterns) * 0.3, 1.0)

    # Keyword imbalance score
    keyword_imbalance = abs(male_ratio - female_ratio) if total else 0.0
    has_imbalance = max(male_ratio, female_ratio) > bias_threshold if total else False

    # Combined gender bias score
    raw_score = pattern_score * 0.6 + keyword_imbalance * 0.4
    adjusted_score = max(raw_score - positive_discount, 0.0)

    has_bias = bool(matched_patterns) or (has_imbalance and adjusted_score > 0.15)

    if male_ratio > female_ratio:
        bias_direction = 'male'
    elif female_ratio > male_ratio:
        bias_direction = 'female'
    else:
        bias_direction = 'balanced'

    return {
        'has_gender_bias': has_bias,
        'male_bias_score': round(male_ratio, 4),
        'female_bias_score': round(female_ratio, 4),
        'bias_direction': bias_direction,
        'confidence': round(adjusted_score, 4),
        'male_keywords_found': male_refs + male_stereo,
        'female_keywords_found': female_refs + female_stereo,
        'bias_patterns_matched': matched_patterns,
        'positive_context_found': positive_found,
        'severity': score_to_severity(adjusted_score),
    }


def detect_discriminatory_language(text: str,
                                   language: str = 'english') -> Dict:
    """Detect discriminatory language by category using keyword matching."""
    text_lower = text.lower()
    keywords_map = DISCRIMINATORY_KEYWORDS.get(
        language, DISCRIMINATORY_KEYWORDS['english']
    )
    results = {}
    for category, keywords in keywords_map.items():
        found = _find_keywords(keywords, text_lower)
        results[category] = {
            'has_discriminatory_language': len(found) > 0,
            'keywords_found': found,
            'count': len(found),
        }
    return results
