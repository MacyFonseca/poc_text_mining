"""Shared keyword dictionaries, bias patterns, and keyword-based detection
logic used by every detector (BiasDetector, MLBiasDetector, LLMBiasDetector).

Analysis is done **per sentence** (spaCy) and then aggregated per page, so a
single biased sentence is not diluted by a long neutral page, and every hit
carries its character offsets, an explanation, and (when one exists) an
inclusive-language suggestion.

Kinds of evidence, from strongest to weakest:

* **Explicit patterns** - regexes for discriminatory requirements
  ("se busca hombre", "only men may apply").  Never discounted.
* **Stereotypes** - a gendered reference in the same sentence as a
  stereotyped trait, a generalisation ("las mujeres son...") or a
  capability marker ("por naturaleza", "worse at").
* **Discriminatory language** - outdated or pejorative terms
  ("minusválido", "illegal aliens") and negative generalisations about a
  social group ("los inmigrantes son una carga").  A *neutral* mention of a
  group ("población inmigrante") is recorded but never scored.
* **Non-inclusive language** - Spanish generic masculine ("los
  investigadores"), androcentric forms ("el hombre", "mankind"), gendered
  job titles ("chairman") and generic "he".  Capped per page so that it can
  raise, but never dominate, the score.

Sentences that *discuss* bias ("el estudio analiza los estereotipos...")
discount stereotype evidence in that sentence only.

All matching is lemma-based, so inflected forms (enfermeras, agresivas) are
found from a single base form in the lists.
"""
import re
import logging
from functools import lru_cache
from typing import Dict, List, Optional

from utils.nlp import get_nlp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  GENDERED REFERENCES  (neutral alone - used for direction + co-occurrence)
#     Base forms; matched against token lemma *and* lowercase surface form.
# ---------------------------------------------------------------------------
GENDERED_REFERENCES = {
    'english': {
        'male': ['he', 'his', 'him', 'man', 'men', 'boy', 'male', 'father',
                 'husband', 'son', 'brother', 'gentleman', 'dad'],
        'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'female',
                   'mother', 'wife', 'daughter', 'sister', 'lady', 'mom',
                   'mum', 'housewife'],
    },
    'spanish': {
        'male': ['él', 'hombre', 'chico', 'niño', 'padre', 'esposo', 'marido',
                 'hijo', 'hermano', 'señor', 'varón', 'masculino'],
        'female': ['ella', 'mujer', 'chica', 'niña', 'madre', 'esposa', 'hija',
                   'hermana', 'señora', 'señorita', 'femenino', 'femenina',
                   'ama de casa'],
    },
}

# Spanish feminine role nouns are themselves gendered references
# ("las enfermeras son cariñosas").  Masculine forms are not used here
# because they are usually generic masculine, handled separately.
FEMININE_ROLE_NOUNS_ES = [
    'enfermera', 'secretaria', 'maestra', 'ingeniera', 'científica',
    'investigadora', 'profesora', 'doctora', 'directora', 'cuidadora',
    'limpiadora', 'azafata', 'niñera', 'asistenta', 'empleada', 'trabajadora',
    'becaria', 'alumna', 'médica', 'abogada', 'jefa', 'técnica', 'madre',
]

# ---------------------------------------------------------------------------
# 2.  STEREOTYPED TRAITS AND MARKERS
#     Traits only count when they co-occur with a gendered reference.
#     Markers ("naturally", "por naturaleza") turn a mention into a claim.
# ---------------------------------------------------------------------------
STEREOTYPICAL_ASSOCIATIONS = {
    'english': {
        'male_coded': [
            'leader', 'aggressive', 'strong', 'dominant', 'logical',
            'rational', 'assertive', 'decisive', 'competitive', 'tough',
            'brave', 'ambitious',
        ],
        'female_coded': [
            'emotional', 'caring', 'nurturing', 'gentle', 'sensitive',
            'sweet', 'submissive', 'weak', 'fragile', 'delicate',
            'hysterical', 'bossy', 'beautiful', 'pretty', 'docile',
            'irrational', 'motherly', 'maternal', 'supporting role',
            'supportive',
        ],
    },
    'spanish': {
        'male_coded': [
            'líder', 'agresivo', 'fuerte', 'dominante', 'lógico', 'racional',
            'asertivo', 'decidido', 'competitivo', 'duro', 'valiente',
            'ambicioso', 'inquieto',
        ],
        'female_coded': [
            'emocional', 'sensible', 'cariñoso', 'tierno', 'dulce',
            'dulzura', 'ternura', 'delicado', 'delicadeza', 'débil', 'frágil',
            'sumiso', 'maternal', 'histérico', 'guapo', 'bonito', 'hermoso',
            'dócil', 'tranquilo', 'paciente', 'servicial', 'abnegado',
        ],
    },
}

STEREOTYPE_MARKERS = {
    'english': [
        r'\bnaturally\b', r'\bby\s+nature\b', r'\binnately\b', r'\binherently\b',
        r'\bnatural\s+(leaders?|caregivers?|carers?)\b', r'\b(worse|better)\s+at\b',
        r'\bnot\s+(good|suited|fit|capable)\b', r'\bincapable\b', r'\bunable\s+to\b',
        r'\btoo\s+(emotional|weak|sensitive|aggressive|fragile)\b',
        r'\bless\s+(committed|capable|productive|competent|suited|reliable|dedicated|rational|logical)\b',
        r'\bbelong\s+(in|at)\b', r'\bstay\s+at\s+home\b',
    ],
    'spanish': [
        r'\bpor\s+naturaleza\b', r'\bnaturalmente\b', r'\binnat[oa]s?\b',
        r'\bpor\s+instinto\b', r'\bno\s+tienen\s+la\s+capacidad\b',
        r'\bno\s+son\s+capaces\b', r'\bincapaces\b',
        r'\bmenos\s+(productiv|capac|competent|comprometid|fiabl|racional|lógic)\w*',
        r'\bmás\s+(emocional|sensibl|débil|frágil|agresiv|dulce|tranquil|inquiet)\w*',
        r'\b(peores|mejores)\s+(en|para)\b', r'\bno\s+sirven\b',
        r'\bquedarse\s+en\s+casa\b',
    ],
}

# Group nouns (plural) whose use as a sentence subject signals a generalisation
# when followed by one of the generalisation verbs below.
GENDER_GROUPS = {
    'english': ['women', 'men', 'girls', 'boys', 'females', 'males',
                'mothers', 'fathers', 'wives', 'husbands', 'working mothers'],
    'spanish': ['mujeres', 'hombres', 'chicas', 'chicos', 'niñas', 'niños',
                'madres', 'padres', 'esposas', 'varones']
               + [n + 's' for n in FEMININE_ROLE_NOUNS_ES if not n.endswith('s')],
}

SOCIAL_GROUPS = {
    'english': {
        'race': ['immigrants', 'migrants', 'refugees', 'foreigners', 'gypsies',
                 'muslims', 'blacks', 'minorities'],
        'age': ['the elderly', 'elderly people', 'old people', 'older people',
                'young people', 'millennials', 'boomers'],
        'disability': ['the disabled', 'disabled people', 'handicapped people'],
    },
    'spanish': {
        'race': ['inmigrantes', 'migrantes', 'refugiados', 'extranjeros',
                 'gitanos', 'musulmanes', 'moros', 'negros', 'minorías'],
        'age': ['ancianos', 'viejos', 'personas mayores', 'jóvenes',
                'millennials', 'boomers'],
        'disability': ['discapacitados', 'minusválidos'],
    },
}

GENERALISATION_VERBS = {
    'english': r'(are|tend\s+to\s+be|tend\s+to|cannot|can\'t|lack|do\s+not|don\'t|should|must|belong)',
    'spanish': r'(son|suelen|tienden|deben|no\s+(son|tienen|pueden|saben|sirven|deben))',
}

NEGATIVE_ATTRIBUTES = {
    'english': [r'\bburden\b', r'\bcriminals?\b', r'\bdangerous\b', r'\blazy\b',
                r'\bincapable\b', r'\binferior\b', r'\bproblematic\b', r'\bdirty\b',
                r'\bviolent\b', r'\bunable\b', r'\bthreat\b', r'\bparasites?\b'],
    'spanish': [r'\bcarga\b', r'\bdelincuentes?\b', r'\bpeligros[oa]s?\b', r'\bvag[oa]s?\b',
                r'\bincapaces\b', r'\bno\s+son\s+capaces\b', r'\binferiores?\b',
                r'\bconflictiv[oa]s?\b', r'\bproblemátic[oa]s?\b', r'\bsuci[oa]s?\b',
                r'\bviolent[oa]s?\b', r'\bamenaza\b', r'\bparásitos?\b'],
}

# ---------------------------------------------------------------------------
# 3.  EXPLICIT BIAS PATTERNS  (high-confidence discriminatory requirements)
# ---------------------------------------------------------------------------
_EN_BIAS_PATTERNS = [
    r'\b(looking\s+for|seeking|hiring|recruit\w*|need)\s+(a\s+|an\s+)?(male|female|man|woman|men|women)\b',
    r'\bonly\s+(men|women|males|females)\b',
    r'\b(men|women|girls|boys)\s+(should|must)\s+(stay|remain|leave|not\b|be\s+(kept|excluded))',
    r'\b(men|women)\s+(are\s+)?not\s+(suited|fit|capable)\b',
    r'\bprefer\w*\s+(male|female|men|women)\b',
    r'\b(no|without)\s+(women|men|males|females)\s+(allowed|accepted|may|need)\b',
    r'\b(man|woman|boy|girl)\s+needed\b',
    r'\bmust\s+be\s+(male|female|a\s+man|a\s+woman)\b',
]

_ES_BIAS_PATTERNS = [
    r'\b(se\s+busca|buscamos|se\s+necesita|necesitamos|se\s+requiere|contratar\w*)\s+(un\s+|una\s+)?(hombres?|mujer(es)?|varón|varones|chicos?|chicas?)\b',
    r'\b(preferiblemente|preferentemente|exclusivamente|solo|sólo|únicamente)\s+(hombres?|mujer(es)?|varón|varones|masculino|femenin[oa])\b',
    r'\b(las\s+)?mujeres\s+(deben|tienen\s+que|no\s+deben|no\s+pueden)\s+\w+',
    r'\b(los\s+)?hombres\s+(deben|tienen\s+que|no\s+deben|no\s+pueden)\s+\w+',
    r'\b(hombres?|mujer(es)?)\s+(no\s+)?(son\s+)?(aptos?|aptas?)\b',
]

BIAS_PATTERNS: Dict[str, List[re.Pattern]] = {
    'english': [re.compile(p, re.IGNORECASE) for p in _EN_BIAS_PATTERNS],
    'spanish': [re.compile(p, re.IGNORECASE) for p in _ES_BIAS_PATTERNS],
}

# Age and appearance requirements (scored as discriminatory language).
# The age number 18 is ignored: "mayores de 18 años" is the legal age of
# consent in ethics sections, not age discrimination.
_AGE_REQUIREMENT = {
    'english': re.compile(
        r'\b(candidates?|applicants?|must\s+be|should\s+be|aged?|preferably)\b[^.;]{0,30}?'
        r'\b(under|over|below|above|younger\s+than|older\s+than)\s+(\d{2})\b', re.I),
    'spanish': re.compile(
        r'\b(candidat[oa]s?|aspirantes?|edad|ser|preferiblemente|hombre|mujer|buscamos|se\s+busca)\b[^.;]{0,30}?'
        r'\b(menor|mayor|menores|mayores)\s+de\s+(\d{2})\b', re.I),
}
_AGE_RANGE = {
    'english': re.compile(r'\bage(d)?\s*\d{2}\s*[-–]\s*\d{2}\b', re.I),
    'spanish': re.compile(r'\bedad\s*(entre\s*)?\d{2}\s*([-–]|y)\s*\d{2}\b', re.I),
}
_YOUTH_REQUIREMENT = {
    'english': re.compile(r'\b(young|younger)\s+(candidates?|professionals?|applicants?|scientists?)\b', re.I),
    'spanish': re.compile(r'\b(buscamos|se\s+busca|se\s+requiere|se\s+necesita)\b[^.;]{0,40}\bjóvenes?\b', re.I),
}
_APPEARANCE_PATTERNS = {
    'english': [
        re.compile(r'\bgood\s+(appearance|looks|presence)\b', re.I),
        re.compile(r'\b(attractive|beautiful|handsome|good-looking|pretty)\s+(young\s+)?'
                   r'(woman|man|girl|boy|female|male|candidates?|applicants?|secretary|assistant)\b', re.I),
    ],
    'spanish': [
        re.compile(r'\bbuena\s+presencia\b', re.I),
        re.compile(r'\b(chica|chico|mujer|hombre|secretaria|azafata|candidat[oa])\s+(muy\s+)?'
                   r'(guap[oa]|atractiv[oa]|mon[oa]|bonit[oa])\b', re.I),
    ],
}

# ---------------------------------------------------------------------------
# 4.  DISCRIMINATORY TERMS  (problematic on their own)
#     term regex -> (category, score, suggestion)
# ---------------------------------------------------------------------------
DISCRIMINATORY_TERMS = {
    'english': {
        r'\bhandicapped\b': ('disability', 0.7, 'people with disabilities / disabled people'),
        r'\bcrippled?\b': ('disability', 0.7, 'person with a physical disability'),
        r'\b(mentally\s+)?retarded\b': ('disability', 0.8, 'person with an intellectual disability'),
        r'\b(wheelchair[- ]bound|confined\s+to\s+a\s+wheelchair)\b': ('disability', 0.7, 'wheelchair user'),
        r'\bsuffer(s|ing)?\s+from\s+(autism|down\s+syndrome|a\s+disability|blindness|deafness|cerebral\s+palsy)\b':
            ('disability', 0.5, 'has / lives with (neutral wording)'),
        r'\bthe\s+disabled\b': ('disability', 0.5, 'disabled people / people with disabilities'),
        r'\billegal\s+(aliens?|immigrants?|migrants?)\b': ('race', 0.7, 'undocumented migrants'),
        r'\bthe\s+elderly\b': ('age', 0.35, 'older adults / older people'),
        r'\bsenile\b': ('age', 0.6, 'person with dementia'),
    },
    'spanish': {
        r'\bminusválid[oa]s?\b': ('disability', 0.7, 'personas con discapacidad'),
        r'\binválid[oa]s?\b': ('disability', 0.6, 'personas con discapacidad'),
        r'\bsubnormal(es)?\b': ('disability', 0.9, 'persona con discapacidad intelectual'),
        r'\bretrasad[oa]s?\s+mental(es)?\b': ('disability', 0.8, 'persona con discapacidad intelectual'),
        r'\b(los|las|unos|unas)\s+discapacitad[oa]s\b': ('disability', 0.5, 'las personas con discapacidad'),
        r'\b(confinad[oa]|postrad[oa]|atad[oa])\s+a\s+una\s+silla\s+de\s+ruedas\b':
            ('disability', 0.7, 'persona usuaria de silla de ruedas'),
        r'\b(sufre|sufren|padece|padecen)\s+(de\s+)?(autismo|síndrome\s+de\s+down|una\s+discapacidad|ceguera|sordera)\b':
            ('disability', 0.5, 'tiene / vive con (redacción neutra)'),
        r'\b(inmigrantes?|migrantes?)\s+ilegales?\b': ('race', 0.7, 'personas migrantes en situación irregular'),
        r'\bsudacas?\b': ('race', 0.9, 'personas latinoamericanas'),
        r'\bancian[oa]s?\b': ('age', 0.35, 'personas mayores'),
        r'\bviejos\b': ('age', 0.4, 'personas mayores'),
        r'\bsenil(es)?\b': ('age', 0.6, 'persona con demencia'),
    },
}

# Kept for backward compatibility / reporting: neutral group mentions.
DISCRIMINATORY_KEYWORDS = {
    lang: {cat: list(terms) for cat, terms in groups.items()} | {'appearance': []}
    for lang, groups in SOCIAL_GROUPS.items()
}
DISCRIMINATION_CATEGORIES = ['age', 'race', 'disability', 'appearance']

# ---------------------------------------------------------------------------
# 5.  NON-INCLUSIVE LANGUAGE
# ---------------------------------------------------------------------------
# Spanish person nouns that are usually used as generic masculine in the
# plural.  lemma -> (feminine plural, inclusive alternative)
GENERIC_MASCULINE_ES = {
    'investigador': ('investigadoras', 'el personal investigador / el equipo de investigación'),
    'alumno': ('alumnas', 'el alumnado'),
    'profesor': ('profesoras', 'el profesorado / el personal docente'),
    'director': ('directoras', 'la dirección'),
    'coordinador': ('coordinadoras', 'la coordinación / las personas coordinadoras'),
    'becario': ('becarias', 'el personal becario'),
    'colaborador': ('colaboradoras', 'el equipo colaborador / las personas colaboradoras'),
    'usuario': ('usuarias', 'las personas usuarias'),
    'trabajador': ('trabajadoras', 'la plantilla / las personas trabajadoras'),
    'ciudadano': ('ciudadanas', 'la ciudadanía'),
    'niño': ('niñas', 'la infancia / niñas y niños'),
    'autor': ('autoras', 'la autoría / las personas autoras'),
    'doctor': ('doctoras', 'el personal doctor / doctoras y doctores'),
    'técnico': ('técnicas', 'el personal técnico'),
    'científico': ('científicas', 'la comunidad científica / el personal científico'),
    'experto': ('expertas', 'las personas expertas'),
    'evaluador': ('evaluadoras', 'el panel evaluador / las personas evaluadoras'),
    'candidato': ('candidatas', 'las candidaturas / las personas candidatas'),
    'beneficiario': ('beneficiarias', 'las personas beneficiarias'),
    'empleado': ('empleadas', 'la plantilla / el personal'),
    'jefe': ('jefas', 'la jefatura'),
    'interesado': ('interesadas', 'las personas interesadas'),
    'ingeniero': ('ingenieras', 'el personal de ingeniería'),
    'médico': ('médicas', 'el personal médico'),
    'enfermero': ('enfermeras', 'el personal de enfermería'),
    'tutor': ('tutoras', 'las personas tutoras / la tutoría'),
    'compañero': ('compañeras', 'el equipo / las personas compañeras'),
    'socio': ('socias', 'las entidades socias'),
    'hijo': ('hijas', 'las hijas y los hijos / la descendencia'),
    'abuelo': ('abuelas', 'las personas mayores'),
    'educador': ('educadoras', 'el personal educador'),
    'asesor': ('asesoras', 'el equipo asesor'),
    'consumidor': ('consumidoras', 'las personas consumidoras'),
    'votante': ('votantes', 'el electorado'),
    'vecino': ('vecinas', 'el vecindario'),
    'funcionario': ('funcionarias', 'el funcionariado / el personal funcionario'),
}
# Epicene nouns: only generic masculine when a masculine determiner precedes
# ("los participantes" -> "las personas participantes").
EPICENE_NOUNS_ES = {
    'participante': 'las personas participantes',
    'estudiante': 'el estudiantado',
    'solicitante': 'las personas solicitantes',
    'paciente': 'las personas pacientes / la población atendida',
    'responsable': 'las personas responsables',
    'docente': 'el personal docente',
    'representante': 'la representación / las personas representantes',
    'integrante': 'las personas integrantes',
    'joven': 'la juventud / las personas jóvenes',
    'adolescente': 'la adolescencia / las personas adolescentes',
}

# Androcentric or gendered forms with neutral alternatives.
NON_INCLUSIVE_TERMS = {
    'english': {
        r'\bmankind\b': ('androcentric', 'humankind / humanity'),
        r'\b(early|modern|primitive|prehistoric)\s+man\b': ('androcentric', 'early humans / modern humans'),
        r'\bthe\s+common\s+man\b': ('androcentric', 'ordinary people'),
        r'\bman-made\b': ('gendered_term', 'artificial / synthetic / human-made'),
        r'\bmanpower\b': ('gendered_term', 'workforce / staff / personnel'),
        r'\bman-hours?\b': ('gendered_term', 'person-hours / work hours'),
        r'\bchair(man|men)\b': ('gendered_term', 'chair / chairperson'),
        r'\bpolice(man|men)\b': ('gendered_term', 'police officer(s)'),
        r'\bfire(man|men)\b': ('gendered_term', 'firefighter(s)'),
        r'\bbusiness(man|men)\b': ('gendered_term', 'businessperson / business people'),
        r'\bspokes(man|men)\b': ('gendered_term', 'spokesperson'),
        r'\bsales(man|men)\b': ('gendered_term', 'salesperson / sales staff'),
        r'\bfore(man|men)\b': ('gendered_term', 'supervisor'),
        r'\blay(man|men)\b': ('gendered_term', 'layperson / non-specialist'),
        r'\bcongress(man|men)\b': ('gendered_term', 'member(s) of Congress'),
        r'\bforefathers\b': ('gendered_term', 'ancestors / predecessors'),
    },
    'spanish': {
        r'\b(el|del|al)\s+hombre\b(?!\s+y\s+la\s+mujer)': ('androcentric', 'el ser humano / la humanidad / las personas'),
        r'\bderechos\s+del\s+hombre\b': ('androcentric', 'derechos humanos'),
        r'\bhombres\s+de\s+(ciencia|negocios|estado)\b': ('androcentric', 'personas de ciencia / de negocios'),
        r'\bseñorita\b': ('asymmetric_treatment', 'señora, o el nombre y apellido sin tratamiento'),
        r'\bpaternidad\s+responsable\b': ('androcentric', 'maternidad y paternidad responsables'),
    },
}

# English generic "he": a singular generic person noun followed by he/his/him.
_GENERIC_HE_NOUNS = (
    'researcher|participant|student|scientist|applicant|candidate|author|user|'
    'patient|employee|doctor|teacher|professor|investigator|reviewer|child|'
    'person|individual|respondent|member|worker|manager|engineer|nurse|reader|'
    'client|citizen|scholar'
)
_GENERIC_HE = re.compile(
    rf'\b(each|every|a|an|any|the|no)\s+(\w+\s+)?({_GENERIC_HE_NOUNS})\b[^.;]{{0,80}}?\b(he|his|him|himself)\b',
    re.I,
)

# ---------------------------------------------------------------------------
# 6.  CONTEXT THAT DISCOUNTS STEREOTYPE EVIDENCE (per sentence)
#     Sentences that promote equality or *study* bias are not themselves biased.
# ---------------------------------------------------------------------------
POSITIVE_CONTEXT_PHRASES = {
    'english': [
        'gender equality', 'equal opportunity', 'equal opportunities',
        'diversity and inclusion', 'inclusion', 'inclusive', 'empowerment',
        'empower', 'welcome applications from', 'underrepresented',
        'under-represented', 'diverse backgrounds', 'all qualified candidates',
        'regardless of gender', 'gender perspective', 'stereotype',
        'stereotypes', 'gender bias', 'gender gap', 'pay gap',
        'discrimination', 'gender-based violence', 'sexism',
    ],
    'spanish': [
        'igualdad de género', 'igualdad de oportunidades', 'igualdad entre',
        'perspectiva de género', 'inclusión', 'lenguaje inclusivo',
        'empoderamiento', 'empoderar', 'sin distinción de',
        'infrarrepresentad', 'estereotipo', 'estereotipos', 'sesgo', 'sesgos',
        'brecha', 'discriminación', 'violencia de género', 'sexismo',
        'candidatas y candidatos', 'todas las personas',
    ],
}
POSITIVE_CONTEXT_DISCOUNT = 0.4   # multiply stereotype scores by this

# ---------------------------------------------------------------------------
# 7.  SCORES AND SEVERITY
# ---------------------------------------------------------------------------
FINDING_SCORES = {
    'explicit': 0.8,
    'stereotype_strong': 0.6,     # generalisation/marker + trait, or 2+ traits
    'stereotype_weak': 0.3,       # one trait next to a gendered reference
    'generalisation': 0.3,        # "women are ..." with no trait
    'negative_generalisation': 0.9,
    'age_requirement': 0.7,
    'appearance_requirement': 0.7,
    'non_inclusive': 0.45,
}
NON_INCLUSIVE_PAGE_CAP = 0.6

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


def noisy_or(scores: List[float]) -> float:
    """Combine independent pieces of evidence: 1 - prod(1 - s)."""
    remaining = 1.0
    for s in scores:
        remaining *= (1.0 - max(0.0, min(s, 1.0)))
    return 1.0 - remaining


# ---------------------------------------------------------------------------
# 8.  SENTENCE-LEVEL ANALYSIS
# ---------------------------------------------------------------------------

def _compile_word_list(words: List[str]) -> re.Pattern:
    return re.compile(r'\b(' + '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r')\b', re.I)


_GENERALISATION_RE = {
    lang: {
        'gender': re.compile(
            r'\b(' + '|'.join(re.escape(g) for g in GENDER_GROUPS[lang]) + r')\b'
            r'(\s+[\w-]+){0,2}?\s+' + GENERALISATION_VERBS[lang] + r'\b', re.I),
        **{
            cat: re.compile(
                r'\b(' + '|'.join(re.escape(g) for g in groups) + r')\b'
                r'(\s+[\w-]+){0,2}?\s+' + GENERALISATION_VERBS[lang] + r'\b', re.I)
            for cat, groups in SOCIAL_GROUPS[lang].items()
        },
    }
    for lang in ('english', 'spanish')
}
_GROUP_MENTION_RE = {
    lang: {cat: _compile_word_list(groups + [g.rstrip('s') for g in groups if ' ' not in g])
           for cat, groups in SOCIAL_GROUPS[lang].items()}
    for lang in ('english', 'spanish')
}
_MARKER_RE = {lang: [re.compile(p, re.I) for p in pats] for lang, pats in STEREOTYPE_MARKERS.items()}
_NEGATIVE_RE = {lang: [re.compile(p, re.I) for p in pats] for lang, pats in NEGATIVE_ATTRIBUTES.items()}
_DISCRIM_TERM_RE = {lang: [(re.compile(p, re.I), v) for p, v in terms.items()]
                    for lang, terms in DISCRIMINATORY_TERMS.items()}
_NON_INCLUSIVE_RE = {lang: [(re.compile(p, re.I), v) for p, v in terms.items()]
                     for lang, terms in NON_INCLUSIVE_TERMS.items()}


def _token_matches(sent, words: List[str], pos: Optional[set] = None) -> List:
    """Tokens of *sent* whose lemma or lowercase form is in *words*."""
    single = {w for w in words if ' ' not in w}
    out = []
    for tok in sent:
        if pos and tok.pos_ not in pos:
            continue
        if tok.lower_ in single or tok.lemma_.lower() in single:
            out.append(tok)
    return out


def _phrase_matches(sent_text: str, words: List[str]) -> List[re.Match]:
    multi = [w for w in words if ' ' in w]
    return [m for w in multi for m in re.finditer(r'\b' + re.escape(w) + r'\b', sent_text, re.I)]


def _finding(sent, start, end, kind, category, subtype, score, explanation, suggestion=None):
    text = sent.doc.text
    return {
        'type': kind,
        'category': category,
        'subtype': subtype,
        'match': text[start:end],
        'start': start,
        'end': end,
        'sentence': sent.text.strip(),
        'score': round(score, 4),
        'explanation': explanation,
        'suggestion': suggestion,
    }


def _gendered_tokens(sent, language: str) -> Dict[str, List]:
    refs = GENDERED_REFERENCES[language]
    male = _token_matches(sent, refs['male'])
    female = _token_matches(sent, refs['female'])
    if language == 'spanish':
        female += [t for t in sent if t.pos_ == 'NOUN' and t.lemma_.lower() in FEMININE_ROLE_NOUNS_ES
                   and 'Fem' in t.morph.get('Gender') and t not in female]
        # "hombre" used generically is androcentric, not a male reference
        male = [t for t in male if not (t.lower_ == 'hombre' and t.i > sent.start
                                        and t.nbor(-1).lower_ in ('el', 'del', 'al'))]
    return {'male': male, 'female': female}


def _epicene_key(tok) -> Optional[str]:
    """Singular form of an epicene noun ("participantes" -> "participante")."""
    lower = tok.lower_
    for key in (tok.lemma_.lower(), lower[:-1] if lower.endswith('s') else None,
                lower[:-2] if lower.endswith('es') else None):
        if key in EPICENE_NOUNS_ES:
            return key
    return None


def _detect_generic_masculine_es(sent) -> List[Dict]:
    findings = []
    lower = sent.text.lower()
    for tok in sent:
        lemma = tok.lemma_.lower()
        number = tok.morph.get('Number')
        if 'Plur' not in number:
            continue
        if tok.dep_ == 'amod':
            continue   # adjective modifying a noun: "servicios técnicos", "personas participantes"
        suggestion = None
        if tok.pos_ in ('NOUN', 'ADJ') and lemma in GENERIC_MASCULINE_ES and 'Masc' in tok.morph.get('Gender'):
            fem_plural, suggestion = GENERIC_MASCULINE_ES[lemma]
            if tok.lower_ == fem_plural:
                continue   # the feminine form itself, mis-tagged as masculine
            if re.search(r'\b' + re.escape(fem_plural) + r'\b', lower):
                continue   # doubled form: "investigadoras e investigadores"
        elif _epicene_key(tok):
            key = _epicene_key(tok)
            prev = [t for t in sent[max(0, tok.i - sent.start - 2):tok.i - sent.start]]
            if not any(t.pos_ == 'DET' and 'Masc' in t.morph.get('Gender') for t in prev):
                continue
            if any(t.lower_ in ('las', 'unas', 'estas') for t in prev):
                continue
            suggestion = EPICENE_NOUNS_ES[key]
        else:
            continue
        # Explicit inclusive marks right after the word: investigadores/as, (as), -as
        tail = sent.doc.text[tok.idx + len(tok.text): tok.idx + len(tok.text) + 5]
        if re.match(r'\s*(/|\(|-)\s*(a|as)\b', tail):
            continue
        findings.append(_finding(
            sent, tok.idx, tok.idx + len(tok.text), 'non_inclusive', 'gender', 'generic_masculine',
            FINDING_SCORES['non_inclusive'],
            'Generic masculine used for a mixed or unknown group.',
            suggestion,
        ))
    # "todos" used alone as a pronoun ("todos deberán...")
    for tok in sent:
        if tok.lower_ == 'todos' and tok.pos_ == 'PRON':
            findings.append(_finding(
                sent, tok.idx, tok.idx + len(tok.text), 'non_inclusive', 'gender', 'generic_masculine',
                FINDING_SCORES['non_inclusive'], 'Generic masculine pronoun.', 'todas las personas / todo el mundo',
            ))
    return findings


def _analyse_sentence(sent, language: str) -> Dict:
    """Return findings and reference tokens for one spaCy sentence span."""
    text = sent.text
    base = sent.start_char
    findings: List[Dict] = []

    def span(m, group=0):
        return base + m.start(group), base + m.end(group)

    # --- explicit discriminatory requirements ---
    for pat in BIAS_PATTERNS[language]:
        for m in pat.finditer(text):
            s, e = span(m)
            findings.append(_finding(sent, s, e, 'explicit', 'gender', 'explicit_requirement',
                                     FINDING_SCORES['explicit'],
                                     'Explicit gender requirement or prescription.', None))

    m = _AGE_REQUIREMENT[language].search(text)
    if m and m.group(3) != '18':
        s, e = span(m)
        findings.append(_finding(sent, s, e, 'discriminatory', 'age', 'age_requirement',
                                 FINDING_SCORES['age_requirement'], 'Age limit used as a requirement.',
                                 'Describe the competences needed instead of an age.' if language == 'english'
                                 else 'Describir las competencias requeridas en lugar de la edad.'))
    for pat in (_AGE_RANGE[language], _YOUTH_REQUIREMENT[language]):
        for m in pat.finditer(text):
            s, e = span(m)
            findings.append(_finding(sent, s, e, 'discriminatory', 'age', 'age_requirement',
                                     FINDING_SCORES['age_requirement'], 'Age used as a requirement.', None))
    for pat in _APPEARANCE_PATTERNS[language]:
        for m in pat.finditer(text):
            s, e = span(m)
            findings.append(_finding(sent, s, e, 'discriminatory', 'appearance', 'appearance_requirement',
                                     FINDING_SCORES['appearance_requirement'],
                                     'Physical appearance used to describe or select a person.',
                                     'Remove the reference to appearance.' if language == 'english'
                                     else 'Eliminar la referencia al aspecto físico.'))

    # --- discriminatory terms (problematic in themselves) ---
    for pat, (category, score, suggestion) in _DISCRIM_TERM_RE[language]:
        for m in pat.finditer(text):
            s, e = span(m)
            findings.append(_finding(sent, s, e, 'discriminatory', category, 'outdated_term', score,
                                     'Outdated or pejorative term.', suggestion))

    # --- negative generalisations about social groups ---
    negative = [m for p in _NEGATIVE_RE[language] for m in p.finditer(text)]
    group_mentions: Dict[str, List[str]] = {}
    for cat in SOCIAL_GROUPS[language]:
        found = [m.group(0) for m in _GROUP_MENTION_RE[language][cat].finditer(text)]
        if found:
            group_mentions[cat] = found
        g = _GENERALISATION_RE[language][cat].search(text)
        if g and negative:
            s, e = span(g)
            e = max(e, base + max(n.end() for n in negative))
            findings.append(_finding(sent, s, e, 'discriminatory', cat, 'negative_generalisation',
                                     FINDING_SCORES['negative_generalisation'],
                                     'Negative generalisation about a social group.', None))

    # --- gender stereotypes ---
    refs = _gendered_tokens(sent, language)
    stereo = STEREOTYPICAL_ASSOCIATIONS[language]
    traits = (_token_matches(sent, stereo['male_coded'] + stereo['female_coded'], pos={'ADJ', 'NOUN', 'VERB'})
              + _phrase_matches(text, stereo['male_coded'] + stereo['female_coded']))
    # a token that is itself a gendered reference is not also a trait
    ref_ids = {t.i for t in refs['male'] + refs['female']}
    traits = [t for t in traits if not hasattr(t, 'i') or t.i not in ref_ids]
    markers = [m for p in _MARKER_RE[language] for m in p.finditer(text)]
    generalisation = _GENERALISATION_RE[language]['gender'].search(text)
    has_ref = bool(refs['male'] or refs['female'])
    explicit_here = any(f['type'] == 'explicit' for f in findings)

    score = 0.0
    if has_ref and not explicit_here:
        if (generalisation or markers) and (traits or markers):
            score = FINDING_SCORES['stereotype_strong']
        elif len(traits) >= 2:
            score = FINDING_SCORES['stereotype_strong']
        elif traits:
            score = FINDING_SCORES['stereotype_weak']
        elif generalisation:
            score = FINDING_SCORES['generalisation']

    lower = text.lower()
    positive = [p for p in POSITIVE_CONTEXT_PHRASES[language] if p in lower]
    if score and positive:
        score *= POSITIVE_CONTEXT_DISCOUNT
    if score:
        findings.append(_finding(sent, sent.start_char, sent.end_char, 'stereotype', 'gender',
                                 'gender_stereotype', score,
                                 'Gender stereotype: a trait or ability attributed to a gender.', None))

    # --- non-inclusive language ---
    for pat, (subtype, suggestion) in _NON_INCLUSIVE_RE[language]:
        for m in pat.finditer(text):
            s, e = span(m)
            findings.append(_finding(sent, s, e, 'non_inclusive', 'gender', subtype,
                                     FINDING_SCORES['non_inclusive'],
                                     {'androcentric': 'Masculine form used to mean all people.',
                                      'gendered_term': 'Gendered job title or term with a neutral alternative.',
                                      'asymmetric_treatment': 'Courtesy title applied only to women.'}[subtype],
                                     suggestion))
    if language == 'spanish':
        findings += _detect_generic_masculine_es(sent)
    else:
        m = _GENERIC_HE.search(text)
        if m and not re.search(r'\b(she|her|hers)\b', text, re.I) \
                and not any(ent.label_ == 'PERSON' for ent in sent.ents):
            s, e = span(m)
            findings.append(_finding(sent, s, e, 'non_inclusive', 'gender', 'generic_he',
                                     FINDING_SCORES['non_inclusive'],
                                     "Generic 'he' for a person of unspecified gender.",
                                     'they / their, or rephrase in the plural'))

    return {
        'findings': findings,
        'male_refs': [t.text.lower() for t in refs['male']],
        'female_refs': [t.text.lower() for t in refs['female']],
        'male_traits': [t.text.lower() if hasattr(t, 'text') else t.group(0).lower()
                        for t in traits if _is_coded(t, stereo['male_coded'])],
        'female_traits': [t.text.lower() if hasattr(t, 'text') else t.group(0).lower()
                          for t in traits if _is_coded(t, stereo['female_coded'])],
        'positive': positive,
        'group_mentions': group_mentions,
    }


def _is_coded(t, words) -> bool:
    if hasattr(t, 'lemma_'):
        return t.lower_ in words or t.lemma_.lower() in words
    return t.group(0).lower() in words


@lru_cache(maxsize=128)
def analyse_text(text: str, language: str = 'english') -> Dict:
    """Sentence-level analysis of *text* (cached).

    Returns ``{'findings': [...], 'sentences': [...], ...}`` where every
    finding has ``type, category, subtype, match, start, end, sentence,
    score, explanation, suggestion``.
    """
    if language not in BIAS_PATTERNS:
        language = 'english'
    doc = get_nlp(language)(text)
    agg = {'findings': [], 'male_refs': [], 'female_refs': [], 'male_traits': [],
           'female_traits': [], 'positive': [], 'group_mentions': {}, 'sentences': []}
    for sent in doc.sents:
        if not sent.text.strip():
            continue
        res = _analyse_sentence(sent, language)
        agg['sentences'].append({'text': sent.text.strip(), 'start': sent.start_char, 'end': sent.end_char})
        for key in ('findings', 'male_refs', 'female_refs', 'male_traits', 'female_traits', 'positive'):
            agg[key].extend(res[key])
        for cat, found in res['group_mentions'].items():
            agg['group_mentions'].setdefault(cat, []).extend(found)
    agg['findings'] = _dedupe(agg['findings'])
    return agg


def _dedupe(findings: List[Dict]) -> List[Dict]:
    """Drop findings fully contained in a stronger finding of the same type."""
    findings = sorted(findings, key=lambda f: (-f['score'], f['start']))
    kept: List[Dict] = []
    for f in findings:
        if any(k['type'] == f['type'] and k['category'] == f['category']
               and k['start'] <= f['start'] and f['end'] <= k['end'] for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda f: f['start'])


def collect_findings(text: str, language: str = 'english') -> List[Dict]:
    """All findings for *text*, sorted by position."""
    return list(analyse_text(text, language)['findings'])


# ---------------------------------------------------------------------------
# 9.  PAGE-LEVEL DETECTION FUNCTIONS (public interface used by detectors)
# ---------------------------------------------------------------------------

def detect_gender_bias(text: str, language: str = 'english',
                       bias_threshold: float = 0.6) -> Dict:
    """Keyword- and pattern-based gender bias detection for a page.

    ``confidence`` is the page gender score: strong evidence (explicit
    patterns, stereotypes) combined by noisy-OR, plus non-inclusive language
    capped at ``NON_INCLUSIVE_PAGE_CAP``.  The male/female reference ratio is
    reported for context only and does not drive the score.
    """
    a = analyse_text(text, language)
    gender = [f for f in a['findings'] if f['category'] == 'gender']
    strong = noisy_or([f['score'] for f in gender if f['type'] in ('explicit', 'stereotype')])
    inclusive = min(NON_INCLUSIVE_PAGE_CAP,
                    noisy_or([f['score'] for f in gender if f['type'] == 'non_inclusive']))
    score = noisy_or([strong, inclusive])

    male = a['male_refs'] + a['male_traits']
    female = a['female_refs'] + a['female_traits']
    total = len(male) + len(female)
    male_ratio = len(male) / total if total else 0.0
    female_ratio = len(female) / total if total else 0.0
    if not total:
        direction = 'neutral'
    elif max(male_ratio, female_ratio) <= bias_threshold:
        direction = 'balanced'
    else:
        direction = 'male' if male_ratio > female_ratio else 'female'

    return {
        'has_gender_bias': score > 0,
        'male_bias_score': round(male_ratio, 4),
        'female_bias_score': round(female_ratio, 4),
        'bias_direction': direction,
        'confidence': round(score, 4),
        'male_keywords_found': sorted(set(male)),
        'female_keywords_found': sorted(set(female)),
        'bias_patterns_matched': [f['match'] for f in gender if f['type'] == 'explicit'],
        'non_inclusive_found': [f['match'] for f in gender if f['type'] == 'non_inclusive'],
        'positive_context_found': sorted(set(a['positive'])),
        'findings': gender,
        'severity': score_to_severity(score),
    }


def detect_discriminatory_language(text: str,
                                   language: str = 'english') -> Dict:
    """Discriminatory language by category (age, race, disability, appearance).

    ``keywords_found`` lists only *problematic* matches; neutral mentions of
    a group are listed under ``mentions`` and do not score.
    """
    a = analyse_text(text, language)
    results = {}
    for category in DISCRIMINATION_CATEGORIES:
        found = [f for f in a['findings'] if f['type'] == 'discriminatory' and f['category'] == category]
        score = noisy_or([f['score'] for f in found])
        results[category] = {
            'has_discriminatory_language': bool(found),
            'keywords_found': [f['match'] for f in found],
            'count': len(found),
            'score': round(score, 4),
            'mentions': a['group_mentions'].get(category, []),
            'findings': found,
        }
    return results


def discriminatory_score(discriminatory: Dict) -> float:
    """Overall 0-1 score from the output of ``detect_discriminatory_language``."""
    return noisy_or([v.get('score', 0.0) for v in discriminatory.values()])
