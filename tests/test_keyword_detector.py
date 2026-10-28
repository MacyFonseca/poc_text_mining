"""Regression and contrast tests for the keyword detector (ROADMAP P0.3).

Each test pins a behaviour that was wrong in the original proof of concept
or that a later change must not break.  Run with:  python -m pytest tests/
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.bias_detector import BiasDetector
from analysis.bias_keywords import collect_findings


@pytest.fixture(scope='module')
def es():
    return BiasDetector(language='spanish')


@pytest.fixture(scope='module')
def en():
    return BiasDetector(language='english')


def subtypes(text, language):
    return {f['subtype'] for f in collect_findings(text, language)}


# --- the four probe sentences from the diagnosis ---------------------------

def test_neutral_group_mention_is_not_biased(es):
    r = es.comprehensive_bias_analysis(
        'El estudio analiza la integración laboral de la población inmigrante en Canarias.')
    assert not r['is_biased']
    assert r['discriminatory_language']['race']['mentions']  # still recorded


def test_inflected_spanish_stereotype_is_detected(es):
    r = es.comprehensive_bias_analysis('Las enfermeras son cariñosas y las ingenieras agresivas.')
    assert r['is_biased']
    assert 'gender_stereotype' in subtypes(r['text'], 'spanish')


def test_generic_masculine_and_androcentric_are_detected(es):
    text = 'Los investigadores y los directores del proyecto serán los responsables. El hombre ha transformado su entorno.'
    found = subtypes(text, 'spanish')
    assert {'generic_masculine', 'androcentric'} <= found
    assert es.comprehensive_bias_analysis(text)['is_biased']


def test_positive_context_does_not_hide_explicit_bias(en):
    r = en.comprehensive_bias_analysis(
        'Women should stay at home. This project promotes gender equality and inclusion.')
    assert r['is_biased']
    assert r['severity'] in ('medium', 'high', 'critical')


# --- contrast pairs: biased vs. neutral wording of the same content ----------

CONTRAST_PAIRS = [
    ('spanish', 'Los investigadores presentarán un informe anual.',
                'El personal investigador presentará un informe anual.'),
    ('spanish', 'Todos los alumnos firmarán el consentimiento.',
                'El alumnado firmará el consentimiento.'),
    ('spanish', 'Los participantes deberán completar el cuestionario.',
                'Las personas participantes deberán completar el cuestionario.'),
    ('spanish', 'El hombre ha transformado su entorno.',
                'La humanidad ha transformado su entorno.'),
    ('spanish', 'Los minusválidos accederán por la rampa.',
                'Las personas con discapacidad accederán por la rampa.'),
    ('english', 'Each researcher must submit his report.',
                'Each researcher must submit their report.'),
    ('english', 'The chairman will open the session.',
                'The chair will open the session.'),
    ('english', 'Mankind has always looked at the stars.',
                'Humankind has always looked at the stars.'),
    ('english', 'Handicapped participants will use the rear entrance.',
                'Participants with disabilities will use the step-free entrance.'),
]


@pytest.mark.parametrize('language,biased,neutral', CONTRAST_PAIRS)
def test_contrast_pair(language, biased, neutral):
    d = BiasDetector(language=language)
    assert d.comprehensive_bias_analysis(biased)['is_biased'], biased
    assert not d.comprehensive_bias_analysis(neutral)['is_biased'], neutral


# --- hard negatives: text that discusses gender or groups neutrally ---------

HARD_NEGATIVES = [
    ('spanish', 'Investigadoras e investigadores colaborarán en la recogida de datos.'),
    ('spanish', 'Los investigadores/as colaborarán en la recogida de datos.'),
    ('spanish', 'Las personas participantes deberán ser mayores de 18 años.'),
    ('spanish', 'El estudio examina los estereotipos que asocian a las mujeres con los cuidados.'),
    ('spanish', 'La enfermera responsable del ensayo es la doctora López.'),
    ('spanish', 'Se garantiza la accesibilidad para personas usuarias de silla de ruedas.'),
    ('english', 'Participants must be over 18 years old to give informed consent.'),
    ('english', 'We recruited 20 participants with a disability.'),
    ('english', 'Women remain underrepresented in senior academic positions.'),
    ('english', 'He presented the results at the 2024 annual meeting.'),
    ('english', 'We analyse barriers faced by female engineers in industry.'),
]


@pytest.mark.parametrize('language,text', HARD_NEGATIVES)
def test_hard_negative(language, text):
    assert not BiasDetector(language=language).comprehensive_bias_analysis(text)['is_biased'], text


# --- counterfactual gender swap: score must not depend on which gender -------

SWAPS = [
    ('english', 'Women are naturally worse at leading teams.', 'Men are naturally worse at leading teams.'),
    ('spanish', 'Las mujeres son más emocionales por naturaleza.', 'Los hombres son más emocionales por naturaleza.'),
    ('english', 'We are looking for a male engineer.', 'We are looking for a female engineer.'),
]


@pytest.mark.parametrize('language,a,b', SWAPS)
def test_gender_swap_symmetry(language, a, b):
    d = BiasDetector(language=language)
    ra, rb = d.comprehensive_bias_analysis(a), d.comprehensive_bias_analysis(b)
    assert ra['is_biased'] and rb['is_biased']
    assert abs(ra['overall_bias_score'] - rb['overall_bias_score']) < 0.05


# --- interface contract shared by all detectors -----------------------------

def test_interface_keys(en):
    r = en.comprehensive_bias_analysis('The chairman opened the meeting.')
    for key in ('overall_bias_score', 'is_biased', 'severity', 'gender_bias',
                'discriminatory_language', 'findings'):
        assert key in r
    f = r['findings'][0]
    assert r['text'][f['start']:f['end']] == f['match']
    assert f['suggestion']


def test_findings_offsets_on_multi_sentence_page(es):
    text = 'Introducción del proyecto. Los becarios recogerán los datos. El presupuesto es suficiente.'
    for f in collect_findings(text, 'spanish'):
        assert text[f['start']:f['end']] == f['match']
