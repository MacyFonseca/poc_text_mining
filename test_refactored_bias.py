#!/usr/bin/env python
"""Test script for refactored context-aware bias detector."""

from analysis.bias_detector import BiasDetector

def display_gender_analysis(gender_bias_data):
    """Display gender bias analysis in a formatted way."""
    print("  ║ GENDER BIAS ANALYSIS:")
    print("  ╠════════════════════════════════════")
    
    if gender_bias_data['has_gender_bias']:
        print(f"  ║  Status: ✓ DETECTED (Direction: {gender_bias_data['bias_direction']})")
        print(f"  ║  Score: {gender_bias_data['bias_score']:.2%}")
        
        if gender_bias_data['male_evidence']:
            print(f"  ║  Male Evidence ({len(gender_bias_data['male_evidence'])} found):")
            for evidence in gender_bias_data['male_evidence'][:2]:  # Show top 2
                print(f"  ║    • {evidence[:60]}...")
        
        if gender_bias_data['female_evidence']:
            print(f"  ║  Female Evidence ({len(gender_bias_data['female_evidence'])} found):")
            for evidence in gender_bias_data['female_evidence'][:2]:  # Show top 2
                print(f"  ║    • {evidence[:60]}...")
    else:
        print(f"  ║  Status: ✗ NOT DETECTED")
    
    print("  ╚════════════════════════════════════\n")

def display_discriminatory_analysis(discriminatory_data):
    """Display discriminatory language analysis in a formatted way."""
    print("  ║ DISCRIMINATORY LANGUAGE ANALYSIS:")
    print("  ╠════════════════════════════════════")
    
    if not discriminatory_data:
        print("  ║  Status: ✗ NOT DETECTED")
        print("  ╚════════════════════════════════════\n")
        return
    
    detected_any = False
    for category, details in discriminatory_data.items():
        if details['has_discriminatory_language']:
            detected_any = True
            print(f"  ║  ✓ {category.upper()}")
            print(f"  ║    Confidence: {details['confidence_score']:.2%}")
            print(f"  ║    Match Count: {details['match_count']}")
            
            if details['evidence_windows']:
                print(f"  ║    Evidence ({len(details['evidence_windows'])} found):")
                for evidence in details['evidence_windows'][:2]:  # Show top 2
                    print(f"  ║      • {evidence[:55]}...")
            print()
    
    if not detected_any:
        print("  ║  Status: ✗ NO DISCRIMINATORY LANGUAGE DETECTED")
    
    print("  ╚════════════════════════════════════\n")

def test_bias_detector():
    """Test the context-aware bias detector."""
    print("=" * 70)
    print("TESTING CONTEXT-AWARE BIAS DETECTOR")
    print("=" * 70)
    
    # Initialize detector with CPU for faster testing
    print("\n1. Initializing context-aware bias detector...")
    try:
        detector = BiasDetector(device='cpu')
        print("✓ Bias detector initialized successfully\n")
    except Exception as e:
        print(f"✗ Error initializing detector: {e}")
        return
    
    # Test cases with various types of bias
    test_cases = [
        {
            'text': 'The male engineer led the project while the female nurse provided support.',
            'description': 'Gender stereotypes (male engineer, female nurse)'
        },
        {
            'text': 'We should not hire disabled people as they cannot perform demanding tasks.',
            'description': 'Disability discrimination'
        },
        {
            'text': 'The old workers were slow compared to young employees who worked faster.',
            'description': 'Age discrimination'
        },
        {
            'text': 'We only want to hire local people, not foreigners.',
            'description': 'Race/ethnicity discrimination'
        },
        {
            'text': 'The engineer led the project while the team member provided support.',
            'description': 'No discriminatory language'
        },
    ]
    
    print("2. Running analysis on test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"╔{'='*68}╗")
        print(f"║ Test Case {i}: {test_case['description']:<51}║")
        print(f"║ Text: {test_case['text']:<58}║")
        print(f"╚{'='*68}╝\n")
        
        try:
            analysis = detector.comprehensive_bias_analysis(test_case['text'])
            
            # Overall summary
            print(f"  OVERALL BIAS SCORE: {analysis['overall_bias_score']:.2%}")
            print(f"  Is Biased: {'YES ✓' if analysis['is_biased'] else 'NO ✗'}\n")
            
            # Display gender bias analysis
            if 'gender_bias' in analysis:
                display_gender_analysis(analysis['gender_bias'])
            
            # Display discriminatory language analysis
            if 'discriminatory_language' in analysis:
                display_discriminatory_analysis(analysis['discriminatory_language'])
            
            print()
        
        except Exception as e:
            print(f"  ✗ Error during analysis: {e}\n")
    
    # print("=" * 70)
    # print("3. Key Improvements from Refactoring:")
    # print("=" * 70)
    # print("✓ Context-aware detection using semantic similarity")
    # print("✓ No longer relies on pre-established keyword lists")
    # print("✓ Understands surrounding context via context windows")
    # print("✓ Provides evidence windows showing where bias was detected")
    # print("✓ More robust to false positives from innocent word usage")
    # print("✓ Uses transformer models for better semantic understanding")
    # print("=" * 70)

if __name__ == '__main__':
    test_bias_detector()
