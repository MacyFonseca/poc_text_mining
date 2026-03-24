"""
Example script demonstrating ML-based bias detection using a fine-tuned model
(d4data/bias-detection-model) combined with zero-shot classification for
bias categorization.

Uses MLBiasDetector from analysis.ml_bias_detector.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.ml_bias_detector import MLBiasDetector


def create_sample_data():
    """Create sample job posting descriptions for demonstration."""
    sample_texts = [
        # Example 1: Male-biased
        "We are looking for a strong male engineer to lead our team. "
        "The ideal candidate should be aggressive in pursuing market opportunities. "
        "We need a young leader with ambitious goals to drive our company forward.",

        # Example 2: Inclusive
        "We are seeking a talented software engineer with strong problem-solving skills. "
        "The ideal candidate will have experience in team leadership and collaboration. "
        "We welcome applications from diverse backgrounds and perspectives.",

        # Example 3: Female-biased, discriminatory
        "We need a beautiful, nurturing woman to handle customer relations. "
        "The position requires someone who is emotionally intelligent and care-oriented. "
        "Previous experience in supportive roles is preferred.",

        # Example 4: Professional, unbiased
        "We are recruiting for a technical project manager position. "
        "Required skills include project planning, resource management, and communication. "
        "We are committed to building a diverse and inclusive team.",

        # Example 5: Inclusive research team
        "Join our research team as a data scientist. We value innovation and collaboration. "
        "We welcome candidates of all backgrounds to apply. "
        "Equal opportunities for career growth and development.",

        # Example 6: Male-biased, leadership
        "Experienced male programmer needed for leadership position. "
        "Must be logical, aggressive in negotiations, and ambitious. "
        "Previous experience managing teams required.",

        # Example 7: Age bias
        "We're looking for young, energetic professionals (age 25-35) to join our startup. "
        "Digital natives preferred. Must be quick learners with modern approaches. "
        "Avoid candidates with outdated skills.",

        # Example 8: Inclusive job posting
        "Position: Senior Analyst. We seek candidates with analytical expertise. "
        "Qualifications: Strong analytical skills, attention to detail, problem-solving ability. "
        "We are an equal opportunity employer and value diversity.",

        # Example 9: Male-biased management
        "Hiring experienced male manager for leadership development program. "
        "Must have strong decision-making skills and competitive mindset. "
        "Previous experience managing large teams strongly preferred.",

        # Example 10: Inclusive operations
        "Operations Manager needed for growing technology company. "
        "Must have strong organizational and communication abilities. "
        "We value diverse perspectives and inclusive leadership practices.",

        # Example 11: Discriminatory
        "We need a young, energetic team. Must be flexible and adaptable. "
        "Preference for candidates under 35 with startup mentality.",

        # Example 12: Inclusive
        "Senior Software Developer position. Skills: Python, SQL, cloud technologies. "
        "All qualified candidates welcome. We provide mentorship and growth opportunities.",

        # Example 13: Female-biased
        "Receptionist position for our office. We prefer friendly, helpful female candidates. "
        "Must be detail-oriented and excellent at managing relationships.",

        # Example 14: Neutral
        "Systems Administrator role. Requirements: Linux, networking, troubleshooting. "
        "Competitive salary and benefits. Equal opportunity employer.",

        # Example 15: Male-biased technical
        "Senior Software Architect needed. Must be assertive, detail-oriented, and ambitious. "
        "Experience leading large engineering teams essential.",

        # Example 16: Inclusive technical
        "Software Engineer wanted for innovative project. You will work on challenging problems. "
        "We support continuous learning and career development for all candidates.",

        # Example 17: Potentially biased
        "Marketing Manager position. Must be charming and persuasive. "
        "Preferred candidate: successful salesperson, proven track record.",

        # Example 18: Inclusive marketing
        "Marketing Manager - we seek creative candidates with strong analytical skills. "
        "Diversity and inclusion are core values. We welcome applications from all qualified candidates.",

        # Example 19: Age/disability bias
        "Need energetic team without health issues or physical limitations. "
        "Fast-paced environment requires quick reflexes and perfect health.",

        # Example 20: Inclusive diverse
        "Research Assistant position available. Background requirements: college degree. "
        "We actively encourage applications from underrepresented groups in research.",
    ]

    return sample_texts


def format_result(analysis: dict) -> str:
    """Format a single comprehensive analysis result for display."""
    lines = []

    # Fine-tuned model result
    ml = analysis['ml_detection']
    lines.append(f"  Fine-tuned model : {ml['label']} (confidence: {ml['confidence']:.4f})")

    # Zero-shot categorization
    cat = analysis['ml_categorization']
    lines.append(f"  Bias category    : {cat['top_category']} (score: {cat['top_score']:.4f})")
    lines.append("  Category scores:")
    for label, score in zip(cat['labels'], cat['scores']):
        bar = "#" * int(score * 40)
        lines.append(f"    {label:<25s} {score:.4f}  {bar}")

    # Gender bias keywords
    gb = analysis['gender_bias']
    lines.append(f"  Gender direction : {gb['bias_direction']}")
    if gb['male_keywords_found']:
        lines.append(f"  Male keywords    : {', '.join(gb['male_keywords_found'])}")
    if gb['female_keywords_found']:
        lines.append(f"  Female keywords  : {', '.join(gb['female_keywords_found'])}")

    # Overall
    lines.append(f"  Overall score    : {analysis['overall_bias_score']:.4f}")

    return "\n".join(lines)


def main():
    """Run ML-based bias detection on all sample texts."""
    print("\n" + "=" * 70)
    print("ML BIAS DETECTION (Fine-Tuned + Zero-Shot)")
    print("Binary model : valurank/distilroberta-bias")
    print("Category model: facebook/bart-large-mnli")
    print("=" * 70 + "\n")

    documents = create_sample_data()
    print(f"Loaded {len(documents)} sample documents\n")

    print("Initializing MLBiasDetector (downloads models on first run)...\n")
    detector = MLBiasDetector(device="cpu", language="english")

    biased_count = 0
    results_all = []

    for idx, text in enumerate(documents, start=1):
        print("-" * 70)
        print(f"Document {idx}/{len(documents)}")
        print(f"Text: {text[:120]}...")

        analysis = detector.comprehensive_bias_analysis(text)
        results_all.append(analysis)

        print(format_result(analysis))

        if analysis['is_biased']:
            biased_count += 1
            cat = analysis['ml_categorization']['top_category']
            print(f"  >> BIAS DETECTED: {cat}")
        else:
            print("  >> No significant bias detected")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total documents analysed : {len(documents)}")
    print(f"Documents flagged biased : {biased_count}")
    print(f"Documents deemed neutral : {len(documents) - biased_count}")
    print(f"Bias rate                : {biased_count / len(documents):.1%}")

    # Export results
    output_dir = os.path.join(os.path.dirname(__file__), 'output_ml')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'ml_finetuned_bias_results.json')

    export_data = []
    for text, analysis in zip(documents, results_all):
        export_data.append({
            'text': text,
            'is_biased': analysis['is_biased'],
            'overall_bias_score': analysis['overall_bias_score'],
            'ml_label': analysis['ml_detection']['label'],
            'ml_confidence': analysis['ml_detection']['confidence'],
            'bias_category': analysis['ml_categorization']['top_category'],
            'category_score': analysis['ml_categorization']['top_score'],
            'all_categories': dict(zip(
                analysis['ml_categorization']['labels'],
                analysis['ml_categorization']['scores']
            )),
            'gender_direction': analysis['gender_bias']['bias_direction'],
            'male_keywords': analysis['gender_bias']['male_keywords_found'],
            'female_keywords': analysis['gender_bias']['female_keywords_found'],
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_path}")

    return results_all


if __name__ == "__main__":
    main()
