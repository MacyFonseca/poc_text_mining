"""
Example script demonstrating the complete text mining pipeline.
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.text_mining_pipeline import TextMiningPipeline
from config.settings import PipelineConfig


def create_sample_data():
    """Create sample research project descriptions for demonstration."""
    sample_texts = [
        # Example 1: Potentially biased text
        "We are looking for a strong male engineer to lead our team. "
        "The ideal candidate should be aggressive in pursuing market opportunities. "
        "We need a young leader with ambitious goals to drive our company forward.",
        
        # Example 2: Inclusive text
        "We are seeking a talented software engineer with strong problem-solving skills. "
        "The ideal candidate will have experience in team leadership and collaboration. "
        "We welcome applications from diverse backgrounds and perspectives.",
        
        # Example 3: Potentially discriminatory
        "We need a beautiful, nurturing woman to handle customer relations. "
        "The position requires someone who is emotionally intelligent and care-oriented. "
        "Previous experience in supportive roles is preferred.",
        
        # Example 4: Professional, unbiased
        "We are recruiting for a technical project manager position. "
        "Required skills include project planning, resource management, and communication. "
        "We are committed to building a diverse and inclusive team.",
        
        # Example 5: Another inclusive example
        "Join our research team as a data scientist. We value innovation and collaboration. "
        "We welcome candidates of all backgrounds to apply. "
        "Equal opportunities for career growth and development.",
        
        # Example 6: Subtle gender bias
        "Experienced male programmer needed for leadership position. "
        "Must be logical, aggressive in negotiations, and ambitious. "
        "Previous experience managing teams of 10+ people required.",
        
        # Example 7: Age bias
        "We're looking for young, energetic professionals (age 25-35) to join our startup. "
        "Digital natives preferred. Must be quick learners with modern approaches. "
        "Avoid candidates with elderly values.",
        
        # Example 8: Inclusive job posting
        "Position: Senior Analyst | We seek candidates with analytical expertise. "
        "Qualifications: Strong analytical skills, attention to detail, problem-solving ability. "
        "We are an equal opportunity employer and value diversity.",
    ]
    
    return sample_texts


def main():
    """Run the complete text mining pipeline with sample data."""
    print("\n" + "=" * 70)
    print("TEXT MINING AND BIAS DETECTION PIPELINE - EXAMPLE")
    print("=" * 70 + "\n")
    
    # Load sample data
    print("Loading sample research project descriptions...")
    documents = create_sample_data()
    print(f"Loaded {len(documents)} sample documents\n")
    
    # Initialize pipeline with default configuration
    config = PipelineConfig()
    pipeline = TextMiningPipeline(config)
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(documents)
    
    # Generate and print report
    print(pipeline.generate_report())
    
    # Print summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    summary = pipeline.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export results
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, 'analysis_results.json')
    txt_path = os.path.join(output_dir, 'analysis_report.txt')
    
    pipeline.export_results(json_path, format='json')
    pipeline.export_results(txt_path, format='txt')
    
    print(f"\nResults saved to {output_dir}/")
    
    return pipeline


def analyze_single_text(text: str):
    """Analyze a single text for bias."""
    print("\n" + "=" * 70)
    print("SINGLE TEXT BIAS ANALYSIS")
    print("=" * 70 + "\n")
    
    # Initialize just bias detection
    from analysis.bias_detector import BiasDetector
    
    detector = BiasDetector()
    analysis = detector.comprehensive_bias_analysis(text)
    
    print(detector.generate_bias_report(analysis))
    print(f"\nOverall Bias Score: {analysis['overall_bias_score']:.2%}")
    print(f"Is Biased: {analysis['is_biased']}")


def interactive_analysis():
    """Interactive mode for analyzing custom text."""
    print("\n" + "=" * 70)
    print("INTERACTIVE TEXT ANALYSIS MODE")
    print("=" * 70)
    print("\nEnter text to analyze (type 'quit' to exit):\n")
    
    from analysis.bias_detector import BiasDetector
    
    detector = BiasDetector()
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() == 'quit':
            print("Exiting...")
            break
        
        if not text:
            continue
        
        analysis = detector.comprehensive_bias_analysis(text)
        print(detector.generate_bias_report(analysis))


if __name__ == "__main__":
    # Example 1: Run full pipeline with sample data
    pipeline = main()
    
    # Example 2: Analyze a single custom text
    custom_text = ("We are looking for a talented individual with strong communication skills. "
                   "We welcome applications from all qualified candidates.")
    analyze_single_text(custom_text)
    
    # Example 3: Interactive analysis (uncomment to use)
    # interactive_analysis()
