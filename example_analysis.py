"""
Example script demonstrating the complete text mining pipeline.
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.text_mining_pipeline import TextMiningPipeline
from config.settings import (
    PipelineConfig, TopicModelingConfig, ClusteringConfig
)


def create_sample_data():
    """Create sample research project descriptions for demonstration."""
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


def main():
    """Run the complete text mining pipeline with sample data."""
    print("\n" + "=" * 70)
    print("TEXT MINING AND BIAS DETECTION PIPELINE - EXAMPLE")
    print("=" * 70 + "\n")
    
    # Load sample data
    print("Loading sample research project descriptions...")
    documents = create_sample_data()
    print(f"Loaded {len(documents)} sample documents\n")
    
    # Adjust parameters for small datasets
    n_docs = len(documents)
    umap_neighbors = min(15, max(3, n_docs - 5))  # Reduce for small datasets
    hdbscan_min_size = min(10, max(2, n_docs // 5))  # Scale down for small datasets
    
    config = PipelineConfig(
        topic_modeling=TopicModelingConfig(
            min_topic_size=5,
            nr_topics=None,  # Auto-detect
            umap_n_neighbors=umap_neighbors,  # Adaptive
            hdbscan_min_cluster_size=hdbscan_min_size
        ),
        clustering=ClusteringConfig(
            n_clusters=min(5, max(2, n_docs // 6)),  # Adaptive cluster count
        )
    )
    
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
