#!/usr/bin/env python3
"""
Setup and verification script for the Text Mining and Bias Detection Pipeline.
Ensures all requirements are met and models are downloaded.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Your version:", sys.version)
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
        'nltk', 'textblob', 'spacy', 'transformers', 'torch', 'tensorflow',
        'bertopic', 'umap', 'hdbscan', 'sentence-transformers'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False
    
    print(f"✓ All required packages are installed")
    return True


def download_nltk_data():
    """Download required NLTK data."""
    print("\nDownloading NLTK data...")
    import nltk
    
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    for path, name in required_data:
        try:
            nltk.data.find(path)
            print(f"  ✓ {name} already downloaded")
        except LookupError:
            print(f"  Downloading {name}...")
            nltk.download(name)
            print(f"  ✓ {name} downloaded")


def download_spacy_model():
    """Download spaCy English model."""
    print("\nDownloading spaCy model...")
    try:
        import spacy
        spacy.load('en_core_web_sm')
        print("  ✓ Spacy model already downloaded")
    except OSError:
        print("  Downloading en_core_web_sm...")
        subprocess.check_call([
            sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
        ])
        print("  ✓ Spacy model downloaded")


def verify_project_structure():
    """Verify project structure is complete."""
    print("\nVerifying project structure...")
    
    required_dirs = [
        'config', 'utils', 'analysis', 'models', 'pipelines', 
        'notebooks', 'data', 'output'
    ]
    
    required_files = [
        'config/settings.py',
        'utils/preprocessing.py',
        'utils/analysis_utils.py',
        'analysis/bias_detector.py',
        'analysis/topic_modeler.py',
        'models/clustering.py',
        'models/classification.py',
        'models/decision_engine.py',
        'pipelines/text_mining_pipeline.py',
        'notebooks/text_mining_analysis.ipynb',
        'example_analysis.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'PROJECT_DOCUMENTATION.md'
    ]
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/ exists")
        else:
            print(f"  ❌ {dir_name}/ missing")
    
    # Check files
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"  ✓ {file_name} exists")
        else:
            print(f"  ❌ {file_name} missing")


def test_imports():
    """Test that key modules can be imported."""
    print("\nTesting module imports...")
    
    modules_to_test = [
        'config.settings',
        'utils.preprocessing',
        'utils.analysis_utils',
        'analysis.bias_detector',
        'analysis.topic_modeler',
        'models.clustering',
        'models.classification',
        'models.decision_engine',
        'pipelines.text_mining_pipeline'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module} imported successfully")
        except Exception as e:
            print(f"  ❌ {module} import failed: {e}")


def print_quickstart():
    """Print quick start instructions."""
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nQuick Start:")
    print("  1. Run example: python example_analysis.py")
    print("  2. Open notebook: jupyter notebook notebooks/text_mining_analysis.ipynb")
    print("  3. Read quick start: cat QUICKSTART.md")
    print("\nDocumentation:")
    print("  - Quick Start: QUICKSTART.md")
    print("  - Full Docs: PROJECT_DOCUMENTATION.md")
    print("  - Example Code: example_analysis.py")
    print("="*70 + "\n")


def main():
    """Run all verification checks."""
    print("="*70)
    print("Text Mining and Bias Detection Pipeline - Setup Verification")
    print("="*70 + "\n")
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", verify_project_structure),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"⚠️  {check_name} check error: {e}")
    
    # Download additional resources
    try:
        download_nltk_data()
        download_spacy_model()
    except Exception as e:
        print(f"⚠️  Warning during resource download: {e}")
    
    # Test imports
    test_imports()
    
    # Print results
    if all_passed:
        print_quickstart()
        return 0
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
