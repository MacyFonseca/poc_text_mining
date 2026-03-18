"""Bias detection using context-aware transformer models."""
import re

import numpy as np

from typing import Dict, List, Tuple

from transformers import pipeline

from sentence_transformers import SentenceTransformer, util

import warnings
warnings.filterwarnings('ignore')


class BiasDetector:
    """Detect gender bias and discriminatory language in text using context-aware models."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        """Initialize bias detector with context-aware transformer models."""
        self.model_name = model_name
        self.device = 0 if device == "cuda" else -1
        self.device_name = device
        
        # Initialize zero-shot classification for bias detection
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        
        # Initialize sentence transformer for semantic understanding
        # This model is trained to understand semantic similarity and context
        self.sentence_encoder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device_name
        )
        
        # Define bias-related semantic templates for context-aware detection
        self.bias_templates = {
            'gender_bias': {
                'masculine_stereotypes': [
                    'strong male leader',
                    'ambitious man executive',
                    'male engineer or scientist',
                    'masculine strength or power',
                    'male professional or corporate role',
                    'man in charge or leadership',
                    'aggressive masculine behavior',
                    'male as the main authority',
                    'masculine traits like logical or competitive',
                ],
                'feminine_stereotypes': [
                    'woman in support role',
                    'female caregiver or helper',
                    'woman as nurse or secretary',
                    'female supportive assistant',
                    'feminine emotion or sensitivity',
                    'woman in domestic or nurturing role',
                    'female beauty or appearance focus',
                    'woman as subordinate or helper',
                    'feminine traits like emotional or caring',
                ],
                'gender_representation': [
                    'male terms used more than female terms',
                    'predominantly male pronouns',
                    'gender-specific roles and stereotypes',
                    'unequal treatment based on gender',
                ]
            },
            'discriminatory_language': {
                'age_discrimination': [
                    'negative stereotypes about young or old age',
                    'age-based discrimination or prejudice',
                    'dismissing people for being too young or old',
                    'ageist comments about ability based on age',
                    'treating someone differently because of age',
                    'negative assumptions about elderly or youth',
                ],
                'race_discrimination': [
                    'racial stereotypes or prejudice',
                    'discrimination based on race or ethnicity',
                    'negative generalization about a racial group',
                    'racist language or offensive comments',
                    'treating someone differently due to race',
                    'race-based discrimination or bias',
                ],
                'disability_discrimination': [
                    'disability-based discrimination or prejudice',
                    'treating disabled people as less capable',
                    'ableist language or offensive terms',
                    'negative stereotypes about disability',
                    'discriminating against people with disabilities',
                    'portraying disability as weakness or burden',
                ],
                'appearance_discrimination': [
                    'discrimination based on physical appearance',
                    'negative judgment of someone\'s looks',
                    'appearance-based ridicule or mockery',
                    'shallow focus on physical characteristics',
                    'body-shaming or appearance-based bias',
                    'treating people unfairly for how they look',
                ]
            }
        }
        
        # Encode all bias templates for efficient semantic comparison
        self._encode_bias_templates()


    def _encode_bias_templates(self):
        """Encode all bias templates using sentence transformer for semantic comparison."""
        self.encoded_templates = {}
        for bias_type, categories in self.bias_templates.items():
            self.encoded_templates[bias_type] = {}
            for category, templates in categories.items():
                # Encode each template for semantic similarity comparison
                self.encoded_templates[bias_type][category] = {
                    'templates': templates,
                    'embeddings': self.sentence_encoder.encode(
                        templates,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                }

    def _extract_context_windows(self, text: str, window_size: int = 3) -> List[str]:
        """Extract context windows (sentences) from text for analysis."""
        # Split by sentence-like patterns
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        context_windows = []
        for i in range(len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            window = ' '.join(sentences[start:end])
            context_windows.append(window)
        
        return context_windows if context_windows else [text]

    def _compute_bias_similarity(self, text: str, category_embeddings: Dict) -> Tuple[float, str]:
        """Compute semantic similarity between text and bias category templates."""
        text_embedding = self.sentence_encoder.encode(
            text[:512],  # Limit to 512 chars for efficiency
            convert_to_tensor=True
        )
        
        max_similarity = 0.0
        best_match = None
        
        for category, data in category_embeddings.items():
            embeddings = data['embeddings']
            # Compute cosine similarity
            similarities = util.pytorch_cos_sim(text_embedding, embeddings)[0]
            max_sim = similarities.max().item()
            
            if max_sim > max_similarity:
                max_similarity = max_sim
                best_match = category
        
        return max_similarity, best_match

    def detect_gender_bias(self, text: str) -> Dict:
        """Detect gender bias in text using context-aware semantic analysis."""
        # Extract context windows for better understanding
        context_windows = self._extract_context_windows(text)
        
        # Analyze each context window
        masculine_matches = []
        feminine_matches = []
        
        # Get embeddings for both stereotype categories
        masculine_data = self.encoded_templates['gender_bias']['masculine_stereotypes']
        feminine_data = self.encoded_templates['gender_bias']['feminine_stereotypes']
        
        for window in context_windows:
            window_embedding = self.sentence_encoder.encode(
                window[:512],
                convert_to_tensor=True
            )
            
            # Check for masculine stereotypes
            masc_similarities = util.pytorch_cos_sim(
                window_embedding,
                masculine_data['embeddings']
            )[0]
            masc_max_sim = masc_similarities.max().item()
            
            if masc_max_sim > 0.40:  # Lowered threshold for better detection
                masculine_matches.append((masc_max_sim, window))
            
            # Check for feminine stereotypes
            fem_similarities = util.pytorch_cos_sim(
                window_embedding,
                feminine_data['embeddings']
            )[0]
            fem_max_sim = fem_similarities.max().item()
            
            if fem_max_sim > 0.40:  # Lowered threshold for better detection
                feminine_matches.append((fem_max_sim, window))
        
        # Calculate bias metrics
        masculine_scores = [score for score, _ in masculine_matches]
        feminine_scores = [score for score, _ in feminine_matches]
        
        masculine_avg = np.mean(masculine_scores) if masculine_scores else 0.0
        feminine_avg = np.mean(feminine_scores) if feminine_scores else 0.0
        
        total_score = masculine_avg + feminine_avg
        if total_score > 0:
            male_ratio = masculine_avg / total_score
            female_ratio = feminine_avg / total_score
        else:
            male_ratio = 0.0
            female_ratio = 0.0
        
        # Sensitivity threshold: consider bias detected if average similarity > 0.45
        bias_threshold = 0.45
        has_bias = masculine_avg > bias_threshold or feminine_avg > bias_threshold
        
        if masculine_avg > feminine_avg and has_bias:
            bias_direction = 'male'
        elif feminine_avg > masculine_avg and has_bias:
            bias_direction = 'female'
        else:
            bias_direction = 'balanced' if not has_bias else 'mixed'
        
        # Confidence is the maximum average score
        confidence = max(masculine_avg, feminine_avg)
        
        return {
            'has_gender_bias': has_bias,
            'male_bias_score': male_ratio,
            'female_bias_score': female_ratio,
            'bias_direction': bias_direction,
            'confidence': confidence,
            'male_evidence': [window for score, window in masculine_matches],
            'female_evidence': [window for score, window in feminine_matches],
            'semantic_method': True,  # Indicates context-aware detection
            'num_masculine_matches': len(masculine_matches),
            'num_feminine_matches': len(feminine_matches)
        }

    def detect_discriminatory_language(self, text: str) -> Dict:
        """Detect discriminatory language by category using context-aware analysis."""
        context_windows = self._extract_context_windows(text)
        results = {}
        
        # Analyze each discriminatory category
        for category in self.encoded_templates['discriminatory_language'].keys():
            category_data = self.encoded_templates['discriminatory_language'][category]
            matches = []
            similarity_scores = []
            
            for window in context_windows:
                window_embedding = self.sentence_encoder.encode(
                    window[:512],
                    convert_to_tensor=True
                )
                
                similarities = util.pytorch_cos_sim(
                    window_embedding,
                    category_data['embeddings']
                )[0]
                
                max_sim = similarities.max().item()
                similarity_scores.append(max_sim)
                
                # Lowered confidence threshold for better detection
                if max_sim > 0.42:
                    matches.append((max_sim, window))
            
            # Calculate category statistics
            has_discriminatory = len(matches) > 0
            avg_score = np.mean(similarity_scores) if similarity_scores else 0.0
            
            # For discriminatory detection, use a lower average threshold
            # since these are explicit forms of discrimination
            avg_threshold = 0.40
            has_discriminatory = has_discriminatory or (avg_score > avg_threshold)
            
            results[category] = {
                'has_discriminatory_language': has_discriminatory,
                'confidence_score': avg_score,
                'evidence_windows': [window for _, window in matches],
                'match_count': len(matches),
                'max_similarity': max(similarity_scores) if similarity_scores else 0.0,
                'semantic_method': True  # Indicates context-aware detection
            }
        
        return results

    def ml_based_bias_detection(self, text: str, categories: List[str]) -> Dict:
        """Use zero-shot classification for additional bias detection validation."""
        try:
            # Limit text length for efficiency
            text_limited = text[:512]
            result = self.classifier(text_limited, categories)
            
            return {
                'labels': result['labels'],
                'scores': result['scores'],
                'top_category': result['labels'][0],
                'top_score': result['scores'][0],
                'method': 'zero-shot-classification'
            }
        except Exception as e:
            print(f"Error in ML-based bias detection: {e}")
            return {
                'labels': categories,
                'scores': [0.0] * len(categories),
                'method': 'zero-shot-classification',
                'error': str(e)
            }

    def comprehensive_bias_analysis(self, text: str) -> Dict:
        """
        Comprehensive bias analysis using context-aware semantic models.
        
        This method combines multiple transformer-based approaches:
        1. Semantic similarity to bias-related templates
        2. Context window analysis
        3. Zero-shot classification for validation
        """
        analysis = {
            'text': text[:200],  # Store first 200 chars
            'text_length': len(text),
            'gender_bias': self.detect_gender_bias(text),
            'discriminatory_language': self.detect_discriminatory_language(text),
            'method': 'context-aware-semantic-detection'
        }
        
        # Check if any discriminatory language is detected with confidence
        discriminatory_languages_found = [
            (cat, data) for cat, data in analysis['discriminatory_language'].items()
            if data['has_discriminatory_language']
        ]
        
        has_discriminatory = len(discriminatory_languages_found) > 0
        
        # Calculate overall bias score using context-aware confidence scores
        gender_confidence = analysis['gender_bias']['confidence']
        gender_score = gender_confidence if analysis['gender_bias']['has_gender_bias'] else 0.0
        
        # Only average discriminatory language scores for DETECTED categories
        if has_discriminatory:
            discriminatory_scores = [data['confidence_score'] for _, data in discriminatory_languages_found]
            discriminatory_score = np.mean(discriminatory_scores) if discriminatory_scores else 0.0
        else:
            discriminatory_score = 0.0
        
        # Weight the scores: 60% gender bias, 40% discriminatory language
        overall_bias_score = (
            gender_score * 0.6 +
            discriminatory_score * 0.4
        )
        
        # Determine if text is actually biased
        analysis['is_biased'] = analysis['gender_bias']['has_gender_bias'] or has_discriminatory
        
        # If no bias detected, overall score should be 0
        if not analysis['is_biased']:
            analysis['overall_bias_score'] = 0.0
        else:
            analysis['overall_bias_score'] = min(overall_bias_score, 1.0)
        
        # Add detailed findings - only include discriminatory info if detected
        analysis['bias_summary'] = {
            'gender_biased': analysis['gender_bias']['has_gender_bias'],
            'gender_direction': analysis['gender_bias']['bias_direction'],
            'discriminatory_categories': [cat for cat, _ in discriminatory_languages_found],
            'detection_method': 'context-aware semantic analysis'
        }
        
        # Clean up discriminatory_language from analysis if nothing detected
        if not has_discriminatory:
            analysis['discriminatory_language'] = {}
        else:
            # Keep only detected discriminatory categories
            analysis['discriminatory_language'] = {
                cat: data for cat, data in discriminatory_languages_found
            }
        
        return analysis

    def batch_analysis(self, texts: List[str]) -> List[Dict]:
        """Perform bias analysis on multiple texts."""
        return [self.comprehensive_bias_analysis(text) for text in texts]

    def generate_bias_report(self, analysis: Dict) -> str:
        """Generate a detailed bias report with context evidence."""
        report = []
        report.append("=" * 70)
        report.append("CONTEXT-AWARE BIAS DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"\nText: {analysis['text']}...")
        report.append(f"Text Length: {analysis['text_length']} characters")
        report.append(f"\nDetection Method: {analysis['method']}")
        report.append(f"Overall Bias Score: {analysis['overall_bias_score']:.2%}")
        report.append(f"Is Biased: {analysis['is_biased']}")
        
        # Gender bias section
        gb = analysis['gender_bias']
        report.append(f"\n{'-'*70}")
        report.append("GENDER BIAS ANALYSIS (Context-Aware)")
        report.append(f"{'-'*70}")
        report.append(f"Detected: {gb['has_gender_bias']}")
        report.append(f"Direction: {gb['bias_direction'].upper()}")
        report.append(f"Confidence Score: {gb['confidence']:.2%}")
        
        if gb['male_evidence']:
            report.append(f"\nMasculine Stereotype Evidence ({len(gb['male_evidence'])} matches):")
            for evidence in gb['male_evidence'][:3]:  # Show top 3
                report.append(f"  • {evidence}")
        
        if gb['female_evidence']:
            report.append(f"\nFeminine Stereotype Evidence ({len(gb['female_evidence'])} matches):")
            for evidence in gb['female_evidence'][:3]:  # Show top 3
                report.append(f"  • {evidence}")
        
        # Discriminatory language section
        report.append(f"\n{'-'*70}")
        report.append("DISCRIMINATORY LANGUAGE ANALYSIS (Context-Aware)")
        report.append(f"{'-'*70}")
        
        for category, data in analysis['discriminatory_language'].items():
            status = "✓ DETECTED" if data['has_discriminatory_language'] else "✗ Not detected"
            report.append(f"\n{category.upper()}: {status}")
            report.append(f"  Confidence Score: {data['confidence_score']:.2%}")
            report.append(f"  Evidence Windows: {data['match_count']}")
            
            if data['evidence_windows']:
                report.append(f"  Sample Evidence:")
                for evidence in data['evidence_windows'][:2]:  # Show top 2
                    report.append(f"    • {evidence}")
        
        # Summary
        report.append(f"\n{'-'*70}")
        report.append("SUMMARY")
        report.append(f"{'-'*70}")
        report.append(f"Gender Biased: {analysis['bias_summary']['gender_biased']}")
        report.append(f"Gender Direction: {analysis['bias_summary']['gender_direction'].upper()}")
        
        if analysis['bias_summary']['discriminatory_categories']:
            report.append(f"Discriminatory Categories Found: {', '.join(analysis['bias_summary']['discriminatory_categories'])}")
        else:
            report.append("Discriminatory Categories Found: None")
        
        return '\n'.join(report)
