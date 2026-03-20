"""Decision-making engine for personalized guidance."""
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from config.settings import DecisionEngineConfig


@dataclass
class Decision:
    """Represents a decision with reasoning."""
    action: str
    confidence: float
    reasoning: List[str]
    recommendations: List[str]
    priority: str  # high, medium, low


class DecisionEngine:
    """Decision-making engine for bias analysis results."""

    def __init__(self, config: DecisionEngineConfig):
        """Initialize decision engine."""
        self.config = config
        
        # Define decision rules
        self.bias_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        # Action recommendations based on analysis
        self.action_recommendations = {
            'gender_bias': {
                'high': [
                    'Review content for gendered language',
                    'Use gender-neutral alternatives',
                    'Add diverse perspective representation',
                    'Include examples from both genders'
                ],
                'medium': [
                    'Audit content for subtle bias',
                    'Include balanced pronouns',
                    'Ensure diverse representation'
                ],
                'low': [
                    'Monitor for bias patterns',
                    'Maintain balanced language'
                ]
            },
            'discriminatory_language': {
                'high': [
                    'Replace discriminatory terms immediately',
                    'Reword to be inclusive',
                    'Add context on inclusive language',
                    'Train on diversity guidelines'
                ],
                'medium': [
                    'Review potentially problematic terms',
                    'Consider alternative phrasing',
                    'Check for unintended implications'
                ],
                'low': [
                    'Standard language review',
                    'Maintain inclusivity'
                ]
            }
        }

    def analyze_bias_results(self, bias_analysis: Dict) -> Decision:
        """Make decision based on bias analysis."""
        score = bias_analysis['overall_bias_score']
        is_biased = bias_analysis['is_biased']
        
        # Determine severity level
        if score >= self.bias_thresholds['high']:
            severity = 'high'
        elif score >= self.bias_thresholds['medium']:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Build reasoning
        reasoning = []
        if is_biased:
            reasoning.append(f"Overall bias score: {score:.2%}")
            
            # Check gender bias
            gb = bias_analysis['gender_bias']
            if gb['has_gender_bias']:
                reasoning.append(f"Gender bias detected: {gb['bias_direction']} bias")
            
            # Check discriminatory language
            disc_lang = bias_analysis['discriminatory_language']
            found_categories = [k for k, v in disc_lang.items() if v['count'] > 0]
            if found_categories:
                reasoning.append(f"Discriminatory language in: {', '.join(found_categories)}")
        else:
            reasoning.append("No significant bias detected")
        
        # Collect recommendations
        recommendations = []
        if bias_analysis['gender_bias']['has_gender_bias']:
            recommendations.extend(
                self.action_recommendations['gender_bias'].get(severity, [])
            )
        
        for category, data in bias_analysis['discriminatory_language'].items():
            if data['count'] > 0:
                recommendations.extend(
                    self.action_recommendations['discriminatory_language'].get(severity, [])
                )
                break
        

        
        # Determine priority
        if score >= 0.8:
            priority = 'high'
        elif score >= 0.5:
            priority = 'medium'
        else:
            priority = 'low'
        
        action = f"Review and potentially revise content - {severity.upper()} bias level"
        
        return Decision(
            action=action,
            confidence=min(score + 0.2, 1.0),  # Adjust confidence
            reasoning=reasoning,
            recommendations=recommendations[:self.config.recommendation_count],
            priority=priority
        )

    def analyze_classification_results(self, predictions: Dict, confidence_threshold: float = None) -> Decision:
        """Make decision based on classification results."""
        if confidence_threshold is None:
            confidence_threshold = self.config.confidence_threshold
        
        # Aggregate predictions from multiple models
        predictions_list = []
        confidences_list = []
        
        for model_name, result in predictions.items():
            label = result['labels'][0] if isinstance(result['labels'], list) else result['labels']
            predictions_list.append(label)
            
            if result['probabilities']:
                proba = result['probabilities'][0] if isinstance(result['probabilities'], list) else result['probabilities']
                max_confidence = max(proba) if isinstance(proba, list) else proba
                confidences_list.append(max_confidence)
        
        # Majority vote
        from collections import Counter
        most_common = Counter(predictions_list).most_common(1)[0][0]
        avg_confidence = sum(confidences_list) / len(confidences_list) if confidences_list else 0.5
        
        # Determine if biased
        is_biased = most_common == 1  # Assuming 1 = biased, 0 = not biased
        
        reasoning = [
            f"Classification consensus: {'BIASED' if is_biased else 'NOT BIASED'}",
            f"Average confidence: {avg_confidence:.2%}",
            f"Models agreement: {len(predictions)} models evaluated"
        ]
        
        recommendations = []
        if is_biased and avg_confidence >= confidence_threshold:
            recommendations = [
                "Apply primary recommendations from bias analysis",
                "Review flagged content sections",
                "Implement suggested language changes",
                "Schedule follow-up review"
            ]
        elif is_biased and avg_confidence < confidence_threshold:
            recommendations = [
                "Manual review recommended",
                "Uncertain classification - human judgment needed",
                "Consider context and domain expertise"
            ]
        else:
            recommendations = [
                "Content appears unbiased",
                "Maintain current quality standards",
                "Continue standard review practices"
            ]
        
        priority = 'high' if is_biased and avg_confidence >= 0.8 else 'medium' if is_biased else 'low'
        
        action = f"{'Flag' if is_biased else 'Approve'} content for bias - Confidence: {avg_confidence:.2%}"
        
        return Decision(
            action=action,
            confidence=avg_confidence,
            reasoning=reasoning,
            recommendations=recommendations[:self.config.recommendation_count],
            priority=priority
        )

    def analyze_topic_results(self, topic_summary: Dict, bias_analysis_results: Dict) -> Decision:
        """Make decision based on topic modeling results."""
        reasoning = [f"Topics identified: {len(topic_summary)}"]
        recommendations = []
        
        # Check if any topic contains potential bias keywords (both languages)
        bias_indicator_words = [
            'male', 'female', 'discriminat',
            'masculino', 'femenino', 'discrimina'
        ]
        biased_topics = []
        for topic_id, info in topic_summary.items():
            keywords = info['keywords']
            if any(word in ' '.join(keywords).lower() for word in bias_indicator_words):
                biased_topics.append(topic_id)
        
        if biased_topics:
            reasoning.append(f"Potentially biased topics found: {biased_topics}")
            recommendations = [
                f"Investigate Topic {tid} for bias patterns" for tid in biased_topics
            ]
        else:
            reasoning.append("No obviously biased topics detected in keywords")
            recommendations = ["Continue monitoring topic distribution"]
        
        priority = 'high' if biased_topics else 'low'
        action = f"Topic analysis: {len(biased_topics)} potentially problematic topics"
        
        return Decision(
            action=action,
            confidence=0.7,
            reasoning=reasoning,
            recommendations=recommendations[:self.config.recommendation_count],
            priority=priority
        )

    def generate_guidance(self, text: str, all_analyses: Dict) -> str:
        """Generate comprehensive guidance based on all analyses."""
        guidance = []
        guidance.append("=" * 70)
        guidance.append("PERSONALIZED GUIDANCE REPORT")
        guidance.append("=" * 70)
        
        guidance.append(f"\nAnalyzed Text (first 300 chars):\n{text[:300]}...")
        
        if 'bias_analysis' in all_analyses:
            decision = self.analyze_bias_results(all_analyses['bias_analysis'])
            guidance.append(self._format_decision("Bias Detection", decision))
        
        if 'classification' in all_analyses:
            decision = self.analyze_classification_results(all_analyses['classification'])
            guidance.append(self._format_decision("Classification", decision))
        
        if 'topic_analysis' in all_analyses:
            decision = self.analyze_topic_results(
                all_analyses.get('topic_summary', {}),
                all_analyses.get('bias_analysis', {})
            )
            guidance.append(self._format_decision("Topic Analysis", decision))
        
        return '\n'.join(guidance)

    def _format_decision(self, title: str, decision: Decision) -> str:
        """Format decision for output."""
        lines = []
        lines.append(f"\n### {title}")
        lines.append(f"Action: {decision.action}")
        lines.append(f"Confidence: {decision.confidence:.2%}")
        lines.append(f"Priority: {decision.priority.upper()}")
        
        if decision.reasoning:
            lines.append(f"\nReasoning:")
            for reason in decision.reasoning:
                lines.append(f"  • {reason}")
        
        if decision.recommendations:
            lines.append(f"\nRecommendations:")
            for i, rec in enumerate(decision.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        return '\n'.join(lines)
