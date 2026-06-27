class ScoringEngine:
    """
    Combines independent signals and calculates confidence.
    """
    @staticmethod
    def calculate(local_score, llm_score, local_weight=0.3, llm_weight=0.7):
        # Weighted combined probability that the text is human-written
        combined_prob = (local_score * local_weight) + (llm_score * llm_weight)
        
        # Ensure it stays within [0.0, 1.0]
        combined_prob = max(0.0, min(1.0, combined_prob))
        
        # Confidence score: represents distance from 0.5 (perfect uncertainty)
        # mapped to a 0.0 to 1.0 scale.
        confidence = 2.0 * abs(combined_prob - 0.5)
        
        return combined_prob, confidence
