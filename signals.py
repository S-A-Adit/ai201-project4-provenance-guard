import re
import os
from groq import Groq
import math

class LocalStatsSignal:
    """
    Signal 1: Measures lexical and structural diversity.
    Returns a score between 0.0 (high AI probability) and 1.0 (high human probability).
    """
    @staticmethod
    def clean_and_tokenize(text):
        # Remove non-alphanumeric characters, convert to lowercase, split by whitespace
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    @staticmethod
    def get_sentences(text):
        # Split text into sentences using simple punctuation boundaries
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings/whitespace
        return [s.strip() for s in sentences if s.strip()]

    def analyze(self, text):
        words = self.clean_and_tokenize(text)
        sentences = self.get_sentences(text)
        
        if not words or not sentences:
            return 0.5  # Neutral default for empty/very short text
        
        # 1. Type-Token Ratio (Lexical Diversity)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # 2. Sentence Length Variance (Structural Diversity)
        sentence_lengths = [len(self.clean_and_tokenize(s)) for s in sentences]
        sentence_lengths = [l for l in sentence_lengths if l > 0]
        
        if len(sentence_lengths) < 2:
            std_dev = 0.0
        else:
            mean = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - mean) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)

        # Normalize TTR: AI writing has a TTR of ~0.4-0.6, human writing is higher (~0.6-0.85)
        # We map TTR from range [0.45, 0.75] to [0.0, 1.0]
        if ttr <= 0.45:
            ttr_score = 0.0
        elif ttr >= 0.75:
            ttr_score = 1.0
        else:
            ttr_score = (ttr - 0.45) / 0.30

        # Normalize Std Dev of Sentence Lengths:
        # AI sentence length std_dev is usually small (e.g., 1.5 - 3.5 words)
        # Human sentence length std_dev is usually larger (e.g., 4.0 - 12.0 words)
        # We map std_dev from range [1.5, 7.5] to [0.0, 1.0]
        if std_dev <= 1.5:
            std_dev_score = 0.0
        elif std_dev >= 7.5:
            std_dev_score = 1.0
        else:
            std_dev_score = (std_dev - 1.5) / 6.0

        # Overall statistical score (average of the two, weighted slightly towards sentence variety)
        combined_local = (ttr_score * 0.4) + (std_dev_score * 0.6)
        
        # Keep within bounds
        return max(0.0, min(1.0, combined_local))


class GroqLLMSignal:
    """
    Signal 2: Evaluates the text using Groq's Llama model to spot semantic and stylistic AI footprints.
    Returns a score between 0.0 (high AI probability) and 1.0 (high human probability).
    """
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=self.api_key)

    def analyze(self, text):
        if not text.strip():
            return 0.5
            
        system_prompt = (
            "You are an expert linguistic forensics assistant specializing in distinguishing human-written text from AI-generated text. "
            "Analyze the input text for hallmarks of AI generation (such as overly polite/balanced structures, common transitional clichés "
            "like 'In conclusion', 'Furthermore', 'Moreover', uniform flow, lack of stylistic quirks, or generic hedging) vs. human authorship "
            "(such as irregular pacing, creative/unexpected metaphors, minor typos/grammar irregularities, conversational tone, or emotional resonance). "
            "Respond ONLY with a JSON object containing two fields: "
            "1. 'human_probability': a float between 0.0 (completely AI-generated) and 1.0 (completely human-written). "
            "2. 'reasoning': a short string (maximum 1 sentence) explaining your primary linguistic reason for this probability.\n"
            "Do not include any pre-text, post-text, markdown block wrapping, or explanation outside the JSON."
        )
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_text = chat_completion.choices[0].message.content
            import json
            data = json.loads(response_text)
            
            human_prob = float(data.get("human_probability", 0.5))
            return max(0.0, min(1.0, human_prob))
        except Exception as e:
            # Fallback to neutral/moderate score if API call fails
            print(f"Groq API call error: {e}")
            return 0.5
