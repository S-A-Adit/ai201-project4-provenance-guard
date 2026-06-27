import uuid
from signals import LocalStatsSignal, GroqLLMSignal
from scoring import ScoringEngine
from labels import LabelGenerator
from audit import AuditLog

class DetectionPipeline:
    def __init__(self):
        self.local_stats = LocalStatsSignal()
        self.groq_llm = GroqLLMSignal()

    def process(self, text):
        submission_id = str(uuid.uuid4())
        
        # 1. Run local structural/lexical signal
        local_score = self.local_stats.analyze(text)
        
        # 2. Run Groq stylistic pattern analyzer
        llm_score = self.groq_llm.analyze(text)
        
        # 3. Combine scores & calculate confidence
        combined_prob, confidence = ScoringEngine.calculate(local_score, llm_score)
        
        # 4. Generate user-facing transparency label
        label_details = LabelGenerator.generate(combined_prob, confidence)
        
        # 5. Log the attribution decision in the audit log
        entry = AuditLog.log_submission(
            submission_id=submission_id,
            text=text,
            local_score=local_score,
            llm_score=llm_score,
            combined_prob=combined_prob,
            confidence=confidence,
            label_details=label_details
        )
        
        return entry
