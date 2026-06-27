import unittest
import json
import os
from unittest.mock import patch, MagicMock
from app import app
from signals import LocalStatsSignal
from scoring import ScoringEngine
from labels import LabelGenerator
from audit import AuditLog

class TestProvenanceGuard(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Clear audit logs before each test
        if os.path.exists("audit_log.json"):
            try:
                os.remove("audit_log.json")
            except OSError:
                pass

    def tearDown(self):
        if os.path.exists("audit_log.json"):
            try:
                os.remove("audit_log.json")
            except OSError:
                pass

    def test_local_stats_signal(self):
        signal = LocalStatsSignal()
        # Diverse human text
        human_text = (
            "This is a sentence. And this, right here, is a completely different kind of sentence that has many more words "
            "and represents a complex structural pattern. Short sentence. The quick brown fox jumps over the lazy dog."
        )
        # Uniform repetitive text
        ai_text = (
            "This is a simple sentence. This is a simple sentence. This is a simple sentence. This is a simple sentence."
        )
        
        score_human = signal.analyze(human_text)
        score_ai = signal.analyze(ai_text)
        
        # Human score should be higher than AI score
        self.assertGreater(score_human, score_ai)

    def test_scoring_engine(self):
        # High confidence Human
        prob, conf = ScoringEngine.calculate(0.9, 0.9)
        self.assertGreaterEqual(prob, 0.8)
        self.assertGreaterEqual(conf, 0.7)

        # High confidence AI
        prob, conf = ScoringEngine.calculate(0.1, 0.1)
        self.assertLessEqual(prob, 0.2)
        self.assertGreaterEqual(conf, 0.7)

        # Uncertain
        prob, conf = ScoringEngine.calculate(0.5, 0.5)
        self.assertEqual(prob, 0.5)
        self.assertEqual(conf, 0.0)

    def test_label_generator(self):
        # High confidence human
        lbl = LabelGenerator.generate(0.85, 0.7)
        self.assertEqual(lbl["attribution_result"], "human")
        self.assertEqual(lbl["badge_color"], "green")

        # High confidence AI
        lbl = LabelGenerator.generate(0.15, 0.7)
        self.assertEqual(lbl["attribution_result"], "ai")
        self.assertEqual(lbl["badge_color"], "red")

        # Uncertain
        lbl = LabelGenerator.generate(0.52, 0.1)
        self.assertEqual(lbl["attribution_result"], "uncertain")
        self.assertEqual(lbl["badge_color"], "yellow")

    @patch('signals.GroqLLMSignal.analyze')
    def test_submit_endpoint(self, mock_llm_analyze):
        mock_llm_analyze.return_value = 0.9  # Mock LLM returning human score
        
        payload = {"text": "Hello world, this is a human writing a poem about the sunrise."}
        response = self.app.post('/submit', 
                                 data=json.dumps(payload), 
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn("submission_id", data)
        self.assertIn("attribution_result", data)
        self.assertIn("confidence_score", data)
        self.assertIn("signals", data)
        self.assertIn("transparency_label", data)

    def test_submit_validation(self):
        # Empty payload
        response = self.app.post('/submit', data=json.dumps({}), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Invalid data type
        response = self.app.post('/submit', data=json.dumps({"text": 12345}), content_type='application/json')
        self.assertEqual(response.status_code, 400)

    @patch('signals.GroqLLMSignal.analyze')
    def test_appeal_workflow(self, mock_llm_analyze):
        mock_llm_analyze.return_value = 0.8
        
        # Submit a document first
        payload = {"text": "A standard text excerpt for testing appeals."}
        submit_res = self.app.post('/submit', data=json.dumps(payload), content_type='application/json')
        submit_data = json.loads(submit_res.data)
        sub_id = submit_data["submission_id"]
        
        # Appeal it
        appeal_payload = {
            "submission_id": sub_id,
            "reasoning": "This is completely my own work."
        }
        appeal_res = self.app.post('/appeal', data=json.dumps(appeal_payload), content_type='application/json')
        self.assertEqual(appeal_res.status_code, 200)
        appeal_data = json.loads(appeal_res.data)
        
        self.assertEqual(appeal_data["status"], "under review")
        self.assertEqual(appeal_data["submission_id"], sub_id)
        
        # Check logs to confirm appeal status was updated
        log_res = self.app.get('/log')
        logs = json.loads(log_res.data)
        self.assertEqual(len(logs), 1)
        self.assertIsNotNone(logs[0]["appeal"])
        self.assertEqual(logs[0]["appeal"]["status"], "under review")
        self.assertEqual(logs[0]["appeal"]["reasoning"], "This is completely my own work.")

if __name__ == '__main__':
    unittest.main()
