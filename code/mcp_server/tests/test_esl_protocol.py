import unittest
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp_server.protocols.esl.esl import ESL

class TestSophisticatedESLProtocol(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the ESL instance once for all tests."""
        cls.esl = ESL()

    def test_joyful_statement_with_intensifier(self):
        """Tests a clearly positive statement with a strong intensifier."""
        text = "I'm absolutely thrilled about this amazing opportunity!"
        result = self.esl.execute({"user_input": text})

        self.assertEqual(result['emotional_state'], 'joy')
        self.assertEqual(result['valence'], 'positive')
        self.assertGreater(result['intensity'], 0.7)
        self.assertIn('joy', [e['emotion'] for e in result['detected_emotions']])

    def test_negated_positive_statement(self):
        """Tests that negation correctly inverts the valence of a positive emotion."""
        text = "I'm not happy about this situation."
        result = self.esl.execute({"user_input": text})

        self.assertEqual(result['valence'], 'negative')
        self.assertTrue(result['contextual_factors']['has_negation'])
        # The underlying emotion is still 'joy' (from happy), but valence is flipped
        self.assertEqual(result['emotional_state'], 'joy')
        detected_joy = next(e for e in result['detected_emotions'] if e['emotion'] == 'joy')
        self.assertEqual(detected_joy['valence'], 'negative')

    def test_diminished_cognitive_statement(self):
        """Tests a statement with a cognitive emotion and a diminisher."""
        text = "I'm slightly confused and don't understand."
        result = self.esl.execute({"user_input": text})

        self.assertEqual(result['emotional_state'], 'confusion')
        self.assertEqual(result['valence'], 'negative')
        self.assertLess(result['intensity'], 0.5) # Diminished intensity

    def test_social_emotion_of_gratitude(self):
        """Tests the detection of a social emotion like gratitude."""
        text = "Thank you so much, I really appreciate it."
        result = self.esl.execute({"user_input": text})

        self.assertEqual(result['emotional_state'], 'gratitude')
        self.assertEqual(result['valence'], 'positive')
        self.assertEqual(result['detected_emotions'][0]['dimension'], 'social')

    def test_mixed_emotions_with_uncertainty(self):
        """Tests a complex sentence with both negative and positive emotions."""
        text = "I am so incredibly angry and frustrated, but I guess I'm also a little hopeful."
        result = self.esl.execute({"user_input": text})

        self.assertEqual(result['valence'], 'mixed')
        self.assertEqual(result['emotional_state'], 'anger') # Dominant emotion

        emotions_found = {e['emotion'] for e in result['detected_emotions']}
        self.assertIn('anger', emotions_found)
        self.assertIn('hope', emotions_found)

        hope_emotion = next(e for e in result['detected_emotions'] if e['emotion'] == 'hope')
        self.assertLess(hope_emotion['confidence'], 0.9) # Due to "I guess"

    def test_multiple_strong_negative_emotions(self):
        """Tests detection of multiple negative emotions in one sentence."""
        text = "This is a terrible, disgusting, and very bad situation."
        result = self.esl.execute({"user_input": text})

        self.assertEqual(result['valence'], 'negative')
        emotions_found = {e['emotion'] for e in result['detected_emotions']}
        self.assertIn('sadness', emotions_found) # from 'terrible'
        self.assertIn('disgust', emotions_found) # from 'disgusting'

    def test_question_with_doubt(self):
        """Tests a sentence that is a question and expresses doubt."""
        text = "Do you think this is a good idea? I'm a bit unsure."
        result = self.esl.execute({"user_input": text})

        self.assertTrue(result['contextual_factors']['is_question'])
        self.assertIn('doubt', [e['emotion'] for e in result['detected_emotions']])
        self.assertEqual(result['emotional_state'], 'doubt')

    def test_neutral_statement(self):
        """Tests a neutral, non-emotional statement."""
        text = "The sky is blue."
        result = self.esl.execute({"user_input": text})

        self.assertEqual(result['emotional_state'], 'neutral')
        self.assertEqual(result['valence'], 'neutral')
        self.assertEqual(len(result['detected_emotions']), 0)

    def test_empty_input(self):
        """Tests that the protocol handles empty input gracefully."""
        text = ""
        result = self.esl.execute({"user_input": text})
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['reason'], 'No user_input provided.')

if __name__ == '__main__':
    unittest.main()
