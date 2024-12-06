# services/analyzer/sentiment_analyzer.py
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")

    def analyze(self, text):
        result = self.analyzer(text)[0]
        return {
            "sentiment": result['label'],
            "confidence": round(result['score'], 2)
        }