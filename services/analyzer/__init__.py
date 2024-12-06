# services/analyzer/__init__.py
from .content_quality import ContentQualityAnalyzer
from .keyword_extractor import KeywordExtractor
from .sentiment_analyzer import SentimentAnalyzer
from .topic_classifier import TopicClassifier

class BlogAnalyzer:
    def __init__(self):
        self.quality_analyzer = ContentQualityAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_classifier = TopicClassifier()

    def analyze(self, text):
        return {
            "quality_analysis": self.quality_analyzer.analyze(text),
            "keywords": self.keyword_extractor.extract(text),
            "sentiment": self.sentiment_analyzer.analyze(text),
            "topics": self.topic_classifier.classify(text)
        }