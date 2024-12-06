from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

class ContentQualityAnalyzer:
    def __init__(self):
        try:
            # Initialize sentiment analyzer
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )

            # First ensure punkt is downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading punkt tokenizer...")
                nltk.download('punkt', quiet=True)

            # Then ensure averaged_perceptron_tagger is downloaded
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                print("Downloading perceptron tagger...")
                nltk.download('averaged_perceptron_tagger', quiet=True)

            # Initialize tokenizers
            self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            
            # Test tokenization
            test_text = "This is a test. This is another test."
            test_tokens = self.sentence_tokenizer.tokenize(test_text)
            print("NLTK initialization successful")

        except Exception as e:
            print(f"Error in initialization: {e}")
            raise

    def _clean_text(self, text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        return text

    def _calculate_readability_metrics(self, text):
        try:
            # Use class tokenizer instead of direct sent_tokenize
            sentences = self.sentence_tokenizer.tokenize(text)
            words = word_tokenize(text)
            
            # Filter out punctuation from words
            words = [word for word in words if word.isalnum()]
            
            # Basic metrics
            word_count = len(words)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_word_length": round(avg_word_length, 2)
            }
        except Exception as e:
            print(f"Error in readability metrics calculation: {e}")
            # Fallback to basic splitting if NLTK fails
            words = text.split()
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            word_count = len(words)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_word_length": round(avg_word_length, 2)
            }

    def analyze(self, text):
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Get sentiment analysis
            sentiment_result = self.sentiment_classifier(cleaned_text[:512])[0]
            sentiment_score = 1.0 if sentiment_result['label'] == 'POSITIVE' else 0.0
            
            # Get readability metrics
            readability_metrics = self._calculate_readability_metrics(cleaned_text)
            
            # Calculate quality scores
            length_score = min(1.0, readability_metrics["word_count"] / 1000)
            
            # Ideal average sentence length is between 15-20 words
            sentence_length_score = 1.0 - abs(readability_metrics["avg_sentence_length"] - 17.5) / 17.5
            sentence_length_score = max(0, min(1, sentence_length_score))
            
            # Complexity score based on word length (ideal average word length is 4-6 characters)
            word_length_score = 1.0 - abs(readability_metrics["avg_word_length"] - 5) / 5
            word_length_score = max(0, min(1, word_length_score))
            
            # Calculate overall quality score
            quality_score = (
                sentiment_score * 0.3 +
                length_score * 0.2 +
                sentence_length_score * 0.25 +
                word_length_score * 0.25
            )
            
            return {
                "scores": {
                    "sentiment": round(sentiment_score, 2),
                    "length": round(length_score, 2),
                    "readability": round(sentence_length_score, 2),
                    "complexity": round(word_length_score, 2)
                },
                "metrics": readability_metrics,
                "overall_score": round(quality_score, 2),
                "interpretation": self._get_interpretation(quality_score),
                "recommendations": self._get_recommendations(
                    length_score,
                    sentence_length_score,
                    word_length_score,
                    readability_metrics
                )
            }
        except Exception as e:
            print(f"Error in analysis: {e}")
            return {
                "error": str(e),
                "scores": {},
                "metrics": {},
                "overall_score": 0,
                "interpretation": "Error analyzing content",
                "recommendations": []
            }

    def _get_interpretation(self, score):
        if score >= 0.8:
            return "Excellent quality content"
        elif score >= 0.6:
            return "Good quality content"
        elif score >= 0.4:
            return "Average quality content"
        else:
            return "Needs improvement"

    def _get_recommendations(self, length_score, sentence_score, word_score, metrics):
        recommendations = []
        
        if length_score < 0.6:
            recommendations.append(
                f"Consider adding more content. Current word count: {metrics['word_count']}. "
                "Aim for at least 600 words for better depth."
            )
        
        if sentence_score < 0.6:
            if metrics['avg_sentence_length'] > 20:
                recommendations.append(
                    "Try breaking down some longer sentences. "
                    "Aim for an average length of 15-20 words per sentence."
                )
            elif metrics['avg_sentence_length'] < 10:
                recommendations.append(
                    "Consider combining some shorter sentences for better flow. "
                    "Aim for an average length of 15-20 words per sentence."
                )
                
        if word_score < 0.6:
            if metrics['avg_word_length'] > 6:
                recommendations.append(
                    "Consider using simpler words where possible for better readability."
                )
            elif metrics['avg_word_length'] < 4:
                recommendations.append(
                    "Consider using more varied vocabulary to make the content more engaging."
                )
                
        return recommendations