# services/analyzer/keyword_extractor.py
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk

class KeywordExtractor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

    def extract(self, text):
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        words = [word for word in tokens 
                if word.isalpha() and word not in self.stop_words]
        
        # Calculate frequency distribution
        fdist = FreqDist(words)
        
        # Get top 10 keywords with their frequencies
        keywords = [(word, count/len(words)) 
                   for word, count in fdist.most_common(10)]
        
        return [{"keyword": kw, "relevance": round(score, 2)} 
                for kw, score in keywords]