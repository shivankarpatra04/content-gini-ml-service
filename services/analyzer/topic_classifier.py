from transformers import pipeline

class TopicClassifier:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification",  # Changed from text-classification
                                 model="facebook/bart-large-mnli")

    def classify(self, text):
        topics = ["technology", "business", "health", "education", 
                 "entertainment", "politics", "science", "sports"]
        
        # Updated classification call
        result = self.classifier(
            text,
            candidate_labels=topics,  # Use candidate_labels instead of hypothesis
            multi_label=True
        )
        
        # Format results
        results = [
            {
                "topic": label,
                "confidence": round(score, 2)
            }
            for label, score in zip(result['labels'], result['scores'])
        ]
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)[:3]
