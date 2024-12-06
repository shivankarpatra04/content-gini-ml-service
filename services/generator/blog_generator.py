# services/generator/blog_generator.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
from functools import lru_cache
import re

class BlogGenerator:
    def __init__(self):
        load_dotenv()
        
        # Initialize API with error handling
        try:
            self._initialize_api()
        except Exception as e:
            print(f"Error initializing API: {str(e)}")
            self.model = None
            
        # Define tone mappings
        self.tone_prompts = {
            "professional": "Write in a formal, business-like tone",
            "casual": "Write in a conversational, friendly tone",
            "academic": "Write in a scholarly, research-focused tone",
            "humorous": "Write in a light-hearted, entertaining tone",
            "technical": "Write in a detailed, technical tone"
        }

    def _initialize_api(self):
        """Initialize the Gemini API with error handling"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def _clean_text(self, text):
        """Clean and format the generated text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Fix common formatting issues
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])((?!["\']))', r'\1 ', text)
        return text

    def _create_prompt(self, title="", keywords=[], tone="professional"):
        """Create a structured prompt for the blog post"""
        base_prompt = """
        Write a detailed blog post{topic_part}.
        
        Requirements:
        - Length: 800-1000 words
        - Tone: {tone_instruction}
        - Structure:
          * Engaging introduction that hooks the reader
          * Clear main points with descriptive subheadings
          * Relevant examples and data points
          * Strong conclusion with key takeaways
        - Writing style:
          * Use active voice
          * Include transition sentences between sections
          * Maintain consistent tone throughout
          * Break up long paragraphs
        - Content quality:
          * Include specific examples
          * Add relevant statistics when possible
          * Provide actionable insights
          * Address potential questions readers might have
        """
        
        # Determine topic part
        if title:
            topic_part = f" about {title}"
        else:
            keywords_text = ", ".join(keywords[:5])  # Limit to top 5 keywords
            topic_part = f" about {keywords_text}"
            
        # Get tone instruction
        tone_instruction = self.tone_prompts.get(
            tone.lower(),
            self.tone_prompts["professional"]
        )
        
        return base_prompt.format(
            topic_part=topic_part,
            tone_instruction=tone_instruction
        )

    @lru_cache(maxsize=1)
    def _estimate_read_time(self, word_count):
        """Estimate reading time based on word count"""
        return max(1, word_count // 200)

    def _generate_meta_description(self, content):
        """Generate meta description with error handling"""
        try:
            if not self.model:
                return ""
                
            meta_prompt = (
                f"Write a compelling meta description (max 155 characters) that "
                f"summarizes this blog post and encourages clicks: {content[:200]}..."
            )
            
            meta_response = self.model.generate_content(meta_prompt)
            if meta_response:
                description = self._clean_text(meta_response.text)
                # Ensure description is not too long
                return description[:155] + ('...' if len(description) > 155 else '')
                
        except Exception as e:
            print(f"Meta description generation failed: {str(e)}")
        
        return ""

    def generate(self, title="", keywords=[], tone="professional"):
        """Generate a blog post with comprehensive error handling"""
        try:
            # Input validation
            if not title and not keywords:
                raise ValueError("Either title or keywords must be provided")
                
            if not self.model:
                raise ValueError("API not properly initialized")
                
            # Generate main content
            prompt = self._create_prompt(title, keywords, tone)
            content_response = self.model.generate_content(prompt)
            
            if not content_response:
                raise Exception("Failed to generate blog content")
                
            # Clean and format content
            blog_content = self._clean_text(content_response.text)
            
            # Calculate word count
            word_count = len(blog_content.split())
            
            # Generate meta description
            meta_description = self._generate_meta_description(blog_content)
            
            return {
                "content": blog_content,
                "meta_description": meta_description,
                "estimated_read_time": self._estimate_read_time(word_count),
                "word_count": word_count,
                "tone_used": tone,
                "error": None
            }
            
        except Exception as e:
            error_message = str(e)
            print(f"Blog generation failed: {error_message}")
            
            return {
                "content": None,
                "meta_description": None,
                "estimated_read_time": 0,
                "word_count": 0,
                "tone_used": None,
                "error": error_message
            }

    def get_supported_tones(self):
        """Return list of supported tones"""
        return list(self.tone_prompts.keys())