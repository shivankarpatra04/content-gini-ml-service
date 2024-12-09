# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from services.analyzer import BlogAnalyzer  # Updated import
from services.generator.blog_generator import BlogGenerator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize services
analyzer = BlogAnalyzer()
blog_generator = BlogGenerator()

# Request models
class AnalyzeRequest(BaseModel):
    text: str

class GenerateBlogRequest(BaseModel):
    title: Optional[str] = ""
    keywords: Optional[List[str]] = []
    tone: Optional[str] = "professional"

@app.get("/")
async def root():
    return {"message": "ML Service is running"}

@app.post("/api/analyze")
async def analyze_content(request: AnalyzeRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Use the combined analyzer
        analysis = analyzer.analyze(request.text)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_blog(request: GenerateBlogRequest):
    try:
        if not request.title and not request.keywords:
            raise HTTPException(
                status_code=400, 
                detail="Either title or keywords are required"
            )
        
        blog = blog_generator.generate(
            request.title,
            request.keywords,
            request.tone
        )
        return blog
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)