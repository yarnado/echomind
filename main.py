# =============================================================================
# ECOMIND BACKEND - FastAPI + OpenAI + YOLOv8
# Emotional Wellness Platform Backend
# =============================================================================

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

app = FastAPI(title="Ecomind API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# YOLO Model (load once at startup)
emotion_model = YOLO('yolov8n.pt')  # Use fine-tuned emotion model in production

# Database setup
def init_db():
    conn = sqlite3.connect('ecomind.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT UNIQUE,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Chat sessions
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  message TEXT,
                  role TEXT,
                  sentiment_score REAL,
                  emotional_state TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Emotion detections
    c.execute('''CREATE TABLE IF NOT EXISTS emotion_detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  emotion TEXT,
                  confidence REAL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Emotional patterns
    c.execute('''CREATE TABLE IF NOT EXISTS emotional_patterns
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  pattern_type TEXT,
                  severity TEXT,
                  description TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

init_db()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatMessage(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    emotional_state: str
    sentiment_score: float
    detected_patterns: List[str]

class EmotionDetectionRequest(BaseModel):
    user_id: str
    image_data: str  # base64 encoded

class EmotionDetectionResponse(BaseModel):
    emotion: str
    confidence: float
    advice: str

class InsightsRequest(BaseModel):
    user_id: str
    days: int = 7

# =============================================================================
# EMOTIONAL INTELLIGENCE ENGINE
# =============================================================================

class EmotionalAnalysisEngine:
    """Core engine for emotional pattern detection and analysis"""
    
    # Emotional abuse indicators
    ABUSE_KEYWORDS = [
        'worthless', 'stupid', 'useless', 'pathetic', 'failure',
        'never do anything right', 'always wrong', 'nobody likes you',
        'burden', 'waste of space'
    ]
    
    # Gaslighting patterns
    GASLIGHTING_PATTERNS = [
        'overreacting', 'too sensitive', 'imagining things', 'making it up',
        'never happened', 'crazy', 'paranoid', 'dramatic'
    ]
    
    # Mental health indicators
    ANXIETY_KEYWORDS = [
        'anxious', 'worried', 'panic', 'nervous', 'scared', 'afraid',
        'overwhelming', 'can\'t breathe', 'heart racing'
    ]
    
    DEPRESSION_KEYWORDS = [
        'sad', 'depressed', 'hopeless', 'empty', 'numb', 'worthless',
        'don\'t care', 'give up', 'no point'
    ]
    
    @staticmethod
    def analyze_text(text: str) -> dict:
        """Analyze text for emotional patterns"""
        text_lower = text.lower()
        
        analysis = {
            'sentiment_score': 0.5,  # Default neutral
            'emotional_state': 'neutral',
            'detected_patterns': [],
            'severity': 'low',
            'needs_support': False
        }
        
        # Check for abuse patterns
        abuse_count = sum(1 for word in EmotionalAnalysisEngine.ABUSE_KEYWORDS 
                         if word in text_lower)
        if abuse_count > 0:
            analysis['detected_patterns'].append('self_deprecation')
            analysis['severity'] = 'high' if abuse_count > 2 else 'medium'
            analysis['needs_support'] = True
            analysis['sentiment_score'] = 0.2
        
        # Check for gaslighting
        gaslighting_count = sum(1 for pattern in EmotionalAnalysisEngine.GASLIGHTING_PATTERNS 
                               if pattern in text_lower)
        if gaslighting_count > 0:
            analysis['detected_patterns'].append('gaslighting_language')
            analysis['needs_support'] = True
        
        # Check for anxiety
        anxiety_count = sum(1 for word in EmotionalAnalysisEngine.ANXIETY_KEYWORDS 
                           if word in text_lower)
        if anxiety_count > 0:
            analysis['emotional_state'] = 'anxious'
            analysis['sentiment_score'] = 0.3
            analysis['detected_patterns'].append('anxiety')
        
        # Check for depression
        depression_count = sum(1 for word in EmotionalAnalysisEngine.DEPRESSION_KEYWORDS 
                              if word in text_lower)
        if depression_count > 0:
            analysis['emotional_state'] = 'depressed'
            analysis['sentiment_score'] = 0.2
            analysis['detected_patterns'].append('depression_indicators')
        
        # Positive indicators
        positive_words = ['happy', 'good', 'better', 'grateful', 'thankful', 'excited']
        if any(word in text_lower for word in positive_words):
            analysis['sentiment_score'] = 0.8
            analysis['emotional_state'] = 'positive'
        
        return analysis

    @staticmethod
    def generate_empathic_response(analysis: dict, user_message: str) -> str:
        """Generate empathic, context-aware response"""
        
        patterns = analysis['detected_patterns']
        emotional_state = analysis['emotional_state']
        
        # High-severity responses
        if 'self_deprecation' in patterns and analysis['severity'] == 'high':
            return (
                "I hear the pain in those words. Those harsh labels aren't the truth "
                "about who you are. You're reaching out right now, which shows strength. "
                "What you're feeling is valid, and you deserve kindness—especially from "
                "yourself. Would it help to talk about where these thoughts are coming from?"
            )
        
        # Gaslighting detection
        if 'gaslighting_language' in patterns:
            return (
                "Your feelings and perceptions are valid. You're not 'too sensitive' for "
                "having a reaction—emotions are information. Trust yourself. If someone is "
                "making you question your reality, that's concerning. You deserve to be "
                "heard and believed."
            )
        
        # Anxiety support
        if emotional_state == 'anxious':
            return (
                "Anxiety can feel overwhelming. Let's take this moment by moment together. "
                "You're safe right now. Try this: take a slow breath in for 4 counts, "
                "hold for 4, and out for 6. What's one small thing within your control "
                "right now?"
            )
        
        # Depression support
        if emotional_state == 'depressed':
            return (
                "I'm really glad you shared that with me. Sadness is heavy, and you don't "
                "have to carry it alone. Your feelings make sense given what you're going "
                "through. What has felt even slightly soothing to you lately—maybe music, "
                "a walk, or talking to someone?"
            )
        
        # Positive reinforcement
        if emotional_state == 'positive':
            return (
                "It's wonderful to hear that you're feeling better! These positive moments "
                "are important. What do you think has helped you get to this place?"
            )
        
        # Default empathic response
        return (
            "Thank you for sharing that with me. I'm listening, and what you're feeling "
            "matters. Sometimes just being heard can help us see things more clearly. "
            "Is there a specific part of this you'd like to explore together?"
        )

# =============================================================================
# OPENAI CHATBOT INTEGRATION
# =============================================================================

class EcomindChatbot:
    """OpenAI-powered empathic chatbot with safety guardrails"""
    
    SYSTEM_PROMPT = """You are an empathic AI emotional support companion for Ecomind.

Your role:
- Provide 24/7 emotional support
- Be non-judgmental, warm, and validating
- Detect emotional abuse, gaslighting, and mental health concerns
- Never provide medical diagnosis
- Encourage professional help when appropriate
- Use gentle, human language

Guidelines:
1. ALWAYS validate feelings first
2. Never dismiss or minimize emotions
3. Detect patterns: abuse, gaslighting, self-harm ideation
4. For crisis situations: encourage professional help immediately
5. Use grounding techniques for anxiety
6. Encourage self-compassion
7. Ask open-ended questions
8. Be concise but warm (2-4 sentences typically)

Safety protocols:
- If self-harm or suicide mentioned → immediately provide crisis resources
- If abuse detected → validate and encourage safety planning
- Never roleplay harmful scenarios
- Always prioritize user safety

Tone: Warm, gentle, supportive, human, never clinical."""

    @staticmethod
    async def get_response(user_message: str, analysis: dict) -> str:
        """Get AI response using OpenAI API"""
        
        try:
            # Build context with emotional analysis
            context = f"""
User's emotional state: {analysis['emotional_state']}
Sentiment score: {analysis['sentiment_score']}
Detected patterns: {', '.join(analysis['detected_patterns']) if analysis['detected_patterns'] else 'none'}
Severity: {analysis['severity']}
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use gpt-4 or gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": EcomindChatbot.SYSTEM_PROMPT},
                    {"role": "system", "content": f"Analysis: {context}"},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback to rule-based if API fails
            return EmotionalAnalysisEngine.generate_empathic_response(analysis, user_message)

# =============================================================================
# FACIAL EMOTION DETECTION
# =============================================================================

class FacialEmotionDetector:
    """YOLOv8-based facial emotion detection"""
    
    EMOTION_LABELS = {
        0: 'Angry',
        1: 'Anxious',
        2: 'Happy',
        3: 'Neutral',
        4: 'Sad',
        5: 'Stressed'
    }
    
    @staticmethod
    def detect_emotion(image_data: str) -> dict:
        """
        Detect emotion from base64 encoded image
        
        NOTE: In production, use a fine-tuned YOLOv8 model trained on emotion datasets
        like FER2013, AffectNet, or RAF-DB
        """
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Run YOLO detection
            # In production: use fine-tuned emotion model
            results = emotion_model(img)
            
            # For demonstration, simulate emotion detection
            # Replace with actual model inference
            detected_emotion = np.random.choice(list(FacialEmotionDetector.EMOTION_LABELS.values()))
            confidence = np.random.uniform(0.7, 0.95)
            
            return {
                'emotion': detected_emotion,
                'confidence': float(confidence)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Emotion detection failed: {str(e)}")
    
    @staticmethod
    def get_emotion_advice(emotion: str) -> str:
        """Generate supportive advice based on detected emotion"""
        
        advice_map = {
            'Happy': "It's wonderful to see you feeling good! This is a great moment to notice what's bringing you joy.",
            'Calm': "Your calm energy is showing through. This is a beautiful state to be in—let yourself enjoy it.",
            'Neutral': "You seem at ease right now. Sometimes neutral is exactly where we need to be.",
            'Sad': "I see some sadness in your expression. That's okay—all feelings are valid. Would it help to talk about what's on your mind?",
            'Anxious': "I notice some tension. Anxiety can show up in our faces before we even realize it. Let's take a breath together.",
            'Angry': "There's some intensity in your expression. Anger is information—it tells us something matters. What's behind this feeling?",
            'Stressed': "I can see the stress showing through. You're carrying a lot right now. What's one small way to lighten the load?"
        }
        
        return advice_map.get(emotion, "Thank you for letting me see how you're feeling right now.")

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def save_chat_message(user_id: str, message: str, role: str, 
                     sentiment_score: float, emotional_state: str):
    """Save chat message to database"""
    conn = sqlite3.connect('ecomind.db')
    c = conn.cursor()
    c.execute('''INSERT INTO chat_sessions 
                 (user_id, message, role, sentiment_score, emotional_state)
                 VALUES (?, ?, ?, ?, ?)''',
              (user_id, message, role, sentiment_score, emotional_state))
    conn.commit()
    conn.close()

def save_emotion_detection(user_id: str, emotion: str, confidence: float):
    """Save emotion detection result"""
    conn = sqlite3.connect('ecomind.db')
    c = conn.cursor()
    c.execute('''INSERT INTO emotion_detections 
                 (user_id, emotion, confidence)
                 VALUES (?, ?, ?)''',
              (user_id, emotion, confidence))
    conn.commit()
    conn.close()

def get_user_insights(user_id: str, days: int = 7):
    """Get user's emotional insights"""
    conn = sqlite3.connect('ecomind.db')
    c = conn.cursor()
    
    # Get recent emotions
    c.execute('''SELECT emotion, confidence, timestamp 
                 FROM emotion_detections 
                 WHERE user_id = ? 
                 AND timestamp >= datetime('now', '-' || ? || ' days')
                 ORDER BY timestamp DESC''',
              (user_id, days))
    emotions = c.fetchall()
    
    # Get chat sentiment
    c.execute('''SELECT sentiment_score, emotional_state, timestamp 
                 FROM chat_sessions 
                 WHERE user_id = ? AND role = 'user'
                 AND timestamp >= datetime('now', '-' || ? || ' days')
                 ORDER BY timestamp DESC''',
              (user_id, days))
    sentiments = c.fetchall()
    
    conn.close()
    
    return {
        'emotions': emotions,
        'sentiments': sentiments
    }

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Ecomind API - Emotional Wellness Platform",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatMessage):
    """
    Main chat endpoint - processes user message and returns empathic AI response
    """
    try:
        # Analyze emotional content
        analysis = EmotionalAnalysisEngine.analyze_text(request.message)
        
        # Get AI response
        ai_response = await EcomindChatbot.get_response(request.message, analysis)
        
        # Save to database
        save_chat_message(
            request.user_id,
            request.message,
            'user',
            analysis['sentiment_score'],
            analysis['emotional_state']
        )
        
        save_chat_message(
            request.user_id,
            ai_response,
            'assistant',
            analysis['sentiment_score'],
            analysis['emotional_state']
        )
        
        return ChatResponse(
            response=ai_response,
            emotional_state=analysis['emotional_state'],
            sentiment_score=analysis['sentiment_score'],
            detected_patterns=analysis['detected_patterns']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-emotion", response_model=EmotionDetectionResponse)
async def detect_emotion_endpoint(request: EmotionDetectionRequest):
    """
    Facial emotion detection endpoint
    """
    try:
        # Detect emotion from image
        result = FacialEmotionDetector.detect_emotion(request.image_data)
        
        # Get supportive advice
        advice = FacialEmotionDetector.get_emotion_advice(result['emotion'])
        
        # Save to database
        save_emotion_detection(
            request.user_id,
            result['emotion'],
            result['confidence']
        )
        
        return EmotionDetectionResponse(
            emotion=result['emotion'],
            confidence=result['confidence'],
            advice=advice
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/insights")
async def get_insights_endpoint(request: InsightsRequest):
    """
    Get user's emotional insights and patterns
    """
    try:
        insights = get_user_insights(request.user_id, request.days)
        
        # Process insights
        emotion_summary = {}
        for emotion, confidence, timestamp in insights['emotions']:
            emotion_summary[emotion] = emotion_summary.get(emotion, 0) + 1
        
        avg_sentiment = (
            sum(s[0] for s in insights['sentiments']) / len(insights['sentiments'])
            if insights['sentiments'] else 0.5
        )
        
        return {
            'user_id': request.user_id,
            'period_days': request.days,
            'emotion_summary': emotion_summary,
            'average_sentiment': avg_sentiment,
            'total_interactions': len(insights['emotions']) + len(insights['sentiments']),
            'emotional_trend': 'positive' if avg_sentiment > 0.6 else 'needs_support' if avg_sentiment < 0.4 else 'stable'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# =============================================================================
# STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)