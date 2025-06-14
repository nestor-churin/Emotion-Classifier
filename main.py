"""
Основний модуль FastAPI сервера для класифікації емоцій
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime
import asyncio

from config import config
from emotion_classifier import emotion_classifier
from database import DatabaseManager
from utils import setup_logging

# Налаштування логування
logger = setup_logging()

# Ініціалізація FastAPI
app = FastAPI(
    title="Emotion Classification API",
    description="API для класифікації емоцій в тексті",
    version="1.0.0"
)

# Моделі для API
class TextInput(BaseModel):
    text: str
    save_to_db: bool = True

class EmotionResult(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    timestamp: datetime

class StatsRequest(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

# Глобальні змінні
db_manager: DatabaseManager = None

@app.on_event("startup")
async def startup_event():
    """Ініціалізація при запуску сервера"""
    global db_manager
    
    logger.info("Ініціалізація сервера...")
    
    # Ініціалізація бази даних
    db_manager = DatabaseManager(config.DATABASE_PATH)
    await db_manager.init_db()
    
    # Завантаження моделі класифікатора
    if not emotion_classifier.load_model():
        logger.warning("Модель не завантажена, буде потрібно навчити нову")
    
    logger.info("Сервер успішно ініціалізовано")

@app.on_event("shutdown")
async def shutdown_event():
    """Закриття ресурсів при вимкненні сервера"""
    if db_manager:
        await db_manager.close()
    logger.info("Сервер зупинено")

# Root endpoint removed for security

@app.post("/classify", response_model=EmotionResult)
async def classify_emotion(input_data: TextInput):
    """
    Класифікація емоції в тексті
    """
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Текст не може бути порожнім")
        
        # Класифікація емоції
        predicted_emotion, confidence, all_emotions = emotion_classifier.predict_emotion(input_data.text)
        
        # Створення результату
        emotion_result = EmotionResult(
            text=input_data.text,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            all_emotions=all_emotions,
            timestamp=datetime.now()
        )
        
        # Збереження в базу даних для аналізу
        if input_data.save_to_db:
            await db_manager.save_classification(
                text=input_data.text,
                predicted_emotion=predicted_emotion,
                confidence=confidence,
                all_emotions=all_emotions
            )
        
        logger.info(f"Класифіковано текст: '{input_data.text[:50]}...' -> {predicted_emotion}")
        
        return emotion_result
    
    except Exception as e:
        logger.error(f"Помилка при класифікації: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Помилка класифікації: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Перевірка стану сервера
    """
    try:
        # Перевірка підключення до бази даних
        db_status = await db_manager.check_connection()
          # Перевірка стану моделі
        model_status = emotion_classifier.is_loaded
        
        return {
            "status": "healthy" if db_status and model_status else "unhealthy",
            "database": "connected" if db_status else "disconnected",
            "model": "loaded" if model_status else "not_loaded",
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Помилка при перевірці стану: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )
