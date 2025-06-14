"""
Модуль роботи з базою даних
"""
import aiosqlite
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import logging

from config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Менеджер для роботи з базою даних SQLite"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        
    async def init_db(self):
        """Ініціалізація бази даних та створення таблиць"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Таблиця для збереження результатів класифікації
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS classifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        predicted_emotion TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        all_emotions TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        processing_time REAL,
                        text_length INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Таблиця для логування запитів API
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS api_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        endpoint TEXT NOT NULL,
                        method TEXT NOT NULL,
                        status_code INTEGER,
                        response_time REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        ip_address TEXT,
                        user_agent TEXT
                    )
                ''')
                
                # Таблиця для збереження метрик системи
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        additional_data TEXT
                    )
                ''')
                
                # Індекси для оптимізації запитів
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_classifications_timestamp 
                    ON classifications(timestamp)
                ''')
                
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_classifications_emotion 
                    ON classifications(predicted_emotion)
                ''')
                
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp 
                    ON api_logs(timestamp)
                ''')
                
                await db.commit()
                
            logger.info(f"База даних ініціалізована: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Помилка ініціалізації бази даних: {str(e)}")
            raise
    
    async def save_classification(
        self, 
        text: str, 
        predicted_emotion: str, 
        confidence: float,
        all_emotions: Dict[str, float],
        processing_time: Optional[float] = None
    ) -> int:
        """Збереження результату класифікації"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute('''
                    INSERT INTO classifications 
                    (text, predicted_emotion, confidence, all_emotions, processing_time, text_length)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    text,
                    predicted_emotion,
                    confidence,
                    json.dumps(all_emotions, ensure_ascii=False),
                    processing_time,
                    len(text)
                ))
                
                await db.commit()
                classification_id = cursor.lastrowid
                
                logger.debug(f"Збережено класифікацію з ID: {classification_id}")
                return classification_id
                
        except Exception as e:
            logger.error(f"Помилка збереження класифікації: {str(e)}")
            raise
    
    async def get_classification_stats(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Отримання статистики класифікацій"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Базовий запит
                where_clause = ""
                params = []
                
                if start_date:
                    where_clause += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    where_clause += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                if where_clause:
                    where_clause = "WHERE" + where_clause[4:]  # Прибираємо перший " AND"
                
                # Загальна статистика
                cursor = await db.execute(f'''
                    SELECT 
                        COUNT(*) as total_classifications,
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence,
                        AVG(text_length) as avg_text_length,
                        AVG(processing_time) as avg_processing_time
                    FROM classifications 
                    {where_clause}
                ''', params)
                
                general_stats = await cursor.fetchone()
                
                # Статистика по емоціях
                cursor = await db.execute(f'''
                    SELECT 
                        predicted_emotion,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence,
                        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM classifications {where_clause}) as percentage
                    FROM classifications 
                    {where_clause}
                    GROUP BY predicted_emotion
                    ORDER BY count DESC
                ''', params + params)  # Подвоюємо params для підзапиту
                
                emotion_stats = await cursor.fetchall()
                
                # Статистика по часу (останні 24 години по годинах)
                cursor = await db.execute(f'''
                    SELECT 
                        strftime('%H', timestamp) as hour,
                        COUNT(*) as count
                    FROM classifications 
                    WHERE timestamp >= datetime('now', '-24 hours')
                    {where_clause.replace('WHERE', 'AND') if where_clause else ''}
                    GROUP BY hour
                    ORDER BY hour
                ''', params)
                
                hourly_stats = await cursor.fetchall()
                
                # Топ текстів з найвищим confidence
                cursor = await db.execute(f'''
                    SELECT text, predicted_emotion, confidence, timestamp
                    FROM classifications 
                    {where_clause}
                    ORDER BY confidence DESC
                    LIMIT 10
                ''', params)
                
                top_confident = await cursor.fetchall()
                
                # Топ текстів з найнижчим confidence
                cursor = await db.execute(f'''
                    SELECT text, predicted_emotion, confidence, timestamp
                    FROM classifications 
                    {where_clause}
                    ORDER BY confidence ASC
                    LIMIT 10
                ''', params)
                
                low_confident = await cursor.fetchall()
                
                # Формування результату
                result = {
                    "period": {
                        "start_date": start_date.isoformat() if start_date else None,
                        "end_date": end_date.isoformat() if end_date else None
                    },
                    "general": {
                        "total_classifications": general_stats[0] or 0,
                        "avg_confidence": round(general_stats[1] or 0, 4),
                        "min_confidence": round(general_stats[2] or 0, 4),
                        "max_confidence": round(general_stats[3] or 0, 4),
                        "avg_text_length": round(general_stats[4] or 0, 2),
                        "avg_processing_time": round(general_stats[5] or 0, 4)
                    },
                    "emotions": [
                        {
                            "emotion": row[0],
                            "count": row[1],
                            "avg_confidence": round(row[2], 4),
                            "percentage": round(row[3], 2)
                        }
                        for row in emotion_stats
                    ],
                    "hourly_distribution": [
                        {"hour": row[0], "count": row[1]}
                        for row in hourly_stats
                    ],
                    "top_confident": [
                        {
                            "text": row[0][:100] + "..." if len(row[0]) > 100 else row[0],
                            "emotion": row[1],
                            "confidence": round(row[2], 4),
                            "timestamp": row[3]
                        }
                        for row in top_confident
                    ],
                    "low_confident": [
                        {
                            "text": row[0][:100] + "..." if len(row[0]) > 100 else row[0],
                            "emotion": row[1],
                            "confidence": round(row[2], 4),
                            "timestamp": row[3]
                        }
                        for row in low_confident
                    ]
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Помилка отримання статистики: {str(e)}")
            raise
    
    async def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Логування API запиту"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO api_logs 
                    (endpoint, method, status_code, response_time, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (endpoint, method, status_code, response_time, ip_address, user_agent))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Помилка логування API запиту: {str(e)}")
    
    async def save_metric(self, metric_name: str, metric_value: float, additional_data: Optional[Dict] = None):
        """Збереження системної метрики"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO system_metrics (metric_name, metric_value, additional_data)
                    VALUES (?, ?, ?)
                ''', (
                    metric_name,
                    metric_value,
                    json.dumps(additional_data, ensure_ascii=False) if additional_data else None
                ))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Помилка збереження метрики: {str(e)}")
    
    async def get_recent_classifications(self, limit: int = 100) -> List[Dict]:
        """Отримання останніх класифікацій"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute('''
                    SELECT text, predicted_emotion, confidence, timestamp
                    FROM classifications
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                rows = await cursor.fetchall()
                
                return [
                    {
                        "text": row[0],
                        "predicted_emotion": row[1],
                        "confidence": row[2],
                        "timestamp": row[3]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Помилка отримання останніх класифікацій: {str(e)}")
            raise
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Очищення старих даних"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Видалення старих класифікацій
                cursor = await db.execute('''
                    DELETE FROM classifications 
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                classifications_deleted = cursor.rowcount
                
                # Видалення старих логів API
                cursor = await db.execute('''
                    DELETE FROM api_logs 
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                logs_deleted = cursor.rowcount
                
                # Видалення старих метрик
                cursor = await db.execute('''
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                metrics_deleted = cursor.rowcount
                
                await db.commit()
                
                logger.info(f"Очищено старі дані: {classifications_deleted} класифікацій, "
                          f"{logs_deleted} логів, {metrics_deleted} метрик")
                
                return {
                    "classifications_deleted": classifications_deleted,
                    "logs_deleted": logs_deleted,
                    "metrics_deleted": metrics_deleted
                }
                
        except Exception as e:
            logger.error(f"Помилка очищення старих даних: {str(e)}")
            raise
    
    async def check_connection(self) -> bool:
        """Перевірка підключення до бази даних"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute('SELECT 1')
                await cursor.fetchone()
                return True
        except Exception as e:
            logger.error(f"Помилка підключення до БД: {str(e)}")
            return False
    
    async def close(self):
        """Закриття підключення"""
        if self.connection:
            await self.connection.close()
            self.connection = None

# Функції для тестування
async def test_database():
    """Тестування роботи з базою даних"""
    db_manager = DatabaseManager("test_emotions.db")
    
    # Ініціалізація
    await db_manager.init_db()
    
    # Тестове збереження
    classification_id = await db_manager.save_classification(
        text="Тестовий текст для перевірки",
        predicted_emotion="Joy",
        confidence=0.85,
        all_emotions={"Joy": 0.85, "Fear": 0.10, "Anger": 0.05},
        processing_time=0.123
    )
    
    print(f"Збережено класифікацію з ID: {classification_id}")
    
    # Отримання статистики
    stats = await db_manager.get_classification_stats()
    print("Статистика:", json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Закриття
    await db_manager.close()

if __name__ == "__main__":
    asyncio.run(test_database())
