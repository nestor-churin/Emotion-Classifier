#!/usr/bin/env python3
"""
Скрипт для запуску системи класифікації емоцій
"""
import asyncio
import argparse
import site
import sys
import os
import subprocess

from config import settings, validate_settings, create_directories, print_settings
from emotion_classifier import EmotionClassifier
from database import DatabaseManager
from utils import setup_logging

async def setup_system():
    """Налаштування системи"""
    print("🚀 Ініціалізація системи класифікації емоцій...")
    
    # Створення необхідних директорій
    create_directories()
    
    # Налаштування логування
    setup_logging(settings.LOG_LEVEL, settings.LOG_FILE)
    
    # Валідація налаштувань
    validate_settings()
    
    # Виведення налаштувань
    print_settings()
    
    return True

async def train_model():
    """Навчання моделі"""
    print("🧠 Навчання моделі...")
    classifier = EmotionClassifier()
    
    # Тренування моделі з датасетом з налаштувань
    dataset_path = os.path.join(settings.DATASET_PATH, "all_data.csv")
    success = classifier.train_model(dataset_path)
    
    if success:
        print("✅ Модель навчена!")
        print("📊 Модель збережена успішно")
        return True
    else:
        print("❌ Помилка навчання моделі")
        return False

async def test_classifier():
    """Тестування класифікатора"""
    print("🧪 Тестування класифікатора...")
    
    classifier = EmotionClassifier()
    
    # Завантаження моделі
    if not classifier.load_model():
        print("❌ Модель не знайдена. Спочатку навчіть модель командою: python start.py --train")
        return False
    
    # Тестові тексти
    test_texts = [
        "Я дуже щасливий сьогодні! Це чудовий день!",
        "Мене дуже злить ця ситуація, це просто жахливо!",
        "Мені сумно через те, що сталося...",
        "Боюся, що щось погане може статися",
        "Не можу повірити, що це сталося! Яке диво!",
        "Це огидно і противно. Фу!"
    ]
    
    print("\n📝 Результати тестування:")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        predicted_emotion, confidence, all_emotions = classifier.predict_emotion(text)
        
        print(f"\n{i}. Текст: {text}")
        print(f"   Емоція: {predicted_emotion}")
        print(f"   Впевненість: {confidence:.3f}")
        print(f"   Всі емоції: {', '.join([f'{k}: {v:.3f}' for k, v in all_emotions.items()])}")
    
    print("=" * 70)
    print("✅ Тестування завершено!")
    
    return True

async def init_database():
    """Ініціалізація бази даних"""
    print("🗄️ Ініціалізація бази даних...")
    
    db_manager = DatabaseManager(settings.DATABASE_PATH)
    await db_manager.init_db()
    
    print("✅ База даних ініціалізована!")
    
    return True

def run_server():
    """Запуск сервера"""
    print("🌐 Запуск сервера...")
    
    import uvicorn
    
    # Перевірка наявності моделі
    classifier = EmotionClassifier()
    
    if not classifier.load_model():
        print("⚠️ Модель не знайдена. Навчаємо нову модель...")
        # Навчаємо модель синхронно
        asyncio.run(train_model())
    
    print(f"🚀 Сервер запускається на http://{settings.HOST}:{settings.PORT}")
    print(f"📖 Документація API: http://{settings.HOST}:{settings.PORT}/docs")
    print("🛑 Для зупинки сервера натисніть Ctrl+C")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description="Система класифікації емоцій")
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Налаштування системи"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Навчання моделі"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Тестування класифікатора"
    )
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Ініціалізація бази даних"
    )
    
    parser.add_argument(
        "--server",
        action="store_true",
        help="Запуск сервера"
    )
    
    args = parser.parse_args()
    
    # Якщо не передано аргументів, запускаємо сервер
    if not any(vars(args).values()):
        args.server = True
    
    try:
        if args.setup:
            asyncio.run(setup_system())
        
        elif args.train:
            asyncio.run(setup_system())
            asyncio.run(train_model())
        
        elif args.test:
            asyncio.run(setup_system())
            asyncio.run(test_classifier())
        
        elif args.init_db:
            asyncio.run(setup_system())
            asyncio.run(init_database())
        elif args.server:
            asyncio.run(setup_system())
            run_server()
    
    except KeyboardInterrupt:
        print("\n🛑 Роботу перервано користувачем")
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
