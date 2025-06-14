"""
Модуль конфігурації для системи класифікації емоцій
"""
import os
from typing import List, Dict
from dotenv import load_dotenv

# Завантаження змінних з .env файлу
load_dotenv()
class Settings:
    """Налаштування додатку"""
    
    def __init__(self):
        # Основні налаштування сервера
        self.HOST: str = os.getenv("HOST", "127.0.0.1")
        self.PORT: int = int(os.getenv("PORT", "8000"))
        self.DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"        # Шляхи до файлів
        self.BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_PATH: str = os.getenv("DATASET_PATH", os.path.join(self.BASE_DIR, "ukr_emotions_dataset"))
        
        # Фіксований шлях до моделі (не залежить від .env)
        self.MODEL_PATH: str = os.path.join(self.BASE_DIR, "models", "emotion_model.pkl")
            
        self.DATABASE_PATH: str = os.getenv("DATABASE_PATH", os.path.join(self.BASE_DIR, "emotions.db"))
        
        # Налаштування бази даних
        self.DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
        self.DB_TIMEOUT: int = int(os.getenv("DB_TIMEOUT", "30"))
        
        # Налаштування моделі
        self.MODEL_TYPE: str = os.getenv("MODEL_TYPE", "sklearn")
        self.MODEL_NAME: str = os.getenv("MODEL_NAME", "ukr-emotion-classifier")
        self.MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "512"))
        self.BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
        
        # Емоції для класифікації
        self.EMOTIONS: List[str] = ["Joy", "Fear", "Anger", "Sadness", "Disgust", "Surprise"]
        self.EMOTION_LABELS: Dict[str, str] = {
            "Joy": "Радість",
            "Fear": "Страх", 
            "Anger": "Гнів",
            "Sadness": "Сум",
            "Disgust": "Огида",
            "Surprise": "Здивування"
        }
        
        # Маппінг для нового класифікатора
        self.EMOTION_MAPPING: Dict[str, str] = {
            "Joy": "радість",
            "Fear": "страх", 
            "Anger": "гнів",
            "Sadness": "сум",
            "Disgust": "огида",
            "Surprise": "здивування"
        }        # Розширений список українських стоп-слів (оптимізований для токенізації)
        self.UKRAINIAN_STOP_WORDS = [
            # Займенники
            'і', 'в', 'на', 'з', 'за', 'до', 'по', 'від', 'для', 'про', 'як', 'що', 'це', 'той', 'та', 'те',
            'він', 'вона', 'воно', 'вони', 'ми', 'ви', 'я', 'ти', 'мене', 'тебе', 'його', 'її', 'їх', 'нас', 'вас',
            'себе', 'собі', 'сам', 'сама', 'само', 'самі', 'свій', 'своя', 'своє', 'свої', 'котрий', 'яка', 'яке', 'які',
            
            # Сполучники та частки
            'але', 'або', 'якщо', 'коли', 'тому', 'тоді', 'хоча', 'щоб', 'бо', 'адже', 'однак', 'проте',
            'теж', 'також', 'навіть', 'тобто', 'отже', 'через', 'чи', 'ні',
            'оскільки', 'якби', 'ніби', 'мов', 'немов', 'неначе', 'хай', 'нехай',
            
            # Прислівники та частки
            'не', 'так', 'да', 'уже', 'ще', 'тільки', 'лише', 'майже', 'дуже', 'більш', 'менш',
            'досить', 'надто', 'занадто', 'трохи', 'трішки', 'мало', 'багато', 'дещо', 'щось', 'ніщо',
            'де', 'куди', 'звідки', 'звідси', 'тут', 'там', 'всюди', 'ніде', 'туди', 'сюди',
            'тепер', 'зараз', 'вчора', 'завтра', 'сьогодні', 'колись', 'ніколи', 'часом', 'іноді',
            'може', 'можливо', 'звичайно', 'звісно', 'взагалі',
            
            # Дієслова зв'язки та допоміжні
            'буде', 'було', 'бути', 'є', 'був', 'була', 'були', 'мати', 'має', 'мав', 'мала', 'мали',
            'стати', 'став', 'стала', 'стало', 'стали', 'стане', 'ставати', 'мусити', 'треба', 'можна',
            'повинен', 'повинна', 'повинно', 'повинні', 'варто', 'слід', 'потрібно',
            
            # Прийменники
            'у', 'під', 'над', 'перед', 'після', 'між', 'серед', 'крізь',
            'біля', 'коло', 'поряд', 'поруч', 'всередині', 'зовні', 'навколо', 'навкруги',
            'замість', 'крім', 'окрім', 'без', 'проти', 'супроти', 'всупереч', 'завдяки',
            'при', 'зі', 'із', 'ко', 'зо', 'во', 'до', 'ко', 'об', 'від', 'зі',
            
            # Числівники та кількісні слова (очищені від апострофів)
            'один', 'одна', 'одне', 'два', 'три', 'чотири', 'пять', 'шість', 'сім', 'вісім', 'девять', 'десять',
            'багато', 'мало', 'декілька', 'кілька', 'всі', 'всіх', 'всім', 'всього', 'весь', 'вся', 'все', 
            'кожен', 'кожна', 'кожне', 'деякі', 'інші', 'інший', 'інша', 'інше',
            
            # Артиклі та означувальні слова
            'цей', 'ця', 'це', 'ці', 'ті', 'такий', 'така', 'таке', 'такі',
            'який', 'яке', 'які', 'котрий', 'котра', 'котре', 'котрі',
            
            # Інші службові слова та частки
            'ось', 'от', 'оце', 'он', 'ето', 'ото', 'звідти',
            'то', 'ти', 'ся', 'сь', 'те', 'ж', 'же', 'би', 'б', 'ли', 'лі',
            
            # Вигуки та модальні слова
            'ну', 'ох', 'ах', 'ей', 'ого', 'еге', 'ага', 'так', 'ні', 'ой',
            
            # Часові та модальні маркери
            'вже', 'ще', 'лише', 'тільки', 'майже', 'ледь', 'ледве', 'досі', 'відтоді',
            
            # Логічні зв'язки
            'отож', 'тож', 'отак', 'зрештою', 'врешті', 'нарешті', 'принаймні'
        ]
        
        # Налаштування обробки тексту
        self.MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "5"))
        self.LANGUAGE: str = os.getenv("LANGUAGE", "uk")
        
        # Налаштування кешування
        self.CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
        self.CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
        
        # Налаштування логування
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE: str = os.getenv("LOG_FILE", os.path.join(self.BASE_DIR, "logs", "app.log"))
        self.LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Налаштування безпеки
        self.API_KEY: str = os.getenv("API_KEY", "")
        self.RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "100"))
        
        # Налаштування для розробки
        self.SAVE_PREDICTIONS: bool = os.getenv("SAVE_PREDICTIONS", "True").lower() == "true"
        self.ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "True").lower() == "true"

# Створення глобального об'єкта налаштувань
settings = Settings()

# Додатковий об'єкт для зворотної сумісності з новим класифікатором
config = settings

# Функція для валідації налаштувань
def validate_settings():
    """Перевірка коректності налаштувань"""
    errors = []
    
    # Перевірка існування директорій
    if not os.path.exists(settings.DATASET_PATH):
        errors.append(f"Датасет не знайдено: {settings.DATASET_PATH}")
    
    # Перевірка коректності значень
    if settings.PORT < 1 or settings.PORT > 65535:
        errors.append(f"Некоректний порт: {settings.PORT}")
    
    if settings.MAX_TEXT_LENGTH < 10:
        errors.append(f"Занадто мала максимальна довжина тексту: {settings.MAX_TEXT_LENGTH}")
    
    if settings.MIN_TEXT_LENGTH < 1:
        errors.append(f"Занадто мала мінімальна довжина тексту: {settings.MIN_TEXT_LENGTH}")
    
    if settings.BATCH_SIZE < 1:
        errors.append(f"Некоректний розмір батча: {settings.BATCH_SIZE}")
    
    if errors:
        raise ValueError("Помилки в конфігурації:\n" + "\n".join(errors))
    
    return True

# Функція для створення необхідних директорій
def create_directories():
    """Створення необхідних директорій"""
    directories = [
        os.path.dirname(settings.DATABASE_PATH),
        os.path.dirname(settings.LOG_FILE),
        os.path.dirname(settings.MODEL_PATH)  # Створюємо директорію models/
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

# Функція для виведення поточних налаштувань
def print_settings():
    """Виведення поточних налаштувань для дебагу"""
    print("=== НАЛАШТУВАННЯ СИСТЕМИ ===")
    print(f"Хост: {settings.HOST}:{settings.PORT}")
    print(f"Режим дебагу: {settings.DEBUG}")
    print(f"Шлях до датасету: {settings.DATASET_PATH}")
    print(f"Шлях до моделі: {settings.MODEL_PATH}")
    print(f"База даних: {settings.DATABASE_PATH}")
    print(f"Тип моделі: {settings.MODEL_TYPE}")
    print(f"Емоції: {', '.join(settings.EMOTIONS)}")
    print(f"Мова: {settings.LANGUAGE}")
    print(f"Кешування: {settings.CACHE_ENABLED}")
    print(f"Рівень логування: {settings.LOG_LEVEL}")
    print("===========================\n")

if __name__ == "__main__":
    # Перевірка та створення директорій
    create_directories()
    
    # Валідація налаштувань
    try:
        validate_settings()
        print("✓ Налаштування валідні")
    except ValueError as e:
        print(f"✗ Помилка в налаштуваннях: {e}")
    
    # Виведення налаштувань
    print_settings()
