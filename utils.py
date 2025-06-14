"""
Утилітний модуль для допоміжних функцій
"""
import re
import string
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from functools import wraps
import time
import unicodedata
import asyncio

# Налаштування логування
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Налаштування системи логування"""
    
    # Створення директорії для логів
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    # Налаштування форматування
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Налаштування обробників
    handlers = []
    
    # Консольний обробник
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # Файловий обробник
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Налаштування кореневого логера
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)

# Функції для обробки тексту
def clean_text(text: str) -> str:
    """Базове очищення тексту"""
    if not text:
        return ""
    
    # Видалення зайвих пробілів та переносів рядків
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Видалення URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Видалення емейлів
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Видалення тегів (якщо є)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Видалення надмірних символів пунктуації
    text = re.sub(r'[^\w\s\u0400-\u04FF.,!?;:()\-\'\"]+', '', text)
    
    # Нормалізація пунктуації
    text = re.sub(r'[.,!?;:]+', lambda m: m.group()[:2], text)  # Максимум 2 знаки пунктуації підряд
    
    return text.strip()

def normalize_text(text: str) -> str:
    """Нормалізація тексту"""
    if not text:
        return ""
    
    # Нормалізація Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Приведення до нижнього регістру
    text = text.lower()
    
    # Видалення діакритичних знаків (за потреби)
    # text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    
    # Заміна множинних пробілів одним
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def detect_language(text: str) -> str:
    """Простий детектор мови (українська/російська/інші)"""
    if not text:
        return "unknown"
    
    # Підрахунок українських літер
    ukrainian_chars = len(re.findall(r'[іїєґ]', text.lower()))
    
    # Підрахунок російських літер
    russian_chars = len(re.findall(r'[ыэъё]', text.lower()))
    
    # Загальна кількість кирилічних літер
    cyrillic_chars = len(re.findall(r'[а-яёіїєґ]', text.lower()))
    
    if cyrillic_chars < 5:  # Занадто мало кирилиці
        return "other"
    
    if ukrainian_chars > russian_chars:
        return "ukrainian"
    elif russian_chars > 0:
        return "russian"
    else:
        return "ukrainian"  # За замовчуванням українська для кирилиці

def remove_stopwords(words: List[str], stopwords: Optional[List[str]] = None) -> List[str]:
    """Видалення стоп-слів"""
    if not stopwords:
        # Базові українські стоп-слова
        stopwords = [
            'і', 'в', 'на', 'з', 'за', 'до', 'по', 'від', 'для', 'про', 'як', 'що', 'це', 'той', 'та', 'те',
            'він', 'вона', 'воно', 'вони', 'ми', 'ви', 'я', 'ти', 'мене', 'тебе', 'його', 'її', 'їх', 'нас', 'вас',
            'але', 'або', 'якщо', 'коли', 'тому', 'тоді', 'хоча', 'щоб', 'бо', 'адже', 'однак', 'проте',
            'не', 'ні', 'так', 'да', 'уже', 'ще', 'тільки', 'лише', 'навіть', 'майже', 'дуже', 'більш', 'менш',
            'буде', 'було', 'бути', 'є', 'був', 'була', 'були', 'мати', 'має', 'мав', 'мала', 'мали'
        ]
    
    return [word for word in words if word.lower() not in stopwords and len(word) > 2]

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Простий алгоритм виділення ключових слів"""
    if not text:
        return []
    
    # Очищення та нормалізація
    cleaned_text = normalize_text(clean_text(text))
    
    # Токенізація
    words = re.findall(r'\b[а-яіїєґ]+\b', cleaned_text)
    
    # Видалення стоп-слів
    keywords = remove_stopwords(words)
    
    # Підрахунок частоти
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Сортування за частотою
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Повернення топ ключових слів
    return [word for word, freq in sorted_words[:max_keywords]]

# Функції для роботи з часом
def get_current_timestamp() -> datetime:
    """Поточний час з UTC timezone"""
    return datetime.now(timezone.utc)

def format_timestamp(timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Форматування timestamp"""
    return timestamp.strftime(format_str)

def parse_timestamp(timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """Парсинг timestamp з рядка"""
    return datetime.strptime(timestamp_str, format_str)

def time_ago(timestamp: datetime) -> str:
    """Відносний час (наприклад, '2 години тому')"""
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} дн. тому"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} год. тому"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} хв. тому"
    else:
        return "щойно"

# Декоратори
def measure_time(func):
    """Декоратор для вимірювання часу виконання функції"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Функція {func.__name__} виконалась за {execution_time:.4f} сек")
        
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Функція {func.__name__} виконалась за {execution_time:.4f} сек")
        
        return result
    
    if hasattr(func, '__call__') and hasattr(func, '__await__'):
        return async_wrapper
    else:
        return sync_wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Декоратор для повторних спроб виконання функції"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger = logging.getLogger(func.__module__)
                        logger.warning(f"Спроба {attempt + 1} не вдалась: {str(e)}")
                        await asyncio.sleep(delay)
                    else:
                        break
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger = logging.getLogger(func.__module__)
                        logger.warning(f"Спроба {attempt + 1} не вдалась: {str(e)}")
                        time.sleep(delay)
                    else:
                        break
            
            raise last_exception
        
        if hasattr(func, '__call__') and hasattr(func, '__await__'):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Функції для валідації
def validate_text(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """Валідація тексту"""
    if not isinstance(text, str):
        return False
    
    text = text.strip()
    if len(text) < min_length or len(text) > max_length:
        return False
    
    return True

def validate_emotion(emotion: str, valid_emotions: List[str]) -> bool:
    """Валідація емоції"""
    return emotion in valid_emotions

def sanitize_filename(filename: str) -> str:
    """Очищення імені файлу від небезпечних символів"""
    # Видалення небезпечних символів
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Видалення контрольних символів
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Обмеження довжини
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename.strip()

# Функції для роботи з даними
def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Розділення списку на чанки"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Розгортання вкладеного словника"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_confidence_stats(confidences: List[float]) -> Dict[str, float]:
    """Розрахунок статистики по впевненості"""
    if not confidences:
        return {}
    
    return {
        "mean": sum(confidences) / len(confidences),
        "min": min(confidences),
        "max": max(confidences),
        "std": (sum((x - sum(confidences) / len(confidences)) ** 2 for x in confidences) / len(confidences)) ** 0.5
    }

# Функції для debug
def debug_info(obj: Any, max_depth: int = 3) -> Dict[str, Any]:
    """Отримання debug інформації об'єкта"""
    if max_depth <= 0:
        return {"type": type(obj).__name__, "value": str(obj)[:100]}
    
    info = {
        "type": type(obj).__name__,
        "id": id(obj)
    }
    
    if hasattr(obj, '__dict__'):
        info["attributes"] = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                info["attributes"][key] = debug_info(value, max_depth - 1)
    
    return info

# Тестування утиліт
def test_utils():
    """Тестування утилітних функцій"""
    # Тестування очищення тексту
    test_text = "Це тестовий   текст!!! З https://example.com посиланням..."
    cleaned = clean_text(test_text)
    print(f"Очищений текст: {cleaned}")
    
    # Тестування нормалізації
    normalized = normalize_text(cleaned)
    print(f"Нормалізований текст: {normalized}")
    
    # Тестування виділення ключових слів
    keywords = extract_keywords("Це дуже важливий текст про машинне навчання та штучний інтелект")
    print(f"Ключові слова: {keywords}")
    
    # Тестування детекції мови
    lang = detect_language("Це український текст")
    print(f"Мова: {lang}")
    
    # Тестування часу
    now = get_current_timestamp()
    formatted = format_timestamp(now)
    print(f"Поточний час: {formatted}")

if __name__ == "__main__":
    test_utils()
