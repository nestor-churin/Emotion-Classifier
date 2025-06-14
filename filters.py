"""
Модуль filters: містить функції для попередньої обробки тексту
"""
import re
import unicodedata
from typing import List, Optional

def clean_text_for_prediction(text: str) -> str:
    """
    Очищає текст для передбачення емоцій.
    Видаляє спеціальні символи, зайві пробіли, приводить до нижнього регістру.
    
    Args:
        text: Вхідний текст
        
    Returns:
        Очищений текст
    """
    if not text:
        return ""
    
    # Спочатку нормалізація
    text = normalize_ukrainian_text(text)
    
    # Приведення до нижнього регістру
    text = text.lower()
    
    # Видалення URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # Видалення емейлів
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    
    # Видалення HTML тегів
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Видалення хештегів та @ згадок (залишаємо тільки текст)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    
    # Заміна чисел на спеціальний токен (числа рідко несуть емоційне навантаження)
    text = re.sub(r'\b\d+\b', 'ЧИСЛО', text)
    
    # Збереження важливої пунктуації для емоцій
    # Залишаємо знаки оклику, питання, три крапки як індикатори емоцій
    text = re.sub(r'[^\w\s\u0400-\u04FF.,!?;:()\-\'\"]+', ' ', text)
    
    # Нормалізація пунктуації (залишаємо до 2 символів підряд)
    text = re.sub(r'[!]{3,}', '!!', text)  # Множинні знаки оклику
    text = re.sub(r'[?]{3,}', '??', text)  # Множинні знаки питання
    text = re.sub(r'[.]{4,}', '...', text)  # Множинні крапки
    
    # Видалення надмірних пробілів
    text = re.sub(r'\s+', ' ', text)
    
    # Видалення пробілів навколо пунктуації
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    
    # Видалення окремих символів пунктуації
    text = re.sub(r'\s+[.,;:]\s+', ' ', text)
    
    return text.strip()

def normalize_ukrainian_text(text: str) -> str:
    """
    Нормалізація українського тексту
    
    Args:
        text: Вхідний текст
        
    Returns:
        Нормалізований текст
    """
    if not text:
        return ""
    
    # Unicode нормалізація
    text = unicodedata.normalize('NFKC', text)
    
    # Заміна подібних символів
    replacements = {
        # Російські літери на українські
        'ё': 'е', 'э': 'е', 'ъ': '', 'ы': 'и', 'щ': 'щ',
        
        # Типографські символи
        '—': '-', '–': '-', '«': '"', '»': '"',
        ''': "'", ''': "'", '"': '"', '"': '"',
        '…': '...', '№': '#',
        
        # Специфічні українські літери
        'ґ': 'г',  # Спрощення для кращої обробки
        
        # Емоційні символи що можуть заважати
        '(': ' ', ')': ' ', '[': ' ', ']': ' ', '{': ' ', '}': ' ',
        
        # Множинні знаки пунктуації
        '!!': '!', '???': '?', '...': '.', ',,': ',',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Видалення зайвих символів
    text = re.sub(r'[^\w\s\u0400-\u04FF.,!?;:()\-\'\"]+', ' ', text)
    
    # Нормалізація пробілів
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def remove_extra_whitespace(text: str) -> str:
    """
    Видаляє зайві пробіли та переноси рядків
    
    Args:
        text: Вхідний текст
        
    Returns:
        Текст без зайвих пробілів
    """
    if not text:
        return ""
    
    # Видалення переносів рядків та табуляцій
    text = re.sub(r'[\n\r\t]+', ' ', text)
    
    # Видалення множинних пробілів
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def filter_short_words(words: List[str], min_length: int = 3) -> List[str]:
    """
    Фільтрує короткі слова
    
    Args:
        words: Список слів
        min_length: Мінімальна довжина слова
        
    Returns:
        Відфільтрований список слів
    """
    return [word for word in words if len(word) >= min_length]

def remove_ukrainian_stopwords(words: List[str], custom_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Видаляє українські стоп-слова
    
    Args:
        words: Список слів
        custom_stopwords: Додаткові стоп-слова
        
    Returns:
        Список слів без стоп-слів
    """
    # Базові українські стоп-слова
    default_stopwords = {
        'і', 'в', 'на', 'з', 'за', 'до', 'по', 'від', 'для', 'про', 'як', 'що', 'це', 'той', 'та', 'те',
        'він', 'вона', 'воно', 'вони', 'ми', 'ви', 'я', 'ти', 'мене', 'тебе', 'його', 'її', 'їх', 'нас', 'вас',
        'але', 'або', 'якщо', 'коли', 'тому', 'тоді', 'хоча', 'щоб', 'бо', 'адже', 'однак', 'проте',
        'не', 'ні', 'так', 'да', 'уже', 'ще', 'тільки', 'лише', 'навіть', 'майже', 'дуже', 'більш', 'менш',
        'буде', 'було', 'бути', 'є', 'був', 'була', 'були', 'мати', 'має', 'мав', 'мала', 'мали',
        'може', 'можна', 'треба', 'потрібно', 'хочу', 'хоче', 'хочемо', 'хочете', 'хочуть',
        'один', 'два', 'три', 'чотири', 'п\'ять', 'перший', 'другий', 'третій'
    }
    
    # Додаємо користувацькі стоп-слова
    if custom_stopwords:
        default_stopwords.update(custom_stopwords)
    
    return [word for word in words if word.lower() not in default_stopwords]

def preprocess_text_for_ml(text: str) -> str:
    """
    Повна попередня обробка тексту для машинного навчання
    
    Args:
        text: Вхідний текст
        
    Returns:
        Попередньо оброблений текст
    """
    if not text:
        return ""
    
    # Нормалізація
    text = normalize_ukrainian_text(text)
    
    # Очищення
    text = clean_text_for_prediction(text)
    
    # Видалення зайвих пробілів
    text = remove_extra_whitespace(text)
    
    # Токенізація та фільтрація
    words = text.split()
    
    # Фільтруємо короткі слова (менше 2 символів)
    words = filter_short_words(words, min_length=2)
    
    # Видаляємо українські стоп-слова
    words = remove_ukrainian_stopwords(words)
    
    # Видаляємо дуже довгі слова (ймовірно, помилки або неактуальні)
    words = [word for word in words if len(word) <= 20]
    
    # Фільтруємо слова що складаються тільки з пунктуації
    words = [word for word in words if re.search(r'[а-яё\w]', word)]
    
    # Об'єднуємо назад
    processed_text = ' '.join(words)
    
    # Фінальне очищення
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    return processed_text

def extract_hashtags(text: str) -> List[str]:
    """
    Виділяє хештеги з тексту
    
    Args:
        text: Вхідний текст
        
    Returns:
        Список хештегів
    """
    hashtags = re.findall(r'#\w+', text)
    return [tag.lower() for tag in hashtags]

def extract_mentions(text: str) -> List[str]:
    """
    Виділяє згадки (@username) з тексту
    
    Args:
        text: Вхідний текст
        
    Returns:
        Список згадок
    """
    mentions = re.findall(r'@\w+', text)
    return [mention.lower() for mention in mentions]

def clean_social_media_text(text: str) -> str:
    """
    Очищає текст від соціальних мереж (хештеги, згадки, RT тощо)
    
    Args:
        text: Вхідний текст
        
    Returns:
        Очищений текст
    """
    if not text:
        return ""
    
    # Видалення RT (retweet)
    text = re.sub(r'\bRT\b', '', text)
    
    # Видалення хештегів
    text = re.sub(r'#\w+', '', text)
    
    # Видалення згадок
    text = re.sub(r'@\w+', '', text)
    
    # Очищення від зайвих пробілів
    text = remove_extra_whitespace(text)
    
    return text

# Для зворотної сумісності
def clean_text(text: str) -> str:
    """Базове очищення тексту (псевдонім для clean_text_for_prediction)"""
    return clean_text_for_prediction(text)
