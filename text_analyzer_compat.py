"""
Простий text_analyzer для сумісності з emotion_classifier_new.py
"""
from filters import preprocess_text_for_ml, clean_text_for_prediction

class TextAnalyzerCompat:
    """Простий аналізатор тексту для сумісності"""
    
    def preprocess_for_ml(self, text: str) -> str:
        """
        Попередня обробка тексту для машинного навчання
        
        Args:
            text: Вхідний текст
            
        Returns:
            Оброблений текст
        """
        return preprocess_text_for_ml(text)
    
    def extract_important_words(self, text: str, max_words: int = 20):
        """
        Виділення важливих слів (спрощена версія)
        
        Args:
            text: Вхідний текст
            max_words: Максимальна кількість слів
            
        Returns:
            Список важливих слів
        """
        processed = self.preprocess_for_ml(text)
        words = processed.split()
        # Повертаємо унікальні слова
        return list(dict.fromkeys(words))[:max_words]

# Глобальний екземпляр для сумісності
text_analyzer = TextAnalyzerCompat()
