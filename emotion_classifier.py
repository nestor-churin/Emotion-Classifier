"""
Модуль класифікатора емоцій - ML модель для ініціалізації та класифікації тексту
"""
import pickle
import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from filters import clean_text_for_prediction
from config import config
from text_analyzer_compat import text_analyzer
import logging

logger = logging.getLogger(__name__)


class EmotionClassifier:
    """Клас для класифікації емоцій в тексті"""
    
    def __init__(self):
        """Ініціалізація класифікатора"""
        self.model = None
        self.is_loaded = False
    def load_model(self) -> bool:
        """
        Завантажує збережену модель
        
        Returns:
            True якщо модель завантажена успішно, False інакше
        """
        try:
            # Фіксований шлях до моделі
            model_path = "models/emotion_model.pkl"
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_loaded = True
                logger.info(f"Модель завантажена з {model_path}")
                return True
            else:
                logger.warning(f"Файл моделі {model_path} не знайдено")
                return False
        except Exception as e:
            logger.error(f"Помилка завантаження моделі: {e}")
            return False
    def save_model(self) -> bool:
        """
        Зберігає поточну модель
        
        Returns:
            True якщо модель збережена успішно, False інакше
        """
        try:
            if self.model is not None:
                # Фіксований шлях до моделі
                model_path = "models/emotion_model.pkl"
                
                # Створюємо директорію, якщо вона не існує
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                logger.info(f"Модель збережена в {model_path}")
                return True
            else:
                logger.error("Немає моделі для збереження")
                return False
        except Exception as e:
            logger.error(f"Помилка збереження моделі: {e}")
            return False
    
    def predict_emotion(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Передбачає емоцію для заданого тексту
        
        Args:
            text: Текст для аналізу
            
        Returns:
            Кортеж (передбачена_емоція, впевненість, всі_ймовірності)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Модель не завантажена")
        
        try:
            # Попередня обробка тексту
            processed_text = text_analyzer.preprocess_for_ml(text)
            cleaned_text = clean_text_for_prediction(processed_text)
            
            # Передбачення
            prediction = self.model.predict([cleaned_text])[0]
            probabilities = self.model.predict_proba([cleaned_text])[0]
            
            # Створення словника ймовірностей
            prob_dict = dict(zip(self.model.classes_, probabilities))
            
            # Знаходження максимальної ймовірності
            confidence = max(probabilities)
            
            logger.info(f"Передбачено емоцію: {prediction} з впевненістю {confidence:.4f}")
            
            return prediction, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Помилка передбачення емоції: {e}")
            raise
    
    def train_model(self, dataset_path: str = None) -> bool:
        try:
            # Завантаження датасету
            df = self._load_dataset(dataset_path)
            if df.empty:
                logger.error("Не вдалося завантажити датасет")
                return False
            
            # Додавання додаткових тренувальних даних
            additional_data = self._create_additional_training_data()
            additional_df = pd.DataFrame(additional_data, columns=['text', 'emotion_label'])
            df = pd.concat([df, additional_df], ignore_index=True)
            
            # Підготовка даних
            texts = df['text'].tolist()
            emotions = df['emotion_label'].tolist()
            
            # Попередня обробка текстів
            processed_texts = [text_analyzer.preprocess_for_ml(text) for text in texts]
            cleaned_texts = [clean_text_for_prediction(text) for text in processed_texts]
            
            # Розділення на тренувальні та тестові дані
            X_train, X_test, y_train, y_test = train_test_split(
                cleaned_texts, emotions, test_size=0.2, random_state=42, stratify=emotions
            )
              # Обчислення ваг класів для збалансованості
            unique_emotions = list(set(emotions))
            class_weights = compute_class_weight(
                'balanced', classes=np.array(unique_emotions), y=emotions
            )
            class_weight_dict = dict(zip(unique_emotions, class_weights))
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=20000,  # Збільшуємо кількість ознак
                    ngram_range=(1, 3),  # Використовуємо 1-, 2-, 3-грами
                    stop_words=config.UKRAINIAN_STOP_WORDS,
                    min_df=2,  # Мінімальна частота документів
                    max_df=0.85,  # Максимальна частота документів
                    analyzer='word',
                    token_pattern=r'(?u)\b[а-яїієґё]+\b',
                    lowercase=True,
                    sublinear_tf=True,  # Логарифмічне масштабування TF
                    use_idf=True,
                    smooth_idf=True,
                    norm='l2'  # L2 нормалізація
                )),
                ('classifier', LogisticRegression(
                    class_weight=class_weight_dict,
                    random_state=42,
                    max_iter=5000,  # Збільшуємо кількість ітерацій
                    solver='saga',  # Використовуємо SAGA solver для кращої збіжності
                    penalty='l2',  # L2 регуляризація
                    C=1.0,  # Параметр регуляризації
                    tol=1e-4,  # Точність збіжності
                    multi_class='ovr'  # One-vs-Rest для мультикласової класифікації
                ))
            ])
            
            logger.info("Початок тренування моделі...")
            self.model.fit(X_train, y_train)
            
            # Оцінка моделі
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"Точність на тренувальних даних: {train_score:.4f}")
            logger.info(f"Точність на тестових даних: {test_score:.4f}")
            
            # Детальна оцінка
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=unique_emotions)
            logger.info(f"Звіт класифікації:\n{report}")
            
            self.is_loaded = True
            
            # Збереження моделі
            return self.save_model()
            
        except Exception as e:
            logger.error(f"Помилка тренування моделі: {e}")
            return False
    
    def _load_dataset(self, dataset_path: str = None) -> pd.DataFrame:
        if dataset_path is None:
            dataset_path = "ukr_emotions_dataset/all_data.csv"
        
        try:
            if not os.path.exists(dataset_path):
                logger.warning(f"Датасет не знайдено: {dataset_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(dataset_path)
            logger.info(f"Завантажено {len(df)} записів з датасету")
            
            # Перетворення binary labels в single emotion labels
            emotion_columns = ['Joy', 'Fear', 'Anger', 'Sadness', 'Disgust', 'Surprise']
            
            # Створення колонки з назвою емоції
            def get_emotion_label(row):
                for emotion in emotion_columns:
                    if row[emotion] == 1:
                        return config.EMOTION_MAPPING[emotion]  # Повертаємо українську назву
                return 'невідомо'  # Якщо жодна емоція не активна
            
            df['emotion_label'] = df.apply(get_emotion_label, axis=1)
            
            # Видаляємо записи без чіткої емоції
            df = df[df['emotion_label'] != 'невідомо']
            
            logger.info(f"Після фільтрації: {len(df)} записів")
            logger.info("Розподіл емоцій:")
            logger.info(df['emotion_label'].value_counts())
            
            return df[['text', 'emotion_label']]
            
        except Exception as e:
            logger.error(f"Помилка завантаження датасету: {e}")
            return pd.DataFrame()
    def _create_additional_training_data(self) -> List[Tuple[str, str]]:
        """Створює додаткові тренувальні дані для покращення класифікації"""
        return [            # Радість
            ("це чудово!", "радість"),
            ("я щасливий", "радість"),
            ("прекрасний день", "радість"),
            ("все йде добре", "радість"),
            ("я захоплений", "радість"),
            ("чудові новини!", "радість"),
            ("відмінно!", "радість"),
            ("я в захваті!", "радість"),
            ("я дуже щасливий сьогодні", "радість"),
            ("це чудовий день", "радість"),
            ("сьогодні чудовий день", "радість"),
            ("я в прекрасному настрої", "радість"),
            ("все просто супер", "радість"),
            ("я на сьомому небі", "радість"),
            ("життя прекрасне", "радість"),
            ("сонячний настрій", "радість"),
            ("я радію", "радість"),
            ("веселощі", "радість"),
            ("ура!", "радість"),
            ("класно!", "радість"),
            ("чудесно!", "радість"),
            
            # Сум
            ("мені сумно", "сум"),
            ("я засмучений", "сум"),
            ("все погано", "сум"),
            ("депресивний настрій", "сум"),
            ("сльози на очах", "сум"),
            ("розчарований", "сум"),
            ("важко на душі", "сум"),
            
            # Гнів - додаємо більше специфічних прикладів
            ("я злий", "гнів"),
            ("це несправедливо", "гнів"),
            ("я обурений", "гнів"),
            ("розлючений", "гнів"),
            ("це дратує", "гнів"),
            ("мене дуже злить", "гнів"),
            ("злить ця ситуація", "гнів"),
            ("я розсерджений", "гнів"),
            ("це просто обурливо", "гнів"),
            ("дуже злий на це", "гнів"),
            ("лють бере", "гнів"),
            ("до сказу доводить", "гнів"),
            ("розлютований до білого каління", "гнів"),
            ("це мене бісить", "гнів"),
            ("я в люті", "гнів"),
            
            # Страх
            ("я боюся", "страх"),
            ("це лякає", "страх"),
            ("тривожно", "страх"),
            ("страшно", "страх"),
            ("панічний настрій", "страх"),
            ("жах охопив", "страх"),
            ("моторошно", "страх"),
            
            # Огида
            ("це огидно", "огида"),
            ("мене нудить", "огида"),
            ("це відразливо", "огида"),
            ("противно", "огида"),
            ("бридко", "огида"),
            ("відвратливо", "огида"),
            ("мерзенно", "огида"),
              # Здивування - додаємо більше специфічних прикладів
            ("це дивно", "здивування"),
            ("не очікував", "здивування"),
            ("вражений", "здивування"),
            ("несподівано", "здивування"),
            ("шокований", "здивування"),
            ("не можу повірити", "здивування"),
            ("яке диво", "здивування"),
            ("що за диво", "здивування"),
            ("неймовірно", "здивування"),
            ("фантастично дивно", "здивування"),
            ("дивовижно", "здивування"),
            ("це неможливо", "здивування"),
            ("не вірю своїм очам", "здивування"),
            ("приголомшений", "здивування"),
            ("ошелешений", "здивування"),
            ("вау, не чекав", "здивування"),
            ("це ж справжнє чудо", "здивування"),
            ("не можу повірити що це сталося", "здивування"),
            ("яке диво!", "здивування"),
            ("вау!", "здивування"),
            ("офігеть!", "здивування"),
            ("боже мій!", "здивування"),
            ("це просто неймовірно!", "здивування"),
            ("такого не очікував!", "здивування"),
            ("вражаюче!", "здивування"),
            ("приголомшливо!", "здивування"),
            ("несподіванка!", "здивування"),
            ("круто!", "здивування"),
            ("супер!", "здивування"),
            
            # Навчальні та академічні стреси
            ("не здам курсову", "страх"),
            ("боюся провалити іспит", "страх"),
            ("не встигну здати", "страх"),
            ("завалю сесію", "страх"),
            ("не готовий до іспиту", "страх"),
            ("переживаю за оцінки", "страх"),
            ("стрес через навчання", "страх"),
            ("нервую перед захистом", "страх"),
            ("хвилююся через дедлайн", "страх"),
            ("паніка перед іспитом", "страх"),
            
            # Негативні емоції щодо навчання (сум/фрустрація)
            ("не вдається вчитися", "сум"),
            ("все пропащо з навчанням", "сум"),
            ("безнадійно з курсовою", "сум"),
            ("не справляюся", "сум"),
            ("все йде не так", "сум"),
            ("розчарований результатами", "сум"),
            ("засмучений через оцінки", "сум"),
            ("пригнічений навчанням", "сум"),
            ("втратив мотивацію", "сум"),
            ("важко з навчанням", "сум"),
            
            # Гнів щодо навчальних ситуацій
            ("бісить це завдання", "гнів"),
            ("дратує викладач", "гнів"),
            ("злий на систему освіти", "гнів"),
            ("обурений несправедливістю", "гнів"),
            ("розлючений через оцінку", "гнів"),
            ("це завдання дурне", "гнів"),
            ("викладач неадекватний", "гнів"),
            ("система освіти жахлива", "гнів"),
            ("ненавиджу цей предмет", "гнів"),
            ("бісить вся ця бюрократія", "гнів")
        ]
    
    def validate_health(self) -> bool:
        """Перевіряє здоров'я моделі"""
        try:
            if not self.is_loaded or self.model is None:
                return False
            
            # Тестовий запит
            test_text = "Тестовий текст для перевірки моделі"
            _, confidence, _ = self.predict_emotion(test_text)
            
            return confidence > 0.0
            
        except Exception as e:
            logger.error(f"Помилка при перевірці здоров'я моделі: {e}")
            return False


# Глобальний екземпляр класифікатора
emotion_classifier = EmotionClassifier()


def get_model():
    """Повертає завантажену модель"""
    if not emotion_classifier.is_loaded:
        if not emotion_classifier.load_model():
            logger.error("Не вдалося завантажити модель")
            return None
    return emotion_classifier.model


def predict_emotion(model, text: str) -> Tuple[str, float, Dict[str, float]]:
    """Передбачає емоцію для заданого тексту (сумісність з старим API)"""
    if not emotion_classifier.is_loaded:
        emotion_classifier.model = model
        emotion_classifier.is_loaded = True
    
    return emotion_classifier.predict_emotion(text)


def validate_model_health(model) -> bool:
    """Перевіряє здоров'я моделі (сумісність з старим API)"""
    if not emotion_classifier.is_loaded:
        emotion_classifier.model = model
        emotion_classifier.is_loaded = True
    
    return emotion_classifier.validate_health()
