# Ukrainian Text Emotion Classification System

**Course Project**  
**West Ukrainian National University (WUNU)**  
**Faculty of Computer Information Technologies (FCIT)**  
**Specialization**: Computer Science and Artificial Intelligence (CSAI)

A REST API system for automatic emotion classification in Ukrainian text. The system uses machine learning to identify one of 6 emotions: Joy, Fear, Anger, Sadness, Disgust, Surprise.

## 🌟 Features

- ✅ FastAPI-based REST API
- ✅ Classification of 6 emotions in Ukrainian text
- ✅ Results storage in SQLite database
- ✅ Statistics and analytics
- ✅ Automatic model training
- ✅ Text processing with spaCy
- ✅ Interactive API documentation

## 🏗️ Architecture

The project consists of the following modules:

### 1. `main.py` - Main Module
- FastAPI route registration
- REST API server initialization
- Configuration loading from `.env`
- Classification request handling

### 2. `emotion_classifier.py` - Emotion Classification Module
- Machine learning model loading and training
- Text preprocessing
- Classification function with confidence score

### 3. `database.py` - Database Module
- SQLite database initialization
- Classification results storage
- Statistics retrieval

### 4. `text_analyzer.py` - Text Processing Module
- Tokenization and lemmatization using spaCy
- Keyword extraction
- Part-of-speech filtering

### 5. `utils.py` - Utility Module
- Text cleaning and normalization functions
- Time-related helper functions
- Logging and validation

### 6. `config.py` - Configuration Module
- Centralized settings storage
- Parameter reading from `.env` file

## 📊 Dataset

The system uses the Ukrainian emotions dataset `ukr-detect/ukr-emotions-binary` with the following characteristics:

- **Total number of examples**: 4,949
- **Emotions**: Joy, Fear, Anger, Sadness, Disgust, Surprise
- **Distribution**: Train (2,466), Validation (249), Test (2,234)

## ⚙️ Installation

### 1. Clone the project
```bash
git clone <repository-url>
cd КП
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install spaCy model (optional)
```bash
python -m spacy download uk_core_news_sm
```

### 4. Configure settings
Edit the `.env` file as needed:
```env
HOST=127.0.0.1
PORT=8000
DEBUG=true
LOG_LEVEL=INFO
```

## 🚀 Launch

### Automatic launch (recommended)
```bash
python start.py
```
This will automatically:
- Configure the system
- Train the model (if needed)
- Start the server

### Step-by-step setup

#### 1. System setup
```bash
python start.py --setup
```

#### 2. Model training
```bash
python start.py --train
```

#### 3. Testing
```bash
python start.py --test
```

#### 4. Database initialization
```bash
python start.py --init-db
```

#### 5. Server launch
```bash
python start.py --server
```

## 📚 API Documentation

After starting the server, documentation is available at:
- **Swagger UI**: http://localhost:26000/docs
- **ReDoc**: http://localhost:26000/redoc

### Main endpoints:

#### `POST /classify` - Text Classification
```json
{
  "text": "Я дуже щасливий сьогодні!",
  "save_to_db": true
}
```

**Response:**
```json
{
  "text": "Я дуже щасливий сьогодні!",
  "predicted_emotion": "Joy",
  "confidence": 0.892,
  "all_emotions": {    "Joy": 0.892,
    "Fear": 0.034,
    "Anger": 0.028,
    "Sadness": 0.025,
    "Disgust": 0.012,
    "Surprise": 0.009
  },
  "timestamp": "2025-06-08T15:30:45.123Z"
}
```

#### `POST /stats` - Statistics
```json
{
  "start_date": "2025-06-01T00:00:00Z",
  "end_date": "2025-06-08T23:59:59Z"
}
```

#### `GET /health` - System Status
Checks the status of the server, database, and model.

## 📈 Monitoring and Statistics

The system automatically collects statistics:
- Number of classifications
- Distribution by emotions
- Average model confidence
- Request processing time

## 🔧 Configuration

### Main parameters in `.env`:

```env
# Server
HOST=127.0.0.1
PORT=8000
DEBUG=true

# Model
MODEL_TYPE=sklearn
MAX_TEXT_LENGTH=512
MIN_TEXT_LENGTH=5

# Database
DATABASE_PATH=emotions.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## 🧪 Testing

### Manual testing
```bash
python start.py --test
```

### API testing with curl
```bash
# Text classification
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Це чудовий день!", "save_to_db": true}'

# Health check
curl -X GET "http://localhost:8000/health"
```

## 📁 Project Structure

```
КП/
├── main.py                    # Main FastAPI server
├── config.py                  # System configuration
├── emotion_classifier.py      # Emotion classifier
├── database.py               # Database operations
├── text_analyzer.py          # Text processing
├── utils.py                  # Utility functions
├── start.py                  # Launch script
├── requirements.txt          # Dependencies
├── .env                      # Configuration
├── README.md                 # Documentation
├── models/                   # Trained models
├── logs/                     # System logs
└── ukr_emotions_dataset/     # Dataset
    ├── all_data.csv
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── dataset_info.txt
```

## 🐛 Known Limitations

1. **spaCy model**: If the Ukrainian model is not installed, the system uses basic text processing
2. **Text size**: Works optimally with texts up to 512 characters
3. **Language**: The system is optimized for the Ukrainian language

## 🤝 Contribution

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## 📄 License

This project was created for educational purposes.