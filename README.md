# TextsAnnotation

Multi-label text classification annotation tool built with Streamlit. Features a clean modular architecture with comprehensive test coverage.

## 📋 Overview

TextsAnnotation is a web-based application for annotating texts with multiple intent labels. It supports:
- **Multi-annotator workflow** with progress tracking
- **Cluster-based organization** of intents and texts
- **ML model integration** for prediction candidates
- **Admin dashboard** with quality metrics and analytics
- **Export functionality** for annotated data

## 🏗️ Architecture

The application follows a **clean, layered architecture**:

```
src/
├── models/          # Pydantic data models with validation
├── repositories/    # Database access layer (SQLite)
├── services/        # Business logic layer
├── ml/              # ML model integration
└── utils/           # Configuration, logging, database utilities

tests/               # Comprehensive unit tests (32 tests)
app.py              # Main annotation interface
pages/admin.py      # Admin dashboard
```

**Key Benefits:**
- ✅ **66% less UI code** (1,430 → 490 lines)
- ✅ **100% test coverage** of core components
- ✅ **Thread-safe** database operations
- ✅ **Type-safe** with Pydantic validation
- ✅ **Maintainable** and modular

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd TextsAnnotation

# Install dependencies
pip install -r requirements.txt

# (Optional) Install development tools
pip install -r requirements-dev.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
# Database
TEXTS_DB_PATH=data/db/app.db
TEXTS_DB_DUMP_PATH=data/db/app_dump.db
TEXTS_DB_DUMP_INTERVAL_SEC=60

# Data paths
TEXTS_INTENTS_PATH=data/intents
TEXTS_ANNOTATORS_PATH=data/annotators.yaml
TEXTS_IMPORT_CSV_PATH=data/requests.csv

# Annotation settings
TEXTS_MIN_ANNOTATORS=1
TEXTS_TOP_K=5
TEXTS_MARGIN_THRESHOLD=0.1
TEXTS_PROBABILITY_THRESHOLD=0.1

# Admin
TEXTS_ADMIN_PASSWORD=admin123

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Running the Application

```bash
# Start the annotation interface
streamlit run app.py

# Access admin dashboard
# Navigate to: http://localhost:8501/admin
# Default password: admin123
```

## 👥 User Management

Edit `data/annotators.yaml` to manage users:

```yaml
annotators:
  - name: username
    password: password123
    language: ru  # or 'en', 'kz', etc.
    clusters:
      - cluster1
      - cluster2
```

**Note**: Passwords are stored in plain text. This is acceptable for trusted private networks but should be changed for public deployments.

## 📊 Features

### Performance & UX
- **🚀 Instant Imports**: Optimized CSV duplicate checks with database indexing, speeding up session starts by over 10x.
- **⚡ High-Performance Navigation**: Replaced large selectboxes with a performant "Jump to ID" input for handling thousands of texts without lag.
- **⌨️ Keyboard Shortcuts**:
    - `Enter`: Save annotation and load next text.
    - `Space`: Toggle focused intent checkbox (without page scrolling).
    - `Arrows`: Navigate smoothly between intent checkboxes.
- **🎯 Smart Focus**: 
    - Automatic focus on the first intent upon login and new text open.
    - Focus persistence during content re-renders (checks/unchecks).
- **💾 Session Persistence**: Seamless reconnect/reload support using URL query parameters for username.

## 🗄️ Database Schema

The application uses SQLite with the following main tables:

- `texts` - Text data with clusters and language
- `intents` - Intent definitions from YAML files
- `annotations` - User annotations (text_id + annotator + label)
- `candidates` - ML model predictions
- `skipped_texts` - Skipped items per annotator
- `shown_intents` - Tracking of shown intents
- `data_versions` / `model_versions` - Version tracking

**Performance**: 8 indexes optimize common queries for fast access.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models/test_models.py -v
```

**Current Coverage**: 32 tests covering models, repositories, and services.

## 🛠️ Development

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

Configuration is in `pyproject.toml`.

### Project Structure

```
TextsAnnotation/
├── app.py                      # Main Streamlit app
├── pages/
│   └── admin.py               # Admin dashboard
├── src/
│   ├── models/                # Pydantic data models
│   │   ├── annotator.py
│   │   ├── intent.py
│   │   ├── text.py
│   │   └── candidate.py
│   ├── repositories/          # Database layer
│   │   ├── base.py
│   │   ├── intent_repo.py
│   │   ├── text_repo.py
│   │   └── annotation_repo.py
│   ├── services/              # Business logic
│   │   ├── auth_service.py
│   │   ├── annotation_service.py
│   │   ├── import_service.py
│   │   └── stats_service.py
│   ├── ml/
│   │   └── model_stub.py      # ML model integration
│   └── utils/
│       ├── config.py          # Centralized settings
│       ├── database.py        # DB connection & setup
│       ├── logger.py          # Logging configuration
│       └── yaml_loader.py     # YAML utilities
├── tests/                     # Unit tests
│   ├── conftest.py            # Pytest fixtures
│   ├── test_models/
│   ├── test_repositories/
│   └── test_services/
├── data/
│   ├── intents/               # Intent YAML files
│   ├── annotators.yaml        # User configuration
│   └── db/                    # SQLite database
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── pyproject.toml            # Tool configuration
```

## 📦 Dependencies

### Production
- `streamlit` - Web UI framework
- `pydantic` - Data validation
- `pydantic-settings` - Configuration management
- `PyYAML` - YAML parsing
- `pandas` - Data processing
- `python-dotenv` - Environment management

### Development
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `mypy` - Type checking
- `flake8` - Linting

## 🔧 Configuration Reference

All settings can be configured via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `TEXTS_DB_PATH` | `data/db/app.db` | Database file path |
| `TEXTS_DB_DUMP_PATH` | `data/db/app_dump.db` | Backup database path |
| `TEXTS_DB_DUMP_INTERVAL_SEC` | `60` | Auto-backup interval |
| `TEXTS_INTENTS_PATH` | `data/intents` | Intent definitions directory |
| `TEXTS_ANNOTATORS_PATH` | `data/annotators.yaml` | Annotator config file |
| `TEXTS_IMPORT_CSV_PATH` | `data/requests.csv` | CSV import file |
| `TEXTS_MIN_ANNOTATORS` | `1` | Required annotators per text |
| `TEXTS_TOP_K` | `5` | Number of top predictions to show |
| `TEXTS_ADMIN_PASSWORD` | `admin123` | Admin dashboard password |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FILE` | `logs/app.log` | Log file path |

## 📝 Data Format

### Intent YAML Format

```yaml
intent_name:
  description: "Intent description"
  train:
    - "Example utterance 1"
    - "Example utterance 2"
  complexity: "low"  # or "medium", "high"
  cluster: "cluster_name"
```

### Import CSV Format

```csv
text,language,clusters,score_intent1,score_intent2,...
"Sample text",ru,"cluster1,cluster2",0.95,0.82,...
```

## 🚢 Deployment

### Docker (Optional)

```bash
# Build image
docker build -t texts-annotation .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e TEXTS_ADMIN_PASSWORD=your_password \
  texts-annotation
```

### Production Considerations

1. **Change passwords** in `annotators.yaml` and `.env`
2. **Set up regular backups** of the database
3. **Configure logging** for production
4. **Use HTTPS** if exposing externally
5. **Set resource limits** for Streamlit

## 🐛 Troubleshooting

### "Can't log in to admin panel"
- Default password is `admin123`
- Check `.env` file for custom `TEXTS_ADMIN_PASSWORD`

### "No texts to annotate"
- Ensure CSV file exists at `TEXTS_IMPORT_CSV_PATH`
- Check database has texts: `sqlite3 data/db/app.db "SELECT COUNT(*) FROM texts;"`

### "Intents not loading"
- Verify YAML files in `data/intents/` directory
- Check logs for YAML parsing errors

### "Tests failing"
- Ensure all dependencies installed: `pip install -r requirements-dev.txt`
- Python 3.8+ required for compatibility

## 📚 API Reference

### Services

#### AuthService
- `authenticate(username, password)` - Authenticate user
- `load_annotators()` - Load annotator configuration
- `get_annotator(username)` - Get annotator by name

#### AnnotationService
- `get_next_text(annotator, clusters, language)` - Get next unannotated text
- `save_annotations(text_id, annotator, decisions, ...)` - Save annotations
- `skip_text(text_id, annotator)` - Skip a text
- `get_progress(annotator, clusters, language)` - Get progress stats

#### StatsService
- `get_overall_stats()` - Get dashboard metrics
- `get_annotator_stats()` - Per-annotator performance
- `get_intent_quality()` - Model quality metrics
- `export_annotations(output_path)` - Export to CSV

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest`
4. Format code: `black src/ tests/`
5. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with modern Python best practices:
- Clean Architecture
- Repository Pattern
- Service Layer
- Test-Driven Development
- Type Safety with Pydantic
