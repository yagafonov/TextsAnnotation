# TextsAnnotation

Multi-label text classification annotation tool built with Streamlit. Features a clean modular architecture with comprehensive test coverage.

## Overview

TextsAnnotation is a web-based application for annotating texts with multiple intent labels. It supports:
- **Multi-annotator workflow** with cluster-based text assignment and progress tracking
- **ML model integration** for prediction candidates with configurable thresholds
- **Admin dashboard** with quality metrics, analytics, and live settings management
- **Dark theme & wide mode** with browser persistence (cookies)
- **Keyboard shortcuts** for fast annotation
- **CSV import/export** for data management

## Architecture

```
src/
├── models/          # Pydantic data models with validation
├── repositories/    # Database access layer (SQLite)
├── services/        # Business logic layer
├── ml/              # ML model integration
└── utils/           # Configuration, logging, database utilities

tests/               # 110 unit & integration tests
app.py               # Main annotation interface
pages/admin.py       # Admin dashboard
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone <your-repo-url>
cd TextsAnnotation

pip install -r requirements.txt

# (Optional) Install development tools
pip install -r requirements-dev.txt
```

### Configuration

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

All parameters can also be edited from the **Admin > Settings** tab at runtime.

```bash
# Admin dashboard: http://localhost:8501/admin
# Default password: admin123
```

### Docker Setup (Recommended)

Running via Docker ensures all environment dependencies are correctly handled:

1. **Build and start**:
   ```bash
   docker-compose up --build -d
   ```
2. **Access**:
   The app will be available at [http://localhost:8501](http://localhost:8501).

3. **Logs**:
   ```bash
   docker-compose logs -f
   ```

## User Management

Edit `data/annotators.yaml` to manage users:

```yaml
annotators:
  - name: username
    password: password123
    language: Русский      # display language name
    cluster:
      - cluster_name_1
      - cluster_name_2
```

**Note**: Passwords are stored in plain text. This is acceptable for trusted private networks but should be changed for public deployments.

## Features

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Save annotation |
| `Esc` | Skip text / Return skipped text |
| `⌘/Ctrl + ←` | Previous text |
| `⌘/Ctrl + →` | Next text |
| `Space` | Toggle focused intent checkbox |
| `↑ ↓ ← →` | Navigate between intent checkboxes |

Shortcut hints are automatically hidden on mobile devices.

### Annotation Interface
- **Intent filtering** — filter texts by intent or cluster from the sidebar dropdown
- **Skip / Unskip** — mark texts to revisit later; skipped texts shown with ⏭️ marker
- **Navigation dropdown** — searchable text list with status markers (✅ annotated, ⏭️ skipped)
- **Smart focus** — auto-focus on first intent checkbox, preserved across rerenders

### UI & Persistence
- **Dark theme** toggle (🌙/☀️) saved in browser cookies
- **Wide / Centered mode** toggle (↔️/↕️) saved in browser cookies
- **Session persistence** via URL query parameters

### Admin Dashboard Tabs

| Tab | Description |
|-----|-------------|
| Overview | Total texts, annotations, annotators, completion rate |
| Annotators | Per-annotator stats and activity charts |
| Quality | Model quality metrics, top-1 precision by intent |
| Texts | Detailed text overview with filters, bulk assign/re-assign actions |
| Clusters | Per-cluster annotation progress |
| Export | CSV export of annotations |
| Import | CSV import with duplicate detection and unassigned text warnings |
| Settings | Live-editable app parameters + UI toggles |

### Settings (Admin)
- **Show model confidence** — toggle percentage display next to intent names
- **All .env parameters** editable from the UI with Save/Load buttons
- Thresholds displayed as percentages for clarity

## Database Schema

SQLite database with the following main tables:

- `texts` — text data with cluster, language, `is_skipped` flag
- `intents` — intent definitions loaded from YAML files
- `annotations` — user annotations (text_id + annotator + label + decision)
- `candidates` — ML model predictions with probabilities
- `skipped_texts` — skip records per annotator
- `settings` — app-level key-value settings (e.g. show_confidence)
- `data_versions` / `model_versions` — version tracking

Performance: 8+ indexes optimize common queries.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_services/test_stats_service.py -v
```

**Current**: 110 tests covering models, repositories, services, config, and integration workflows.

## Configuration Reference

All settings can be configured via `.env` file or the Admin Settings tab:

| Variable | Default | Description |
|----------|---------|-------------|
| `TEXTS_DB_PATH` | `data/db/app.db` | Database file path |
| `TEXTS_DB_DUMP_PATH` | `data/dumps/backup.sql` | Backup database path |
| `TEXTS_DB_DUMP_INTERVAL_SEC` | `60` | Auto-backup interval (seconds, min 10) |
| `TEXTS_TOP_K` | `5` | Number of top predictions to show (1–20) |
| `TEXTS_MARGIN_THRESHOLD` | `0.1` | Min probability difference between candidates (0–1) |
| `TEXTS_PROBABILITY_THRESHOLD` | `0.1` | Min probability for candidate import (0–1) |
| `ANNOTATORS_INTENTS_CONFIDENCE_THRESHOLD` | `0.4` | Min confidence to show intent to annotator (0–1) |
| `TEXTS_INTENTS_PATH` | `data/intents` | Intent YAML definitions directory |
| `TEXTS_ANNOTATORS_PATH` | `data/annotators.yaml` | Annotator config file |
| `TEXTS_IMPORT_CSV_PATH` | `data/requests.csv` | Default CSV import file |
| `TEXTS_ADMIN_PASSWORD` | `admin123` | Admin dashboard password |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `LOG_FILE` | `logs/app.log` | Log file path |

## Data Formats

### Intent YAML

```yaml
intent_name:
  description: "Intent description"
  train:
    - "Example utterance 1"
    - "Example utterance 2"
  complexity: "low"  # or "medium", "high"
  cluster: "cluster_name"
```

### Import CSV

```csv
text,language,clusters,score_intent1,score_intent2,...
"Sample text",ru,"cluster1,cluster2",0.95,0.82,...
```

## Project Structure

```
TextsAnnotation/
├── app.py                      # Main Streamlit annotation app
├── pages/
│   └── admin.py                # Admin dashboard
├── src/
│   ├── models/                 # Pydantic data models
│   │   ├── annotator.py
│   │   ├── intent.py
│   │   ├── text.py
│   │   └── candidate.py
│   ├── repositories/           # Database layer
│   │   ├── base.py
│   │   ├── intent_repo.py
│   │   ├── text_repo.py
│   │   └── annotation_repo.py
│   ├── services/               # Business logic
│   │   ├── auth_service.py
│   │   ├── annotation_service.py
│   │   ├── import_service.py
│   │   └── stats_service.py
│   ├── ml/
│   │   └── model_stub.py       # ML model integration
│   └── utils/
│       ├── config.py           # Centralized settings (Pydantic)
│       ├── database.py         # DB connection & migrations
│       ├── logger.py           # Logging configuration
│       └── yaml_loader.py      # YAML utilities
├── tests/                      # Unit & integration tests
│   ├── conftest.py
│   ├── test_models/
│   ├── test_repositories/
│   ├── test_services/
│   ├── test_integration/
│   ├── test_ml/
│   └── test_utils/
├── data/
│   ├── intents/                # Intent YAML files
│   ├── annotators.yaml         # User configuration
│   └── db/                     # SQLite database
├── scripts/
│   └── clear_random_annotations.py
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Dependencies

### Production
- `streamlit` — Web UI framework
- `pydantic` / `pydantic-settings` — Data validation & configuration
- `PyYAML` — YAML parsing
- `pandas` — Data processing
- `python-dotenv` — Environment management
- `extra-streamlit-components` — Cookie manager for persistence

### Development
- `pytest` / `pytest-cov` — Testing & coverage
- `black` — Code formatting
- `mypy` — Type checking
- `flake8` — Linting

## Troubleshooting

### "Can't log in to admin panel"
- Default password is `admin123`
- Check `.env` file for custom `TEXTS_ADMIN_PASSWORD`

### "No texts to annotate"
- Import a CSV via Admin > Import tab
- Check database: `sqlite3 data/db/app.db "SELECT COUNT(*) FROM texts;"`

### "Intents not loading"
- Verify YAML files exist in `data/intents/` directory
- Check `logs/app.log` for YAML parsing errors

### "Tests failing"
- Ensure all dependencies installed: `pip install -r requirements-dev.txt`
- Python 3.10+ required
