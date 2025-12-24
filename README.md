# LLM Training Pipeline with arXiv Data

End-to-end ML pipeline for training Large Language Models using arXiv research papers.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Collection Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ arXiv API   │  │   Scrapy    │  │   Custom Collectors     │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer (MinIO/S3)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Raw Data   │  │  Processed  │  │      Curated Data       │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Processing Layer (PySpark)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Dedup Job   │  │ Normalize   │  │   Content Moderation    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tokenization Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │SentencePiece│  │     BPE     │  │  Distributed Tokenize   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Training Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   PyTorch   │  │    W&B      │  │   Distributed Train     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/thienhb/train-llm-with-arxiv-data.git
cd train-llm-with-arxiv-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Start Infrastructure

```bash
# Start MinIO, Spark, and Jupyter
docker-compose up -d

# Verify services
docker-compose ps
```

### Run the Pipeline

```bash
# 1. Collect papers from arXiv (limit to 100 for testing)
arxiv-llm collect --limit 100 --category cs.LG

# 2. Process the collected data
arxiv-llm process --stage all

# 3. Train tokenizer
arxiv-llm tokenize --vocab-size 32000

# 4. Start training
arxiv-llm train --config config/train_config.yaml
```

## 📁 Project Structure

```
train-llm-with-arxiv-data/
├── src/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point
│   ├── collectors/               # Data collection
│   │   ├── arxiv_client.py
│   │   ├── ingestion.py
│   │   └── scrapy_crawler/
│   ├── storage/                  # Storage layer
│   │   ├── minio_client.py
│   │   └── schemas.py
│   ├── processing/               # Spark jobs
│   │   ├── deduplication.py
│   │   ├── normalization.py
│   │   ├── content_moderation.py
│   │   └── pipeline.py
│   ├── tokenization/             # Tokenizer training
│   │   ├── trainer.py
│   │   └── spark_tokenizer.py
│   └── training/                 # ML training
│       ├── model.py
│       ├── trainer.py
│       └── wandb_integration.py
├── config/
│   └── settings.py
├── k8s/                          # Kubernetes manifests
├── kubeflow/                     # Kubeflow pipelines
├── tests/
├── notebooks/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `MINIO_ENDPOINT` | MinIO server endpoint | `localhost:9000` |
| `ARXIV_CATEGORIES` | arXiv categories to collect | `cs.LG,cs.CL,cs.AI` |
| `VOCAB_SIZE` | Tokenizer vocabulary size | `32000` |
| `WANDB_PROJECT` | W&B project name | `arxiv-llm-training` |

## 📊 Monitoring

- **MinIO Console**: http://localhost:9001
- **Spark Master UI**: http://localhost:8080
- **JupyterLab**: http://localhost:8888
- **W&B Dashboard**: https://wandb.ai

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_arxiv_client.py -v
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
