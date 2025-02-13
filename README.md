# TheoremLib: Advanced Mathematics Theorem-Proof Database System

## Overview
TheoremLib is a production-grade system for processing, storing, and analyzing mathematical theorems and proofs from PDF documents. It provides automated extraction, annotation, and graph-based relationship modeling of mathematical content.

## Key Features
- Automated PDF processing with LaTeX text layer verification
- Intelligent theorem-proof pair extraction with formatting preservation
- Metadata enrichment using Cambridge Math Ontology
- Vector-enabled PostgreSQL storage with full-text search
- Theorem dependency graph construction
- RESTful API for content management and querying
- Containerized deployment with Kubernetes orchestration

## Project Structure
```
theoremlib/
├── ingest/          # PDF ingestion and validation
├── ocr/             # OCR processing using Gemini 2.0
├── extraction/      # Theorem-proof content extraction
├── annotation/      # Metadata and ontological annotation
├── database/        # Database models and operations
├── graph/           # Dependency graph construction
└── api/             # FastAPI endpoints
```

## Prerequisites
- Python 3.9+
- PostgreSQL 15+ with pgvector extension
- Docker and Kubernetes for deployment

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/theoremlib.git
cd theoremlib
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure PostgreSQL:
```sql
CREATE EXTENSION vector;
```

## Development Setup
1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Update environment variables in `.env` with your configurations

3. Start the development server:
```bash
uvicorn api.main:app --reload
```

## Docker Deployment
```bash
docker-compose up -d
```

## Kubernetes Deployment
1. Create secrets:
```bash
kubectl create secret generic theoremlib-secrets --from-env-file=.env
```

2. Apply Kubernetes manifests:
```bash
kubectl apply -f k8s/
```

## Testing
Run the test suite:
```bash
pytest
```

## API Documentation
Once running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License
MIT License

## Contributors
[Your Name] - Initial work