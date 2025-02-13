# TheoremLib: Mathematical Theorem Database System

## Overview
TheoremLib is an advanced system for processing mathematical documents, extracting theorem-proof pairs, and building a rich knowledge graph. It provides semantic search capabilities and ontological annotations based on the Cambridge Math Ontology.

### Key Features
- Automated theorem and proof extraction from PDFs
- Rich metadata annotation using mathematical ontologies
- Advanced search with vector embeddings (pgvector)
- Theorem dependency graph construction
- REST API with OpenAPI documentation
- Docker and Kubernetes deployment support

## Architecture
```
theoremlib/
├── annotation/      # Metadata enrichment
├── api/            # FastAPI endpoints
├── database/       # SQLAlchemy models
├── extraction/     # Content extraction
├── graph/          # Graph operations
├── ingest/         # PDF processing
├── ocr/            # Text extraction
├── tests/          # Test suites
└── k8s/            # Kubernetes configs
```

## Prerequisites
- Python 3.9+
- PostgreSQL 15+ with pgvector extension
- Docker and Docker Compose (for containerized deployment)
- Kubernetes cluster (for production deployment)

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/theoremlib.git
cd theoremlib
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/theoremlib"
export ONTOLOGY_PATH="/path/to/math_ontology.json"
export STORAGE_PATH="/path/to/storage"
export OCR_API_KEY="your-api-key"
```

## Development Setup

1. Initialize the database:
```bash
psql -U postgres -c "CREATE DATABASE theoremlib"
psql -U postgres -d theoremlib -c "CREATE EXTENSION vector"
```

2. Run database migrations:
```bash
alembic upgrade head
```

3. Start the development server:
```bash
uvicorn api.main:app --reload
```

## Deployment

### Docker Compose

1. Build and start services:
```bash
docker-compose up --build
```

2. Access the API at http://localhost:8000

### Kubernetes

1. Create secrets:
```bash
kubectl create secret generic theoremlib-secrets \
  --from-literal=database-url='postgresql+asyncpg://user:pass@db:5432/theoremlib' \
  --from-literal=ocr-api-key='your-api-key'
```

2. Deploy the application:
```bash
kubectl apply -f k8s/
```

## Testing

1. Run unit tests:
```bash
pytest tests/
```

2. Run integration tests:
```bash
pytest tests/test_integration.py -v
```

3. Generate coverage report:
```bash
pytest --cov=theoremlib tests/
```

### Manual Testing and Monitoring

1. Access the OpenAPI documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

2. Monitor processing status:
```bash
curl http://localhost:8000/status/{document_id}
```

3. Trigger reprocessing for a document:
```bash
curl -X POST http://localhost:8000/reprocess/{document_id}
```

## Monitoring and Logging

### Log Locations
- Application logs: `/var/log/theoremlib/app.log`
- Processing logs: `/var/log/theoremlib/processing.log`
- Error logs: `/var/log/theoremlib/error.log`

### Health Checks
Monitor service health:
```bash
curl http://localhost:8000/health
```

### Metrics
Prometheus metrics available at: http://localhost:8000/metrics

## Future Enhancements

1. **Machine Learning Improvements**
   - Enhanced theorem classification
   - Automated proof technique detection
   - Semantic similarity improvements

2. **Graph Analytics**
   - Advanced theorem clustering
   - Prerequisite path optimization
   - Community detection algorithms

3. **Performance Optimizations**
   - Batch processing improvements
   - Caching layer implementation
   - Query optimization

4. **User Interface**
   - Web-based theorem browser
   - Interactive graph visualization
   - Annotation interface

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support
For support, please open an issue or contact the maintainers at [maintainers@theoremlib.org].