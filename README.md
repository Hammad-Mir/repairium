# LLMware Library API

A production-ready FastAPI application for managing LLMware libraries, document ingestion, and vector embeddings using ChromaDB.

## Features

- **Library Management**: Create, list, and delete document libraries
- **File Ingestion**: Add documents to libraries with metadata and blob URI support
- **Vector Embeddings**: Generate embeddings using OpenAI models
- **Persistent Storage**: ChromaDB for vector storage, SQLite for metadata
- **Dockerized**: Complete Docker Compose setup with health checks
- **Production Ready**: Proper error handling, logging, and API documentation

## Architecture

```
┌─────────────────┐
│   FastAPI App   │
│   (Port 8000)   │
└────────┬────────┘
         │
         ├─────────┐
         │         │
    ┌────▼───┐  ┌─▼──────────┐
    │ SQLite │  │  ChromaDB  │
    │(Local) │  │ (Port 8001)│
    └────────┘  └────────────┘
```

## Prerequisites

- Docker & Docker Compose
- OpenAI API Key

## Quick Start

### 1. Clone and Setup

```bash
# Create project directory
mkdir llmware-api && cd llmware-api

# Create all necessary files (main.py, Dockerfile, docker-compose.yml, etc.)
# Copy the provided code files

# Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Environment Configuration

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Start Services

```bash
# Build and start all services
docker-compose up -d --build

# Check logs
docker-compose logs -f

# Check health
curl http://localhost:8000/health
```

### 4. Access Services

- **FastAPI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ChromaDB**: http://localhost:8001

## API Endpoints

### Health Check

```bash
GET /health
```

### Library Management

#### Create Library
```bash
POST /libraries
Content-Type: application/json

{
  "name": "my_library"
}
```

#### Get All Libraries
```bash
GET /libraries
```

#### Get Specific Library
```bash
GET /libraries/{library_name}
```

#### Delete Library
```bash
DELETE /libraries/{library_name}
```

### File Operations

#### Add File to Library
```bash
POST /libraries/files
Content-Type: application/json

{
  "library_name": "my_library",
  "blob_uri": "/app/docs/document.pdf",
  "metadata": {
    "filename": "document.pdf",
    "content_type": "application/pdf",
    "size": 123456,
    "custom_metadata": {
      "author": "John Doe",
      "date": "2024-01-01"
    }
  }
}
```

**Note**: The `blob_uri` should point to:
- A local file path accessible in the container (e.g., `/app/docs/document.pdf`)
- A mounted volume path
- An external storage URI (S3, MinIO, etc.)

#### Create Embeddings
```bash
POST /libraries/{library_name}/embed
Content-Type: application/json

{
  "library_name": "my_library",
  "embedding_model": "text-embedding-3-small"
}
```

## Usage Examples

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Create library
response = requests.post(
    f"{BASE_URL}/libraries",
    json={"name": "technical_docs"}
)
print(response.json())

# Add file
response = requests.post(
    f"{BASE_URL}/libraries/files",
    json={
        "library_name": "technical_docs",
        "blob_uri": "/app/docs/manual.pdf",
        "metadata": {
            "filename": "manual.pdf",
            "content_type": "application/pdf"
        }
    }
)
print(response.json())

# Create embeddings
response = requests.post(
    f"{BASE_URL}/libraries/technical_docs/embed",
    json={"embedding_model": "text-embedding-3-small"}
)
print(response.json())

# List all libraries
response = requests.get(f"{BASE_URL}/libraries")
print(response.json())
```

### cURL Examples

```bash
# Create library
curl -X POST "http://localhost:8000/libraries" \\
  -H "Content-Type: application/json" \\
  -d '{"name": "my_docs"}'

# Add file
curl -X POST "http://localhost:8000/libraries/files" \\
  -H "Content-Type: application/json" \\
  -d '{
    "library_name": "my_docs",
    "blob_uri": "/app/docs/document.pdf",
    "metadata": {
      "filename": "document.pdf",
      "content_type": "application/pdf"
    }
  }'

# Get all libraries
curl -X GET "http://localhost:8000/libraries"

# Delete library
curl -X DELETE "http://localhost:8000/libraries/my_docs"
```

## File Storage Options

### Option 1: Local Mounted Volume (Recommended)
```yaml
# In docker-compose.yml
volumes:
  - ./docs:/app/docs
```

Then use `/app/docs/your-file.pdf` as `blob_uri`

### Option 2: External Storage (S3/MinIO)
```python
# First upload to your storage
blob_uri = "s3://bucket/path/to/file.pdf"
# Or
blob_uri = "https://minio.example.com/bucket/file.pdf"

# Then add to library
requests.post(
    f"{BASE_URL}/libraries/files",
    json={
        "library_name": "my_library",
        "blob_uri": blob_uri,
        "metadata": {...}
    }
)
```

## Development

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-key-here

# Run locally
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Project Structure

```
.
├── main.py                 # FastAPI application
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (git-ignored)
├── .env.example            # Environment template
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git exclusions
├── docs/                   # Document storage (mounted)
├── llmware_data/           # LLMware data (persisted)
└── chromadb_data/          # ChromaDB data (persisted)
```

## Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f chromadb

# Restart service
docker-compose restart api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build
```

## Monitoring & Debugging

### Check Service Health
```bash
# API health
curl http://localhost:8000/health

# ChromaDB health
curl http://localhost:8001/api/v1/heartbeat
```

### Access Logs
```bash
# API logs
docker-compose logs -f api

# ChromaDB logs
docker-compose logs -f chromadb
```

### Enter Container
```bash
# Access API container
docker exec -it llmware-api bash

# Check data directories
ls -la /app/llmware_data
ls -la /app/chromadb_data
```

## Troubleshooting

### Issue: ChromaDB connection failed
```bash
# Check if ChromaDB is running
docker-compose ps

# Restart ChromaDB
docker-compose restart chromadb
```

### Issue: Permission denied on data directories
```bash
# Fix permissions
sudo chown -R $USER:$USER llmware_data chromadb_data
```

### Issue: Out of disk space
```bash
# Clean up Docker
docker system prune -a --volumes

# Remove old data
rm -rf llmware_data/* chromadb_data/*
```

## Production Deployment

For production deployment:

1. **Use environment-specific configurations**
   ```bash
   # Create .env.production
   OPENAI_API_KEY=prod-key
   ```

2. **Enable HTTPS**
   - Add nginx reverse proxy
   - Configure SSL certificates

3. **Scale services**
   ```yaml
   # In docker-compose.yml
   api:
     deploy:
       replicas: 3
   ```

4. **Use external databases**
   - PostgreSQL instead of SQLite
   - Managed ChromaDB or Qdrant

5. **Add monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Log aggregation (ELK stack)

## API Documentation

Full interactive API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT

## Support

For issues and questions:
- Check logs: `docker-compose logs -f`
- Review API docs: http://localhost:8000/docs
- Validate environment: Check `.env` file