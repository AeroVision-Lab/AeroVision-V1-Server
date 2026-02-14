# Aerovision-V1-Server

AI-powered aviation photography review API built with FastAPI.

## Features

- **Quality Assessment**: Sharpness, exposure, composition, noise, color evaluation
- **Aircraft Classification**: Aircraft type identification with top-k predictions
- **Airline Recognition**: Airline livery identification
- **Registration OCR**: Aircraft registration number recognition with bounding boxes
- **Aggregated Review**: Combined review endpoint that calls all atomic services
- **Batch Processing**: All endpoints support batch operations (up to 50 images)

## Project Structure

```
Aerovision-V1-Server/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py            # Router aggregation
│   │   ├── deps.py                # Dependencies & request stats
│   │   └── routes/
│   │       ├── health.py          # Health check & statistics
│   │       ├── quality.py         # Quality assessment API
│   │       ├── aircraft.py        # Aircraft classification API
│   │       ├── airline.py         # Airline recognition API
│   │       ├── registration.py    # Registration OCR API
│   │       └── review.py          # Aggregated review API
│   ├── services/
│   │   ├── base.py                # Service base class
│   │   ├── quality_service.py
│   │   ├── aircraft_service.py
│   │   ├── airline_service.py
│   │   ├── registration_service.py
│   │   └── review_service.py      # Aggregated service
│   ├── schemas/
│   │   ├── common.py              # Common models
│   │   ├── quality.py
│   │   ├── aircraft.py
│   │   ├── airline.py
│   │   ├── registration.py
│   │   └── review.py
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   ├── logging.py             # Logging configuration
│   │   └── exceptions.py          # Custom exceptions
│   └── inference/
│       ├── factory.py             # Inference model factory
│       └── wrappers.py            # Result wrappers
├── tests/
│   ├── conftest.py                # Pytest fixtures
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── fixtures/
│       └── images/                # Test images
└── requirements/
    ├── base.txt
    ├── test.txt
    └── dev.txt
```

## Quick Start

### Installation

```bash
cd Aerovision-V1-Server

# Install dependencies
pip install -r requirements/base.txt

# Install test dependencies
pip install -r requirements/test.txt
```

### Configuration

Create a `.env` file or set environment variables:

```bash
# Model Configuration
MODEL_DIR=models              # Model directory
DEVICE=cuda                   # Inference device (cuda/cpu)
PRELOAD_MODELS=true           # Preload models on startup

# OCR Configuration
OCR_MODE=local                # OCR mode (local/api)
OCR_LANG=ch                   # OCR language
USE_ANGLE_CLS=true            # Use angle classifier

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
DEBUG=true
```

### Running the Server

```bash
# Development mode
python -m app.main

# Or using uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Historical Record APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/history/push` | Push historical audit records to vector database |
| `GET` | `/api/v1/history/stats` | Get vector database statistics |

### Atomic APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/quality` | Quality assessment |
| `POST` | `/api/v1/quality/batch` | Batch quality assessment |
| `POST` | `/api/v1/aircraft` | Aircraft classification |
| `POST` | `/api/v1/aircraft/batch` | Batch aircraft classification |
| `POST` | `/api/v1/airline` | Airline recognition |
| `POST` | `/api/v1/airline/batch` | Batch airline recognition |
| `POST` | `/api/v1/registration` | Registration OCR |
| `POST` | `/api/v1/registration/batch` | Batch registration OCR |

### Aggregated API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/review` | Full review (all services) |
| `POST` | `/api/v1/review/batch` | Batch full review |

### System APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/stats` | Request statistics |

## Usage Examples

### Push Historical Records

```bash
curl -X POST "http://localhost:8000/api/v1/history/push" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "id": "record_001",
        "image_url": "https://example.com/image1.jpg",
        "aircraft_type": "A320",
        "airline": "CEA",
        "aircraft_confidence": 0.85,
        "airline_confidence": 0.75,
        "timestamp": "2025-01-28T10:00:00Z",
        "metadata": {
          "image_data": "data:image/jpeg;base64,/9j/4AAQ..."
        }
      }
    ]
  }'
```

**Response**:
```json
{
  "success": true,
  "total": 1,
  "added": 1,
  "failed": 0
}
```

### Get Historical Record Statistics

```bash
curl -X GET "http://localhost:8000/api/v1/history/stats"
```

**Response**:
```json
{
  "available": true,
  "total_records": 1000,
  "aircraft_types": {
    "A320": 350,
    "B738": 450,
    "A321": 200
  },
  "airlines": {
    "CEA": 500,
    "CSN": 300,
    "CAC": 200
  }
}
```

### Quality Assessment

```bash
curl -X POST "http://localhost:8000/api/v1/quality" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

**Response**:
```json
{
  "success": true,
  "pass": true,
  "score": 0.85,
  "details": {
    "sharpness": 0.90,
    "exposure": 0.80,
    "composition": 0.85,
    "noise": 0.88,
    "color": 0.82
  },
  "meta": {
    "processing_time_ms": 123.45,
    "timestamp": "2025-01-28T10:00:00Z"
  }
}
```

### Aircraft Classification

```bash
curl -X POST "http://localhost:8000/api/v1/aircraft?top_k=5" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

**Response**:
```json
{
  "success": true,
  "top1": {
    "class": "A320",
    "confidence": 0.85
  },
  "top_k": 5,
  "predictions": [
    {"class": "A320", "confidence": 0.85},
    {"class": "B738", "confidence": 0.10},
    {"class": "A321", "confidence": 0.05}
  ],
  "meta": {
    "processing_time_ms": 89.2,
    "timestamp": "2025-01-28T10:00:00Z"
  }
}
```

### Registration OCR

```bash
curl -X POST "http://localhost:8000/api/v1/registration" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

**Response**:
```json
{
  "success": true,
  "registration": "B-1234",
  "confidence": 0.95,
  "raw_text": "B-1234",
  "all_matches": [
    {"text": "B-1234", "confidence": 0.95}
  ],
  "yolo_boxes": [
    {
      "class_id": 0,
      "x_center": 0.5,
      "y_center": 0.3,
      "width": 0.2,
      "height": 0.1,
      "text": "B-1234",
      "confidence": 0.95
    }
  ],
  "meta": {
    "processing_time_ms": 156.7,
    "timestamp": "2025-01-28T10:00:00Z"
  }
}
```

### Aggregated Review

```bash
curl -X POST "http://localhost:8000/api/v1/review?include_quality=true&include_aircraft=true&include_airline=true&include_registration=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

**Response**:
```json
{
  "success": true,
  "quality": {
    "score": 0.85,
    "pass": true,
    "details": {
      "sharpness": 0.90,
      "exposure": 0.80,
      "composition": 0.85,
      "noise": 0.88,
      "color": 0.82
    }
  },
  "aircraft": {
    "type_code": "A320",
    "confidence": 0.85
  },
  "airline": {
    "airline_code": "CEA",
    "confidence": 0.75
  },
  "registration": {
    "registration": "B-1234",
    "confidence": 0.95,
    "clarity": 0.95
  },
  "meta": {
    "processing_time_ms": 369.1,
    "timestamp": "2025-01-28T10:00:00Z"
  }
}
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/api/v1/quality/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "data:image/jpeg;base64,/9j/4AAQ...",
      "data:image/jpeg;base64,/9j/4AAQ..."
    ]
  }'
```

**Response**:
```json
{
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "index": 0,
      "success": true,
      "data": {
        "pass": true,
        "score": 0.85,
        "details": {...}
      },
      "error": null
    },
    {
      "index": 1,
      "success": true,
      "data": {
        "pass": true,
        "score": 0.78,
        "details": {...}
      },
      "error": null
    }
  ]
}
```

## Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/schemas/test_schemas.py -v
```

### Test Coverage

Current coverage: **52%** (418/865 lines)

## Architecture

### Inference Layer

The `app/inference/factory.py` provides a unified interface for loading models from `aerovision_inference`:

- `InferenceFactory.get_aircraft_classifier()`
- `InferenceFactory.get_airline_classifier()`
- `InferenceFactory.get_registration_ocr()`
- `InferenceFactory.get_quality_assessor()`

Models are lazy-loaded on first request and cached for subsequent use.

### Service Layer

Each atomic API has a corresponding service class:

- `QualityService` - Image quality assessment
- `AircraftService` - Aircraft type classification
- `AirlineService` - Airline recognition
- `RegistrationService` - Registration OCR
- `ReviewService` - Aggregated review (calls all services)

### Request/Response Format

**Single Image Request**:
```json
{
  "image": "base64 or URL"
}
```

**Batch Request**:
```json
{
  "images": ["base64 or URL", "..."]
}
```

**Response Format**:
```json
{
  "success": true,
  "data": { /* result */ },
  "meta": {
    "processing_time_ms": 123,
    "timestamp": "2025-01-28T10:00:00Z"
  }
}
```

## Dependencies

### Core
- `fastapi>=0.104.1` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.5.0` - Data validation
- `pydantic-settings>=2.1.0` - Configuration management
- `httpx>=0.25.0` - HTTP client
- `pillow>=10.1.0` - Image processing

### Test
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async support
- `pytest-cov>=4.1.0` - Coverage reporting

### Inference
- `aerovision_inference` - Inference backend (local package)

## Compatibility

This server is designed to work with `aerovision_inference` package. The inference layer provides:

- Lazy-loading of models
- Graceful degradation when inference package is not available
- Result wrapping for consistent API responses

## License

See project root for license information.
