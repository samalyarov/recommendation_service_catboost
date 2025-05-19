# Recommendation Service API
![Uploading image.png…]()

A FastAPI-based recommendation service that provides personalized post recommendations to users based on their characteristics and interaction history. The service implements A/B testing functionality (currently works with 2 Catboost models being compared to one another). 

The crux of the project is based on the final project from [Karpov Courses Machine Learning Engineer](https://karpov.courses/ml-start) course, credit goes to its authors for creating an amazing learning opportunity.

## Features

- Personalized post recommendations
- A/B testing support
- Real-time predictions
- Efficient batch data loading
- Comprehensive error handling
- Detailed logging
- Docker support for easy deployment

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Required Python packages (see `requirements.txt`)
- Docker and Docker Compose (for containerized deployment)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```env
DATABASE=your_database_name
USER=your_database_user
PASSWORD=your_database_password
HOST=your_database_host
PORT=your_database_port
SALT=your_salt_value
```

### Docker Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a `.env` file with the required environment variables (same as above)

3. Build and start the containers:
```bash
docker-compose up --build
```

## Usage

### Local Usage

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Access the API documentation at `http://localhost:8000/docs`

3. Make a recommendation request:
```bash
curl -X GET "http://localhost:8000/post/recommendations/?id=123&limit=5"
```

### Docker Usage

The application will be available at `http://localhost:8000` after starting the containers.

## API Endpoints

### GET /post/recommendations/

Returns personalized post recommendations for a user.

**Query Parameters:**
- `id` (int): User identifier
- `limit` (int, optional): Maximum number of recommendations to return (default: 5)

**Response:**
```json
{
    "exp_group": "control",
    "recommendations": [
        {
            "id": 123,
            "text": "Post content",
            "topic": "Topic category"
        }
    ]
}
```

## Project Structure

```
.
├── main.py              # Main FastAPI application
├── requirements.txt     # Project dependencies
├── .env                # Environment variables (not tracked in git)
├── .gitignore         # Git ignore rules
├── pytest.ini         # Pytest configuration
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── README.md          # Project documentation
├── models/            # Trained model files
│   ├── model_control
│   └── model_test
├── notebooks/         # Jupyter notebooks
│   └── model_training.ipynb
└── tests/             # Test directory
    ├── conftest.py    # Shared test fixtures
    ├── unit/          # Unit tests
    │   └── test_utils.py
    └── integration/   # Integration tests
        └── test_api.py
```

## Development

- The service uses CatBoost models for predictions
- A/B testing is implemented using a hash-based user assignment
- Data is loaded in batches for memory efficiency
- Comprehensive error handling and logging are implemented

## Testing

Run the test suite:
```bash
pytest
```

Run tests with coverage report:
```bash
pytest --cov
```

## Docker Commands

Build the Docker image:
```bash
docker build -t recommendation-service .
```

Run the container:
```bash
docker run -p 8000:8000 recommendation-service
```

Stop the containers:
```bash
docker-compose down
```

View logs:
```bash
docker-compose logs -f
```

## Author

Sam Maliarov
