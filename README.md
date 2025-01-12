# Simple RAG Evaluator - S-RAGE

A web-based application for evaluating RAG (Retrieval-Augmented Generation) systems by comparing their responses against ground truth data.

## Features

- Multiple LLM provider support (Azure OpenAI, OpenAI, AWS Bedrock)
- Automated evaluation metrics:
  - Cosine similarity using sentence embeddings
  - ROUGE scores for response evaluation
  - AI-based evaluation using GPT-4
- Web interface for viewing results
- SQLite database for storing responses and evaluations
- Configurable API endpoints and response mapping

## Prerequisites

- Python 3.7+
- SQLite3
- A RAG system with an accessible API endpoint
- Access to one of the supported LLM providers:
  - Azure OpenAI
  - OpenAI
  - AWS Bedrock

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-evaluator
```
    
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the application:

## Configuration

### 1. Environment Variables

Create a `.env` file in the root directory:

```bash
AZURE_API_TYPE=azure
AZURE_API_VERSION=2023-05-15
AZURE_API_BASE=https://your-azure-endpoint.openai.azure.com
AZURE_API_KEY=your-azure-api-key
AZURE_DEPLOYMENT_NAME=your-gpt4-deployment
OPENAI_API_KEY=your-openai-api-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
```
### 2. LLM Configuration

Create or modify `llm.config.json` with your settings:

```json
{
    "default_provider": "azure",
    "azure": {
        "api_type": "azure",
        "api_version": "2023-05-15",
        "api_base": "https://your-azure-endpoint.openai.azure.com",
        "api_key": "your-azure-api-key",
        "deployment_name": "your-gpt4-deployment"
    }
}
```
### 3. Ground Truth Data

Create a CSV file with the following format:

```csv
Question,Answer
"What is RAG?","Retrieval-Augmented Generation..."
```

### 4. RAG System Configuration

Create a `config.json` file to configure your RAG system's API endpoint and response mapping:

```json
{
    "api_endpoint": "http://localhost:8000/v1/chat/completions",
    "api_key": null,
    "response_mapping": {
        "path": ["choices", 0, "message", "content"],
        "is_array": false
    },
    "chunks_mapping": {
        "path": ["choices", 0, "sources"],
        "is_array": true,
        "text_field": "text"
    },
    "ground_truth_path": "data/ground_truth.csv"
}
```

#### Configuration Options:

- `api_endpoint`: URL of your RAG system's API endpoint
- `api_key`: API key for RAG system (if required)
- `response_mapping`: Defines how to extract the response from the API JSON
  - `path`: Array of keys/indices to navigate the JSON response
  - `is_array`: Whether the response is an array that needs to be joined
- `chunks_mapping`: Defines how to extract the chunks from the API JSON
  - `path`: Array of keys/indices to navigate to the chunks
  - `is_array`: Whether chunks are in an array format
  - `text_field`: Field name containing the chunk text
- `ground_truth_path`: Path to your ground truth CSV file

#### Example Configurations:

1. For PrivateGPT:
```json
{
    "api_endpoint": "http://localhost:8000/v1/chat/completions",
    "api_key": null,
    "response_mapping": {
        "path": ["choices", 0, "message", "content"],
        "is_array": false
    },
    "chunks_mapping": {
        "path": ["choices", 0, "sources"],
        "is_array": true,
        "text_field": "text"
    }
}
```

2. For a Custom RAG System:
```json
{
    "api_endpoint": "http://your-rag-system/api/query",
    "api_key": "your-api-key",
    "response_mapping": {
        "path": ["data", "answer"],
        "is_array": false
    },
    "chunks_mapping": {
        "path": ["data", "context"],
        "is_array": true,
        "text_field": "content"
    }
}
```

## Project Structure
```
rag-evaluator/
├── main.py # Core application logic
├── server.py # Flask web server
├── config.json # RAG system configuration
├── llm.config.json # LLM provider configuration
├── .env # Environment variables
├── requirements.txt # Python dependencies
│
├── evaluation/ # Evaluation modules
│ ├── metrics.py # Evaluation metrics implementation
│ ├── evaluator.py # Evaluation manager
│ └── llm_provider.py # LLM provider interface
│
├── static/ # Frontend files
│ ├── index.html # Main webpage
│ ├── styles.css # CSS styles
│ └── app.js # Frontend JavaScript
│
└── data/ # Data directory
└── ground_truth.csv # Ground truth questions and answers
```

Each file's purpose:
- `main.py`: Contains core RAG evaluation logic and database operations
- `server.py`: Flask server implementation for web interface and API endpoints
- `config.json`: RAG system configuration (API endpoints, response mapping)
- `llm.config.json`: LLM provider settings (Azure, OpenAI, Bedrock)
- `.env`: Environment variables for API keys and credentials
- `evaluation/`: Directory containing all evaluation-related modules
- `static/`: Frontend assets and interface files
- `data/`: Directory for data files including ground truth CSV

## Running the Application

1. Start the web server:
```bash
python server.py
```

2. Access the web interface:
   - Open your browser and navigate to `http://localhost:5000`
   - The interface will automatically load and display ground truth questions and their evaluations

## Evaluation Metrics

### Response Evaluation
- **Factual Accuracy**: Uses sentence embeddings to measure semantic similarity
- **ROUGE Scores**: Measures overlap between generated and ground truth responses
- **AI Evaluation**: Uses LLM to evaluate completeness, accuracy, relevance, clarity, and conciseness

### Chunks Evaluation
- **Relevance**: ROUGE scores and cosine similarity for each chunk
- **Completeness**: AI-based evaluation of chunk coverage

## API Endpoints

- `GET /`: Main web interface
- `GET /api/results`: Get all evaluation results
- `POST /api/evaluate/<ground_truth_id>`: Evaluate specific response

## Database Schema

### Ground Truth Table
- `id`: Primary key
- `question`: Question text
- `answer`: Ground truth answer

### RAG Responses Table
- `id`: Primary key
- `ground_truth_id`: Foreign key to ground truth
- `response`: RAG system response
- `chunks`: Retrieved chunks (JSON)
- `evaluation`: Evaluation results (JSON)
- `timestamp`: Response timestamp


## Environment Setup

1. Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

2. Update the `.env` file with your API keys and configuration:

```ini
# Azure OpenAI
AZURE_API_TYPE=azure
AZURE_API_VERSION=2023-05-15
AZURE_API_BASE=your-azure-endpoint
AZURE_API_KEY=your-azure-key
AZURE_DEPLOYMENT_NAME=your-deployment-name
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Sentence-BERT
SBERT_MODEL_NAME=all-MiniLM-L6-v2

# AWS Bedrock
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_REGION=your-aws-region
```

### Environment Variables Description

#### Azure OpenAI
- `AZURE_API_TYPE`: Set to "azure" for Azure OpenAI
- `AZURE_API_VERSION`: API version (e.g., "2023-05-15")
- `AZURE_API_BASE`: Your Azure OpenAI endpoint URL
- `AZURE_API_KEY`: Your Azure OpenAI API key
- `AZURE_DEPLOYMENT_NAME`: Your GPT model deployment name
- `AZURE_EMBEDDING_DEPLOYMENT`: Your embedding model deployment name

#### OpenAI
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_EMBEDDING_MODEL`: Embedding model to use (e.g., "text-embedding-3-small")

#### Sentence-BERT
- `SBERT_MODEL_NAME`: Name of the Sentence-BERT model to use (e.g., "all-MiniLM-L6-v2")

#### AWS Bedrock
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_REGION`: AWS region (e.g., "us-east-1")

3. The application will automatically load these environment variables when starting up.

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT
