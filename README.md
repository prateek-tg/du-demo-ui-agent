# Telecom Assistant System - Multi-Agent Architecture

A sophisticated multi-agent orchestration system for telecom service assistance that intelligently combines intent classification, data retrieval, and UI design generation capabilities with comprehensive conversation memory.

## 📋 Overview

This system uses an advanced multi-agent architecture to process user queries in telecom services:
- **Intent Classification Agent**: Uses OpenAI GPT to understand user intent and handle conversational responses
- **Data Retrieval Agent**: Fetches relevant data from external APIs with intelligent query mapping
- **Design Agent**: Retrieves UI/UX design data for different content types and intents
- **Orchestrator**: Central coordination system managing all agent interactions and conversation history
- **FastAPI Interface**: RESTful API with conversation memory and comprehensive monitoring

## 🏗️ System Architecture

### Core Components

1. **Intent Classification Agent** (`intent_classification_agent.py`)
   - Uses OpenAI GPT-4o-mini for natural language understanding
   - Classifies user queries into predefined intent categories
   - Handles ambiguous queries and inappropriate language
   - Supports conversational responses for non-intent queries

2. **Data Retrieval Agent** (`data_retrieval_agent.py`)
   - Retrieves data from external telecom APIs
   - Maps intents to optimized API queries
   - Handles API failures gracefully
   - Maintains conversation memory

3. **Design Agent** (`design_agent.py`)
   - Retrieves UI/UX design data from external services
   - Provides design templates and styling information
   - Supports different content types and intents

4. **Orchestrator System** (`orchestrator.py`)
   - Central coordination hub for all agent interactions
   - Manages complete query processing pipeline with conversation tracking
   - Maintains memory of last 5 conversations with categorization
   - Handles error scenarios and intelligent response assembly
   - Provides conversation history with intent classification, API calls, and design requests

5. **FastAPI Application** (`main.py`)
   - RESTful API endpoints with comprehensive conversation memory
   - Real-time health monitoring and detailed system information
   - Persistent conversation history with automatic categorization
   - Memory management with 5-conversation limit and FIFO behavior
   - CORS-enabled for web applications with security features

## 🎯 Supported Intents

The system can classify and process the following intent categories:

- **`events`**: Entertainment and event offers
- **`usage`**: Data usage information
- **`billing`**: Billing information and payment history
- **`recommended_plans`**: Personalized plan recommendations
- **`current_plan`**: Current plan details
- **`plans`**: Available plans and pricing
- **`top_hots`**: Trending spots and popular locations
- **`special_spots`**: Special VIP locations
- **`sports_events`**: Sports events and games

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- External telecom API access

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd telecom-assistant-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or manually install:
   pip install openai python-dotenv requests fastapi uvicorn pydantic pytest
   ```

3. **Environment Setup**
   Create a `.env` file with the following variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   DATA_RETRIEVAL_API_URL=http://ec2-52-66-186-107.ap-south-1.compute.amazonaws.com:7000/api/analyzeIntent
   DESIGN_API_URL=http://ec2-52-66-186-107.ap-south-1.compute.amazonaws.com:7000/api/getDesign
   ```

### Running the System

1. **Start the FastAPI server**
   ```bash
   python3 main.py
   # Or use the startup script:
   ./start_api.sh
   ```

2. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Interactive API: http://localhost:8000/redoc
   - Health Check: http://localhost:8000/health

3. **Development Mode**
   ```bash
   # Test individual components:
   python3 orchestrator.py           # CLI interface
   python3 intent_classification_agent.py  # Intent testing
   python3 data_retrieval_agent.py         # Data retrieval testing
   ```

## 🔧 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and version |
| `/health` | GET | System health check |
| `/query` | POST | Process user queries through full pipeline |
| `/classify-intent` | POST | Intent classification only |
| `/system-info` | GET | System configuration and status |
| `/intents` | GET | List of supported intents |
| `/design` | POST | Retrieve design data |
| `/conversation-history` | GET | Get conversation history (last 5 conversations) |
| `/conversation-history` | DELETE | Clear conversation history and reset memory |

### Example Usage

**Process a Query**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "show my data usage"}'
```

**Classify Intent Only**
```bash
curl -X POST "http://localhost:8000/classify-intent" \
  -H "Content-Type: application/json" \
  -d '{"message": "what are the trending spots?"}'
```

**Get Design Data**
```bash
curl -X POST "http://localhost:8000/design" \
  -H "Content-Type: application/json" \
  -d '{"intent": "events", "type": "Events", "message": "Show upcoming concerts"}'
```

**Get Conversation History**
```bash
curl "http://localhost:8000/conversation-history"
```

**Clear Conversation Memory**
```bash
curl -X DELETE "http://localhost:8000/conversation-history"
```

## 🧪 Testing & Development

### Run Individual Components

```bash
# Test intent classification
python3 intent_classification_agent.py

# Test data retrieval
python3 data_retrieval_agent.py

# Test design agent
python3 design_agent.py

# Test orchestrator (CLI interface)
python3 orchestrator.py
```

### Run Test Suite

```bash
# Run all tests
python3 -m pytest test/ -v

# Run with coverage report
python3 -m pytest test/ --cov=. --cov-report=term-missing -v

# Run specific test file
python3 -m pytest test/test_orchestrator.py -v
```

## ⚙️ Configuration

### API Endpoints
- **Data API**: `http://ec2-52-66-186-107.ap-south-1.compute.amazonaws.com:7000/api/analyzeIntent`
- **Design API**: `http://ec2-52-66-186-107.ap-south-1.compute.amazonaws.com:7000/api/getDesign`

### Model Settings
- **LLM Model**: GPT-4o-mini (configurable via environment)
- **Temperature**: 0.2 (for consistent responses)
- **Max Tokens**: 50 (for concise outputs)

### Conversation Memory Settings
- **Memory Size**: Last 5 conversations with automatic FIFO management
- **Memory Categories**: Intent classifications, API calls, design requests
- **Memory Persistence**: In-memory storage with API-based management
- **Memory Tracking**: Timestamps, query types, and response categorization

## 🔍 Response Format

### Query Response
```json
{
  "error": false,
  "message": "Success",
  "results": {
    "usage": [...],
    "events": [...]
  },
  "queries_used": ["usage: usage data"]
}
```

### Intent Classification Response
```json
{
  "intent": ["usage"],
  "inappropriate": false,
  "conversational_response": null,
  "confidence": 1.0,
  "context": "intent_detected"
}
```

### Conversation History Response
```json
{
  "conversation_count": 3,
  "memory_status": "Active (3/5 conversations stored)",
  "conversations": {
    "intent_classifications": [...],
    "api_calls": [...],
    "design_calls": [...]
  }
}
```

## 🛠️ Development & Customization

### Adding New Intents
1. Update `valid_intents` list in `IntentClassificationAgent` class
2. Add intent-to-query mapping in `DataRetrievalAgent._get_query_for_intent()`
3. Update system prompt in `classify_intent()` method
4. Add corresponding test cases in `test_intent.py`

### Customizing API Endpoints
Configure via environment variables or modify agent constructors:
```python
# Via environment
DATA_RETRIEVAL_API_URL=your_custom_api_endpoint
DESIGN_API_URL=your_design_api_endpoint

# Or in code
self.api_url = os.getenv("DATA_RETRIEVAL_API_URL", "default_url")
```

### Conversation Memory Customization
Modify memory settings in `Orchestrator` class:
```python
def __init__(self):
    self.conversation_memory = []
    self.max_conversations = 5  # Adjust memory size
```

### Comprehensive Error Handling
The system includes robust error handling for:
- OpenAI API failures with fallback responses
- External API connection failures and timeouts
- Invalid JSON responses and malformed data
- Network connectivity issues
- Unexpected exceptions with graceful degradation
- Memory overflow with automatic cleanup

## 📊 Monitoring

### System Information
```bash
curl "http://localhost:8000/system-info"
```

### Conversation History & Memory
```bash
# View conversation history
curl "http://localhost:8000/conversation-history"

# Monitor memory usage
curl "http://localhost:8000/system-info" | jq '.conversation_memory'
```

## 🔒 Security & Privacy Features

- **Environment Variable Security**: All sensitive data (API keys, URLs) stored in environment variables
- **Input Validation**: Comprehensive validation and sanitization of user inputs
- **Error Handling**: Secure error responses without exposing internal system details
- **CORS Configuration**: Proper CORS setup for secure web application integration
- **Memory Management**: Automatic conversation memory cleanup to prevent data accumulation
- **API Key Protection**: OpenAI API key validation and secure storage

## 🐛 Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `.env` file exists with `OPENAI_API_KEY`

2. **API Connection Failures**
   - Check network connectivity
   - Verify external API endpoints are accessible

3. **Memory Issues**
   - Clear conversation history: `curl -X DELETE "http://localhost:8000/conversation-history"`
   - Check memory status: `curl "http://localhost:8000/system-info"`
   - Restart the application if memory persists

4. **Test Failures**
   - Run specific tests: `python3 -m pytest test/test_specific.py -v`
   - Check test coverage: `python3 -m pytest test/ --cov=. --cov-report=term-missing`
   - Verify environment variables are set correctly

## 📝 Documentation & Code Quality

- **Comprehensive Comments**: All source files include detailed step-by-step comments
- **Test Documentation**: All test files include brief explanatory comments
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation at `/docs`
- **Type Hints**: Python type hints throughout for better IDE support
- **Error Context**: Detailed error messages with context information
