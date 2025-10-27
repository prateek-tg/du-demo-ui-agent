#!/usr/bin/env python3
"""
FastAPI Application for 2-Agent Intent Classification and Data Retrieval System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import uvicorn
from loguru import logger

# Import agent orchestrator and design agent
from orchestrator import Orchestrator
from src.design_agent import DesignAgent

# Load environment variables from .env file
load_dotenv()

# Global agent instances (initialized during lifespan)
orchestrator = None
design_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.
    Initializes agent system and checks for required environment variables.
    """
    global orchestrator, design_agent
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    try:
        # Initialize agent orchestrator and design agent
        orchestrator = Orchestrator()
        design_agent = DesignAgent()
        logger.success("2-Agent System initialized successfully")
        logger.success("Design Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize 2-Agent System: {e}")
        raise RuntimeError(f"System initialization failed: {e}")
    
    yield
    
    # Cleanup (if needed)
    logger.info("Shutting down 2-Agent System")

# Initialize FastAPI app with metadata and lifespan
app = FastAPI(
    title="2-Agent Telecom Assistant API",
    description="Intent classification and data retrieval API for telecom services",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow cross-origin requests (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    message: str  # User query message
    user_id: Optional[str] = None  # Optional user identifier

class QueryResponse(BaseModel):
    error: bool  # Indicates if an error occurred
    message: str  # Response message
    results: Dict[str, Any]  # Data results from agents
    queries_used: Optional[List[str]] = None  # List of queries used (if any)

class IntentClassificationRequest(BaseModel):
    message: str  # Message to classify intent for

class IntentClassificationResponse(BaseModel):
    intent: Optional[List[str]]  # List of classified intents
    inappropriate: bool  # True if message is inappropriate
    conversational_response: Optional[str]  # Optional conversational reply
    confidence: float  # Confidence score

class SystemInfoResponse(BaseModel):
    agents: Dict[str, Any]  # Info about agents
    status: str  # System status

class HealthResponse(BaseModel):
    status: str  # Health status
    message: str  # Health message

class DesignRequest(BaseModel):
    intent: str = Field(..., description="The intent for which to get design")
    type: str = Field(..., description="The type of design to retrieve")
    message: str = Field(..., description="The message to include in the design request")



@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint. Returns basic API info and links to docs/health.
    """
    return {
        "message": "2-Agent Telecom Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint. Returns system status.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return HealthResponse(
        status="healthy",
        message="2-Agent System is running"
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main endpoint to process user queries through the 2-agent system.
    Steps:
      1. Classifies intent (Agent 1)
      2. Retrieves data (Agent 2)
      3. Returns structured response
    """
    if orchestrator is None:
        return {
            "error": True,
            "message": "System not initialized",
            "results": {},
            "queries_used": []
        }
    if not request.message.strip():
        return {
            "error": True,
            "message": "Message cannot be empty",
            "results": {},
            "queries_used": []
        }
    try:
        # Process the query through both agents
        result = orchestrator.process_query(request.message)
        # Uniform error handling for agent errors
        if result.get("error"):
            error_response = {
                "error": True,
                "message": result.get("error_message", result.get("message", "Query processing failed")),
                "results": result.get("results", {}),
                "queries_used": result.get("queries_used", [])
            }
            # Store error responses in conversation history for analysis and debugging
            orchestrator.add_conversation(request.message, error_response)
            return error_response
        
        # Build successful response with additional fields for conversation tracking
        success_response = {
            "error": False,
            "message": result.get("message", "Success"),
            "results": result.get("results", {}),
            "queries_used": result.get("queries_used", []),
            "context": result.get("context", "success"),
            "intent": result.get("intent"),                    # Include intent for history categorization
            "confidence": result.get("confidence"),            # Include confidence for quality tracking
            "inappropriate": result.get("inappropriate", False) # Include language flag for monitoring
        }
        # Store successful responses in conversation history (maintains last 5 conversations)
        orchestrator.add_conversation(request.message, success_response)
        return success_response
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return {
            "error": True,
            "message": f"Query processing failed: {str(e)}",
            "results": {},
            "queries_used": []
        }

@app.post("/classify-intent", response_model=IntentClassificationResponse)
async def classify_intent_only(request: IntentClassificationRequest):
    """
    Endpoint to only classify intent (no data retrieval).
    Useful for testing or custom workflows.
    """
    if orchestrator is None:
        return {
            "error": True,
            "error_message": "System not initialized",
            "context": "startup",
            "intent": None,
            "inappropriate": False,
            "conversational_response": None,
            "confidence": 0.0
        }
    if not request.message.strip():
        return {
            "error": True,
            "error_message": "Message cannot be empty",
            "context": "validation",
            "intent": None,
            "inappropriate": False,
            "conversational_response": None,
            "confidence": 0.0
        }
    try:
        # Use only the intent classification agent
        intent_result = orchestrator.intent_agent.classify_intent(request.message)
        # Uniform error handling for agent errors
        if intent_result.get("error"):
            return {
                "error": True,
                "error_message": intent_result.get("error_message", "Intent classification failed"),
                "context": intent_result.get("context", "agent_error"),
                "intent": intent_result.get("intent"),
                "inappropriate": intent_result.get("inappropriate", False),
                "conversational_response": intent_result.get("conversational_response"),
                "confidence": intent_result.get("confidence", 0.0)
            }
        return {
            "error": False,
            "intent": intent_result.get("intent"),
            "inappropriate": intent_result.get("inappropriate", False),
            "conversational_response": intent_result.get("conversational_response"),
            "confidence": intent_result.get("confidence", 0.0),
            "context": intent_result.get("context", "success")
        }
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            "error": True,
            "error_message": f"Intent classification failed: {str(e)}",
            "context": "exception",
            "intent": None,
            "inappropriate": False,
            "conversational_response": None,
            "confidence": 0.0
        }

@app.get("/system-info", response_model=SystemInfoResponse)
async def get_system_info():
    """
    Get information about the system configuration and agents.
    """
    if orchestrator is None:
        return {
            "error": True,
            "error_message": "System not initialized",
            "context": "startup",
            "agents": {},
            "status": "error"
        }
    try:
        info = orchestrator.get_system_info()
        return {
            "error": False,
            "agents": info,
            "status": "operational",
            "context": "success"
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {
            "error": True,
            "error_message": f"Failed to get system info: {str(e)}",
            "context": "exception",
            "agents": {},
            "status": "error"
        }

@app.get("/intents", response_model=Dict[str, List[str]])
async def get_supported_intents():
    """
    Get list of supported intents and their descriptions.
    """
    if orchestrator is None:
        return {
            "error": True,
            "error_message": "System not initialized",
            "context": "startup",
            "supported_intents": [],
            "intent_descriptions": []
        }
    try:
        return {
            "error": False,
            "supported_intents": orchestrator.intent_agent.valid_intents,
            "intent_descriptions": [
                "events: Entertainment and event offers",
                "usage: Data usage information", 
                "billing: Billing information and payment history",
                "recommended_plans: Personalized plan recommendations",
                "current_plan: Current plan details",
                "plans: Available plans and pricing",
                "top_hots: Trending spots and popular locations",
                "special_spots: Special VIP locations",
                "sports_events: Sports events and games"
            ],
            "context": "success"
        }
    except Exception as e:
        logger.error(f"Failed to get intents: {e}")
        return {
            "error": True,
            "error_message": f"Failed to get intents: {str(e)}",
            "context": "exception",
            "supported_intents": [],
            "intent_descriptions": []
        }

@app.post("/design")
async def get_design(request: DesignRequest):
    """
    Get design data by calling external design API with user-provided intent, type, and message.
    """
    if orchestrator is None:
        return {
            "error": True,
            "error_message": "System not initialized",
            "context": "startup",
            "results": {},
            "request_payload": {}
        }
    
    try:
        # Call the design agent with the provided intent, type, and message
        design_result = design_agent.get_design(
            intent=request.intent,
            type_value=request.type,
            message=request.message
        )
        
        # Create formatted user message for conversation tracking
        # Include all request parameters for better context in conversation history
        user_message = f"Design request: {request.message} (intent: {request.intent}, type: {request.type})"
        
        # Handle design agent errors with structured response
        if design_result.get("error"):
            error_response = {
                "error": True,
                "error_message": design_result.get("error_message", design_result.get("message", "Failed to get design data")),
                "context": design_result.get("context", "agent_error"),
                "results": design_result.get("results", {}),
                "request_payload": design_result.get("request_payload", {}),
                "agent_type": "design",                    # Mark as design agent response for categorization
                "intent": request.intent,                   # Store original intent for tracking
                "design_type": request.type                # Store design type for analysis
            }
            # Store design errors in conversation history for monitoring
            orchestrator.add_conversation(user_message, error_response)
            return error_response
        
        success_response = {
            "error": False,
            "results": design_result.get("results", design_result),
            "context": design_result.get("context", "success"),
            "request_payload": design_result.get("request_payload", {}),
            "agent_type": "design",
            "intent": request.intent,
            "design_type": request.type
        }
        # Track successful responses in conversation history
        orchestrator.add_conversation(user_message, success_response)
        return success_response
    except Exception as e:
        logger.error(f"Failed to get design data: {e}")
        return {
            "error": True,
            "error_message": f"Failed to get design data: {str(e)}",
            "context": "exception",
            "results": {},
            "request_payload": {}
        }

@app.get("/conversation-history")
async def get_conversation_history():
    """
    Get the conversation history from both agents (last 6 conversations).
    Returns stored context for debugging or improvement.
    """
    if orchestrator is None:
        return {
            "error": True,
            "error_message": "System not initialized",
            "context": "startup",
            "conversation_history": {},
            "memory_status": {}
        }
    try:
        # Retrieve categorized conversation history from orchestrator
        history = orchestrator.get_conversation_history()
        return {
            "error": False,
            "conversation_history": history,               # Categorized into intents, API calls, design calls
            "memory_status": {
                "intent_memory_size": len(history["intent_classifications"]),  # Count of intent detections
                "data_memory_size": len(history["api_calls"]),                 # Count of data retrieval operations
                "design_memory_size": len(history["design_calls"]),            # Count of design requests
                "max_size": 5,                                                 # Maximum conversations stored
                "description": "Short-term memory preserves last 5 chat interactions from query, data retrieval, and design agents"
            },
            "context": "success"
        }
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        return {
            "error": True,
            "error_message": f"Failed to get conversation history: {str(e)}",
            "context": "exception",
            "conversation_history": {},
            "memory_status": {}
        }

@app.delete("/conversation-history")
async def clear_conversation_history():
    """
    Clear the conversation history from both agents.
    Resets memory for a fresh start.
    """
    if orchestrator is None:
        return {
            "error": True,
            "error_message": "System not initialized",
            "context": "startup",
            "message": "",
            "status": "error"
        }
    try:
        orchestrator.clear_conversation_memory()
        return {
            "error": False,
            "message": "Conversation history cleared successfully",
            "status": "success",
            "context": "success"
        }
    except Exception as e:
        logger.error(f"Failed to clear conversation history: {e}")
        return {
            "error": True,
            "error_message": f"Failed to clear conversation history: {str(e)}",
            "context": "exception",
            "message": "",
            "status": "error"
        }

# Error handlers for common HTTP errors
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    Handles 404 errors (endpoint not found).
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={
            "error": True,
            "message": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/query", "/classify-intent", "/system-info", "/intents", "/design", "/conversation-history"]
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """
    Handles 500 errors (internal server error).
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "detail": "Something went wrong processing your request"
        }
    )

if __name__ == "__main__":
    # Entry point for running the FastAPI application
    logger.info("🚀 Starting 2-Agent Telecom Assistant API...")
    logger.info("📝 API Documentation: http://localhost:8000/docs")
    logger.info("🔍 Interactive API: http://localhost:8000/redoc")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )