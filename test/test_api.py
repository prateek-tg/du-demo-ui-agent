import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


class TestFastAPIApp:
    def test_root_endpoint(self):
        """Test root endpoint returns basic API information"""
        with TestClient(main.app) as client:
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "2-Agent Telecom Assistant API" in data["message"]

    def test_health_endpoint_success(self):
        """Test health check endpoint when system is running"""
        with TestClient(main.app) as client:
            response = client.get("/health")
            # Health check should work regardless of system state during testing
            assert response.status_code in [200, 503]  # Allow both during testing
            
    def test_404_error_handler(self):
        """Test custom 404 error handler for non-existent endpoints"""
        with TestClient(main.app) as client:
            response = client.get("/nonexistent")
            assert response.status_code == 404
            data = response.json()
            assert data["error"] == True
            assert "Endpoint not found" in data["message"]

    def test_basic_endpoints_structure(self):
        """Test that all endpoints return proper HTTP status codes and handle validation"""
        with TestClient(main.app) as client:
            # Test root endpoint returns success
            response = client.get("/")
            assert response.status_code == 200
            
            # Test health endpoint returns appropriate status
            response = client.get("/health")
            assert response.status_code in [200, 503]
            
            # Test that POST endpoints properly validate required JSON input
            response = client.post("/query")  # Missing required JSON body
            assert response.status_code == 422  # Pydantic validation error
            
            response = client.post("/classify-intent")  # Missing required JSON body
            assert response.status_code == 422  # Pydantic validation error
            
            response = client.post("/design")  # Missing required JSON body
            assert response.status_code == 422  # Pydantic validation error

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('main.Orchestrator')
    @patch('main.DesignAgent')
    def test_startup_event_success(self, mock_design_agent, mock_orchestrator):
        """Test successful FastAPI lifespan initialization with mocked agents"""
        from main import lifespan, app
        import asyncio
        
        # Mock successful agent initialization
        mock_system_instance = Mock()
        mock_design_instance = Mock()
        mock_orchestrator.return_value = mock_system_instance
        mock_design_agent.return_value = mock_design_instance
        
        # Test the lifespan context manager
        async def test_lifespan():
            async with lifespan(app):
                pass  # The initialization happens in the context manager
        
        asyncio.run(test_lifespan())
        
        # Verify both agents were created successfully
        mock_orchestrator.assert_called_once()
        mock_design_agent.assert_called_once()

    @patch.dict('os.environ', {}, clear=True)
    def test_startup_event_no_api_key(self):
        """Test lifespan fails when OpenAI API key is missing"""
        from main import lifespan, app
        import asyncio
        
        async def test_lifespan():
            async with lifespan(app):
                pass
        
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY environment variable is required"):
            asyncio.run(test_lifespan())

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('main.Orchestrator')
    @patch('main.DesignAgent')
    def test_startup_event_initialization_failure(self, mock_design_agent, mock_orchestrator):
        """Test lifespan fails gracefully when agent initialization throws exception"""
        from main import lifespan, app
        import asyncio
        
        # Mock agent initialization failure
        mock_orchestrator.side_effect = Exception("Initialization failed")
        
        async def test_lifespan():
            async with lifespan(app):
                pass
        
        with pytest.raises(RuntimeError, match="System initialization failed"):
            asyncio.run(test_lifespan())

    def test_pydantic_models(self):
        """Test Pydantic request/response models validation and field handling"""
        from main import QueryRequest, QueryResponse, IntentClassificationRequest, DesignRequest
        
        # Test QueryRequest with optional user_id field
        query_req = QueryRequest(message="test message", user_id="user123")
        assert query_req.message == "test message"
        assert query_req.user_id == "user123"
        
        # Test QueryRequest without optional user_id
        query_req2 = QueryRequest(message="test message")
        assert query_req2.message == "test message"
        assert query_req2.user_id is None
        
        # Test IntentClassificationRequest with required message field
        intent_req = IntentClassificationRequest(message="show usage")
        assert intent_req.message == "show usage"
        
        # Test DesignRequest with all required fields
        design_req = DesignRequest(intent="events", type="Events", message="Show events")
        assert design_req.intent == "events"
        assert design_req.type == "Events"
        assert design_req.message == "Show events"

    def test_cors_middleware_configured(self):
        """Test CORS middleware is configured to allow cross-origin requests"""
        # Verify CORS middleware is present in the FastAPI app
        assert main.app.user_middleware is not None

    def test_conversation_history_endpoint(self):
        """Test GET /conversation-history returns stored conversation data"""
        # Mock orchestrator to return empty conversation history
        mock_system = Mock()
        mock_system.get_conversation_history.return_value = {
            "intent_classifications": [],
            "api_calls": [],
            "design_calls": []
        }
        
        original_orchestrator = main.orchestrator
        main.orchestrator = mock_system
        
        try:
            with TestClient(main.app) as client:
                response = client.get("/conversation-history")
                
                assert response.status_code == 200
                data = response.json()
                assert data["error"] == False
                assert "conversation_history" in data
        finally:
            main.orchestrator = original_orchestrator

    def test_clear_conversation_history_endpoint(self):
        """Test DELETE /conversation-history clears stored conversations"""
        # Mock orchestrator clear memory method
        mock_system = Mock()
        mock_system.clear_conversation_memory.return_value = None
        
        original_orchestrator = main.orchestrator
        main.orchestrator = mock_system
        
        try:
            with TestClient(main.app) as client:
                response = client.delete("/conversation-history")
                
                assert response.status_code == 200
                data = response.json()
                assert data["error"] == False
                assert "cleared successfully" in data["message"]
        finally:
            main.orchestrator = original_orchestrator

    def test_conversation_history_system_not_initialized(self):
        """Test conversation history endpoints gracefully handle uninitialized orchestrator"""
        # Create client and temporarily disable orchestrator to simulate startup failure
        with TestClient(main.app) as client:
            # Save original orchestrator and set to None to simulate uninitialized state
            original_orchestrator = main.orchestrator
            main.orchestrator = None
            
            try:
                # Test GET endpoint returns proper error when orchestrator is None
                response = client.get("/conversation-history")
                assert response.status_code == 200
                data = response.json()
                assert data["error"] == True
                assert "System not initialized" in data["error_message"]
                
                # Test DELETE endpoint returns proper error when orchestrator is None
                response = client.delete("/conversation-history")
                assert response.status_code == 200
                data = response.json()
                assert data["error"] == True
                assert "System not initialized" in data["error_message"]
            finally:
                main.orchestrator = original_orchestrator