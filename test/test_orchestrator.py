#!/usr/bin/env python3
"""
Unit tests for TwoAgentSystem orchestrator
"""

import pytest
from unittest.mock import Mock, patch
import pytest
from unittest.mock import Mock, patch
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator


class TestTwoAgentSystem:
    """Test cases for Orchestrator (formerly TwoAgentSystem)"""
    
    def setup_method(self):
        """Setup mocked agents before each test method"""
        with patch('orchestrator.IntentClassificationAgent') as mock_intent_agent, \
             patch('orchestrator.DataRetrievalAgent') as mock_data_agent:
            
            # Create mock instances for both agents
            self.mock_intent_agent_instance = Mock()
            self.mock_data_agent_instance = Mock()
            
            mock_intent_agent.return_value = self.mock_intent_agent_instance
            mock_data_agent.return_value = self.mock_data_agent_instance
            
            # Initialize orchestrator with mocked agents
            self.system = Orchestrator(
                openai_api_key="test-key",
                api_url="http://test-api.com"
            )
    
    def test_initialization(self):
        """Test orchestrator initializes both agents and conversation history"""
        assert self.system.intent_agent == self.mock_intent_agent_instance
        assert self.system.data_agent == self.mock_data_agent_instance
    
    def test_process_query_successful_flow(self):
        """Test end-to-end successful query processing through both agents"""
        # Mock successful intent classification
        intent_result = {
            "error": False,
            "intent": ["usage"],
            "inappropriate": False,
            "conversational_response": None,
            "confidence": 1.0
        }
        
        # Mock successful data retrieval
        data_result = {
            "error": False,
            "message": "Success",
            "results": {"usage": [{"data": "5GB"}]},
            "queries_used": ["usage: usage data"],
            "context": "success"
        }
        
        self.mock_intent_agent_instance.classify_intent.return_value = intent_result
        self.mock_data_agent_instance.retrieve_data.return_value = data_result
        
        result = self.system.process_query("show my data usage")
        
        # Verify successful processing
        assert result["error"] == False
        assert result["message"] == "Success"
        assert "usage" in result["results"]
    
    def test_process_query_data_retrieval_error(self):
        """Test orchestrator propagates data retrieval agent errors correctly"""
        intent_result = {
            "error": False,
            "intent": ["usage"],
            "inappropriate": False,
            "conversational_response": None,
            "confidence": 1.0
        }
        
        data_result = {
            "error": True,
            "error_message": "API service unavailable",
            "context": "api_call_failed"
        }
        
        self.mock_intent_agent_instance.classify_intent.return_value = intent_result
        self.mock_data_agent_instance.retrieve_data.return_value = data_result
        
        result = self.system.process_query("show my data usage")
        
        assert result["error"] == True
        assert "API service unavailable" in result["error_message"]
    
    def test_process_query_system_exception(self):
        """Test orchestrator handles unexpected exceptions gracefully"""
        # Mock intent agent to throw unexpected exception
        self.mock_intent_agent_instance.classify_intent.side_effect = Exception("System crash")
        
        result = self.system.process_query("test query")
        
        # Verify error is caught and structured properly
        assert result["error"] == True
        assert "System error" in result["error_message"]

    def test_process_query_intent_classification_error(self):
        """Test orchestrator handles intent classification errors properly"""
        intent_result = {
            "error": True,
            "error_message": "Intent classification failed",
            "context": "llm_error"
        }
        
        self.mock_intent_agent_instance.classify_intent.return_value = intent_result
        
        # The system should still proceed to data retrieval even if intent classification has error=True
        # but in practice, let's test what actually happens
        data_result = {
            "error": False,
            "message": "Fallback response",
            "results": {},
            "context": "fallback"
        }
        self.mock_data_agent_instance.retrieve_data.return_value = data_result
        
        result = self.system.process_query("test query")
        
        # The system returns the data agent result
        assert result["error"] == False
        assert result["message"] == "Fallback response"

    def test_get_system_info(self):
        """Test get_system_info method"""
        # Set up mock API URL
        self.mock_data_agent_instance.api_url = "http://test-api.com"
        self.mock_intent_agent_instance.valid_intents = ["usage", "billing"]
        
        result = self.system.get_system_info()
        
        assert "IntentClassificationAgent" in result["intent_agent"]
        assert "DataRetrievalAgent" in result["data_agent"]
        assert result["api_url"] == "http://test-api.com"
        assert result["available_intents"] == ["usage", "billing"]

    def test_initialization_with_env_vars(self):
        """Test system initialization with environment variables"""
        with patch('orchestrator.IntentClassificationAgent') as mock_intent_agent, \
             patch('orchestrator.DataRetrievalAgent') as mock_data_agent, \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'env-openai-key', 'DATA_RETRIEVAL_API_URL': 'env-api-url'}):
            
            system = Orchestrator()
            
            # Check that agents were initialized with environment variables
            mock_intent_agent.assert_called_once_with(None)  # Should use env var
            mock_data_agent.assert_called_once_with(None)   # Should use env var

    def test_main_function_no_api_key(self):
        """Test main function when no API key is set"""
        from orchestrator import main
        
        with patch.dict('os.environ', {}, clear=True), \
             patch('orchestrator.logger') as mock_logger:
            
            main()
            
            mock_logger.error.assert_called_with("Error: Set OPENAI_API_KEY in .env file")

    def test_main_function_success(self):
        """Test main function with successful initialization"""
        from orchestrator import main
    
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
             patch('orchestrator.Orchestrator') as mock_system_class, \
             patch('orchestrator.input', side_effect=['test query', 'quit']), \
             patch('orchestrator.logger') as mock_logger:
            
            mock_system = Mock()
            mock_system.process_query.return_value = {"message": "test response"}
            mock_system_class.return_value = mock_system
            
            main()
            
            mock_logger.info.assert_any_call("2-Agent System Ready")
            mock_system.process_query.assert_called_once_with('test query')

    def test_main_function_keyboard_interrupt(self):
        """Test main function handling keyboard interrupt"""
        from orchestrator import main
    
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
             patch('orchestrator.Orchestrator'), \
             patch('orchestrator.input', side_effect=KeyboardInterrupt):            # Should not raise an exception
            main()

    def test_main_function_exception_handling(self):
        """Test main function handling exceptions during query processing"""
        from orchestrator import main
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
             patch('orchestrator.Orchestrator') as mock_system_class, \
             patch('orchestrator.input', side_effect=['test query', 'quit']), \
             patch('orchestrator.logger') as mock_logger:
            
            mock_system = Mock()
            mock_system.process_query.side_effect = Exception("Process error")
            mock_system_class.return_value = mock_system
            
            main()
            
            mock_logger.error.assert_called_with("Error: Process error")

    def test_add_conversation(self):
        """Test adding conversation to orchestrator's memory storage"""
        user_message = "test message"
        response = {
            "intent": "data_usage",
            "confidence": 0.95,
            "inappropriate": False,
            "results": {"data": "test"},
            "queries_used": ["query1"]
        }
        
        # Add conversation to memory
        self.system.add_conversation(user_message, response)
        
        # Verify conversation was stored with proper structure
        assert len(self.system.conversation_history) == 1
        conv = self.system.conversation_history[0]
        assert conv["user_message"] == user_message
        assert conv["response"] == response
        assert "timestamp" in conv

    def test_conversation_history_limit(self):
        """Test conversation memory maintains 5-conversation limit (FIFO)"""
        # Add 7 conversations
        for i in range(7):
            self.system.add_conversation(f"message {i}", {"test": i})
        
        # Should only have the last 5
        assert len(self.system.conversation_history) == 5
        # Should have messages 2-6
        assert self.system.conversation_history[0]["user_message"] == "message 2"
        assert self.system.conversation_history[4]["user_message"] == "message 6"

    def test_get_conversation_history(self):
        """Test get_conversation_history returns stored conversations"""
        # Add a conversation with intent data
        response1 = {
            "intent": "data_usage",
            "confidence": 0.95,
            "inappropriate": False,
            "results": {"usage": "5GB"},
            "queries_used": ["data_usage_query"]
        }
        self.system.add_conversation("check usage", response1)
        
        # Add a conversation without results (intent only)
        response2 = {
            "intent": "bill_payment",
            "confidence": 0.88,
            "inappropriate": False
        }
        self.system.add_conversation("pay bill", response2)
        
        result = self.system.get_conversation_history()
        
        # Check structure
        assert "intent_classifications" in result
        assert "api_calls" in result
        assert "design_calls" in result
        
        # Should have 2 intent classifications
        assert len(result["intent_classifications"]) == 2
        
        # Should have 1 API call (only first conversation had results)
        assert len(result["api_calls"]) == 1
        
        # Check intent classification data
        intent_data = result["intent_classifications"][0]
        assert intent_data["intent"] == "data_usage"
        assert intent_data["confidence"] == 0.95
        assert intent_data["inappropriate"] == False
        assert intent_data["user_message"] == "check usage"
        
        # Check API call data
        api_data = result["api_calls"][0]
        assert api_data["user_message"] == "check usage"
        assert api_data["queries_used"] == ["data_usage_query"]
        assert api_data["results_count"] == 1

    def test_clear_conversation_memory(self):
        """Test orchestrator clears all stored conversation history"""
        # Add multiple conversations to memory
        self.system.add_conversation("test 1", {"intent": "test"})
        self.system.add_conversation("test 2", {"intent": "test"})
        assert len(self.system.conversation_history) == 2
        
        # Clear all conversation memory
        with patch('orchestrator.logger') as mock_logger:
            self.system.clear_conversation_memory()
            
            # Verify memory was completely cleared
            assert len(self.system.conversation_history) == 0
            mock_logger.info.assert_called_once_with("Conversation memory cleared")

    def test_design_agent_conversation_tracking(self):
        """Test design agent conversations are categorized separately from query conversations"""
        # Add a design agent conversation
        design_response = {
            "agent_type": "design",
            "intent": "bill_payment",
            "design_type": "payment_form",
            "error": False,
            "results": {"form_html": "<form>...</form>"}
        }
        self.system.add_conversation("Design request: create payment form (intent: bill_payment, type: payment_form)", design_response)
        
        # Add a regular query conversation
        query_response = {
            "intent": "data_usage",
            "confidence": 0.95,
            "inappropriate": False,
            "results": {"usage": "5GB"},
            "queries_used": ["data_usage_query"]
        }
        self.system.add_conversation("check my usage", query_response)
        
        result = self.system.get_conversation_history()
        
        # Check that all three categories exist
        assert "intent_classifications" in result
        assert "api_calls" in result
        assert "design_calls" in result
        
        # Should have 1 design call
        assert len(result["design_calls"]) == 1
        assert len(result["intent_classifications"]) == 1
        assert len(result["api_calls"]) == 1
        
        # Check design call data
        design_data = result["design_calls"][0]
        assert design_data["intent"] == "bill_payment"
        assert design_data["design_type"] == "payment_form"
        assert design_data["error"] == False
        assert design_data["results_available"] == True
        assert "Design request:" in design_data["user_message"]
        
        # Verify regular conversation tracking still works
        intent_data = result["intent_classifications"][0]
        assert intent_data["intent"] == "data_usage"
        assert intent_data["user_message"] == "check my usage"