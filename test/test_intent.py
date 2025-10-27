import pytest
from unittest.mock import Mock, patch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intent_classification_agent import IntentClassificationAgent


class TestIntentClassificationAgent:
    def setup_method(self):
        """Setup mocked OpenAI client for each test method"""
        self.mock_api_key = "test-api-key-123"
        # Mock OpenAI client to avoid real API calls during testing
        with patch('src.intent_classification_agent.openai.OpenAI') as mock_openai_class:
            self.mock_client = Mock()
            mock_openai_class.return_value = self.mock_client
            self.agent = IntentClassificationAgent(openai_api_key=self.mock_api_key)
        self.test_message = "show my data usage"

    def test_initialization(self):
        """Test agent initializes OpenAI client with provided API key"""
        with patch('src.intent_classification_agent.openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            agent = IntentClassificationAgent(openai_api_key="test-key")
            assert agent.client == mock_client

    def test_initialization_with_env_var(self):
        """Test agent initialization using OPENAI_API_KEY environment variable"""
        with patch('src.intent_classification_agent.openai.OpenAI') as mock_openai_class, \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'}):
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            agent = IntentClassificationAgent()
            mock_openai_class.assert_called_with(api_key='env-key')

    def test_valid_intents(self):
        """Test agent has correct list of supported telecom intents"""
        expected_intents = [
            "events", "usage", "billing", "recommended_plans", 
            "current_plan", "plans", "top_hots", "special_spots", "sports_events"
        ]
        assert self.agent.valid_intents == expected_intents

    def test_classify_intent_success(self):
        """Test LLM successfully identifies valid telecom intent"""
        # Mock LLM response with direct intent match
        mock_choice = Mock()
        mock_choice.message.content = "usage"
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = self.agent.classify_intent(self.test_message)
        
        # Verify successful intent detection
        assert result["error"] == False
        assert result["intent"] == ["usage"]
        assert result["inappropriate"] == False
        assert result["confidence"] == 1.0
        assert result["context"] == "intent_detected"

    def test_classify_intent_inappropriate_language(self):
        """Test agent detects inappropriate language while still identifying intent"""
        # Mock LLM response indicating intent + inappropriate language
        mock_choice = Mock()
        mock_choice.message.content = "usage - Please use polite language"
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = self.agent.classify_intent("show my damn data usage")
        
        # Verify intent detected but inappropriate flag set
        assert result["error"] == False
        assert result["intent"] == ["usage"]
        assert result["inappropriate"] == True
        assert result["context"] == "inappropriate_language"

    def test_classify_intent_conversational_response(self):
        """Test handling non-telecom queries with helpful conversational response"""
        # Mock LLM providing conversational response for non-telecom query
        mock_choice = Mock()
        mock_choice.message.content = "I can help with telecom services..."
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = self.agent.classify_intent("hello there")
        
        # Verify conversational response provided
        assert result["error"] == False
        assert result["intent"] is None
        assert result["conversational_response"] == "I can help with telecom services..."
        assert result["confidence"] == 0
        assert result["context"] == "conversational_response"

    def test_classify_intent_multiple_intents_detected(self):
        """Test ambiguous query requiring clarification between multiple intents"""
        # Mock LLM response requesting clarification between intents
        mock_choice = Mock()
        mock_choice.message.content = "Please specify which one you want - usage or billing"
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = self.agent.classify_intent("show my usage and billing")
        
        # Verify clarification response structure
        assert result["error"] == False
        assert result["intent"] is None
        assert result["conversational_response"] == "Please specify which one you want - usage or billing"
        assert result["confidence"] == 0

    def test_classify_intent_api_error(self):
        """Test graceful error handling when OpenAI API fails"""
        # Simulate API exception
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = self.agent.classify_intent(self.test_message)
        
        # Verify error response with fallback messaging
        assert result["error"] == True
        assert "LLM classification failed" in result["error_message"]
        assert result["intent"] is None
        assert result["context"] == "exception"
        assert "I'm having trouble understanding" in result["conversational_response"]

    def test_classify_intent_all_valid_intents(self):
        """Test agent correctly identifies each supported telecom intent"""
        # Test all valid intents individually
        for intent in self.agent.valid_intents:
            mock_choice = Mock()
            mock_choice.message.content = intent
            
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            
            self.mock_client.chat.completions.create.return_value = mock_response
            
            result = self.agent.classify_intent(f"show {intent}")
            
            # Verify each intent is properly classified
            assert result["error"] == False
            assert result["intent"] == [intent]
            assert result["inappropriate"] == False
            assert result["confidence"] == 1.0

    def test_classify_intent_with_model_from_env(self):
        """Test agent uses OpenAI model specified in environment variable"""
        with patch.dict('os.environ', {'OPENAI_MODEL': 'gpt-4'}):
            mock_choice = Mock()
            mock_choice.message.content = "usage"
            
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            
            self.mock_client.chat.completions.create.return_value = mock_response
            
            result = self.agent.classify_intent("show my usage")
            
            # Verify environment variable model is used in API call
            self.mock_client.chat.completions.create.assert_called_once()
            call_args = self.mock_client.chat.completions.create.call_args
            assert call_args[1]['model'] == 'gpt-4'