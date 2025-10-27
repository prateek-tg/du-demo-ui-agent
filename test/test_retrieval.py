import pytest
from unittest.mock import Mock, patch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_retrieval_agent import DataRetrievalAgent


class TestDataRetrievalAgent:
    """Test suite for data retrieval agent external API interactions"""
    
    def setup_method(self):
        """Initialize test environment with mock API and sample intent results"""
        self.mock_api_url = "http://test-api.example.com/data"
        self.agent = DataRetrievalAgent(api_url=self.mock_api_url)
        
        # Sample intent result with single telecom intent
        self.intent_result_single = {
            "intent": ["usage"],
            "inappropriate": False,
            "conversational_response": None,
            "confidence": 1.0
        }
        
        # Sample conversational result with no telecom intent
        self.intent_result_conversational = {
            "intent": None,
            "inappropriate": False,
            "conversational_response": "I can help with telecom services...",
            "confidence": 0
        }

    def test_intent_to_query_mapping(self):
        """Test agent maps telecom intents to appropriate API query parameters"""
        assert self.agent._get_query_for_intent("usage") == "usage data"
        assert self.agent._get_query_for_intent("billing") == "billing information"
        assert self.agent._get_query_for_intent("unknown_intent") == "available plans"

    @patch('src.data_retrieval_agent.requests.get')
    def test_retrieve_data_single_intent(self, mock_get):
        """Test successful data retrieval for single telecom intent"""
        # Mock successful API response with usage data
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": {
                "usage": [{"data_used": "5GB", "remaining": "15GB"}]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.agent.retrieve_data(self.intent_result_single)
        
        # Verify successful data retrieval and response structure
        assert result["error"] == False
        assert "usage" in result["results"]
        assert len(result["queries_used"]) == 1

    def test_retrieve_data_conversational_response(self):
        """Test handling conversational response with no data retrieval needed"""
        result = self.agent.retrieve_data(self.intent_result_conversational)
        
        # Verify conversational response passed through without API calls
        assert result["error"] == False
        assert result["message"] == "I can help with telecom services..."
        assert result["results"] == {}

    @patch('src.data_retrieval_agent.requests.get')
    def test_retrieve_data_api_error(self, mock_get):
        """Test graceful error handling when external API call fails"""
        # Mock the _call_api method to return the expected error structure
        with patch.object(self.agent, '_call_api') as mock_call_api:
            mock_call_api.return_value = {
                "error": True,
                "error_message": "API call failed: Network error",
                "query_used": "usage data",
                "context": "network_error"
            }
            
            result = self.agent.retrieve_data(self.intent_result_single)
            
            # Verify error response structure
            assert result["error"] == True
            assert "API call failed" in result["error_message"]

    def test_retrieve_data_multiple_intents(self):
        """Test handling multiple telecom intents in single request"""
        # Intent result with multiple telecom intents  
        intent_result_multiple = {
            "intent": ["usage", "billing"],
            "inappropriate": False,
            "conversational_response": None,
            "confidence": 1.0
        }
        
        with patch.object(self.agent, '_call_api') as mock_call_api:
            mock_call_api.return_value = {
                "results": {
                    "usage": [{"data_used": "5GB"}],
                    "billing": [{"amount": "$50"}]
                }
            }
            
            result = self.agent.retrieve_data(intent_result_multiple)
            
            # Verify multiple intents processed and queries tracked
            assert result["error"] == False
            assert len(result["queries_used"]) == 2
            assert "usage: usage data" in result["queries_used"]
            assert "billing: billing information" in result["queries_used"]

    def test_retrieve_data_no_intents(self):
        """Test handling when no telecom intents are detected"""
        intent_result_empty = {
            "intent": [],
            "inappropriate": False,
            "conversational_response": None,
            "confidence": 0
        }
        
        result = self.agent.retrieve_data(intent_result_empty)
        
        # Verify default response when no intents provided
        assert result["error"] == False
        assert "I can help you with plans" in result["message"]
        assert result["context"] == "no_intent"

    def test_retrieve_data_exception_handling(self):
        """Test graceful handling of unexpected exceptions during retrieval"""
        with patch.object(self.agent, '_get_query_for_intent', side_effect=Exception("Test error")):
            result = self.agent.retrieve_data(self.intent_result_single)
            
            # Verify exception caught and error response returned
            assert result["error"] == True
            assert "Data retrieval failed" in result["error_message"]
            assert result["context"] == "exception"

    def test_get_query_for_intent_all_mappings(self):
        """Test agent correctly maps all supported intents to API queries"""
        expected_mappings = {
            "events": "event offers",
            "usage": "usage data",
            "billing": "billing information",
            "recommended_plans": "recommended plans",
            "current_plan": "current plan",
            "plans": "available plans",
            "top_hots": "trending spots",
            "special_spots": "secret vip spots",
            "sports_events": "sports events"
        }
        
        # Verify each intent maps to correct API query
        for intent, expected_query in expected_mappings.items():
            assert self.agent._get_query_for_intent(intent) == expected_query

    def test_get_query_info(self):
        """Test query information retrieval for debugging/monitoring"""
        result = self.agent.get_query_info("usage")
        
        # Verify query info structure for monitoring
        assert result["intent"] == "usage"
        assert result["query"] == "usage data"
        assert result["api_url"] == self.mock_api_url

    @patch('src.data_retrieval_agent.requests.get')
    def test_call_api_success(self, mock_get):
        """Test successful external API call with proper response handling"""
        # Mock successful API response
        expected_data = {"results": {"test": "data"}}
        mock_response = Mock()
        mock_response.json.return_value = expected_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.agent._call_api("test query")
        
        # Verify successful API response returned
        assert result == expected_data

    @patch('src.data_retrieval_agent.requests.get')
    def test_call_api_network_error(self, mock_get):
        """Test handling of network connectivity errors during API calls"""
        from requests.exceptions import ConnectionError
        mock_get.side_effect = ConnectionError("Network error")
        
        result = self.agent._call_api("test query")
        
        # Verify network error handled gracefully
        assert result["error"] == True
        assert "API call failed" in result["error_message"]
        assert result["context"] == "network_error"

    @patch('src.data_retrieval_agent.requests.get')
    def test_call_api_unexpected_error(self, mock_get):
        """Test handling of unexpected exceptions during API calls"""
        mock_get.side_effect = ValueError("Unexpected error")
        
        result = self.agent._call_api("test query")
        
        # Verify unexpected errors handled gracefully
        assert result["error"] == True
        assert "Unexpected error" in result["error_message"]
        assert result["context"] == "unexpected_error"

    def test_initialization_with_env_var(self):
        """Test agent initialization using environment variable for API URL"""
        with patch.dict('os.environ', {'DATA_RETRIEVAL_API_URL': 'http://env-api.com'}):
            agent = DataRetrievalAgent()
            # Verify environment variable used for API URL
            assert agent.api_url == 'http://env-api.com'