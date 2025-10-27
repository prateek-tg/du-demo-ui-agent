import pytest
from unittest.mock import Mock, patch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_agent import DesignAgent


class TestDesignAgent:
    """Test suite for design agent external API interactions"""
    
    def setup_method(self):
        """Initialize test environment with mock design API"""
        self.mock_design_api_url = "http://design-api.example.com"
        self.agent = DesignAgent(design_api_url=self.mock_design_api_url)

    @patch('design_agent.requests.post')
    def test_get_design_success(self, mock_post):
        """Test successful UI design generation from external API"""
        # Mock successful design API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "templates": ["template1", "template2"],
            "styles": {"color": "blue"},
            "layout": "grid"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.agent.get_design(
            intent="events",
            type_value="Events",
            message="Show upcoming concerts"
        )
        
        # Verify successful design response
        assert result["error"] == False
        assert "Design API call successful" in result["message"]
        assert "templates" in result["results"]

    @patch('design_agent.requests.post')
    def test_get_design_http_error(self, mock_post):
        """Test handling of HTTP errors from design API"""
        # Create a proper requests exception
        from requests.exceptions import HTTPError
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_post.return_value = mock_response
        
        result = self.agent.get_design(
            intent="invalid",
            type_value="Invalid"
        )
        
        # Verify HTTP error handled gracefully
        assert result["error"] == True
        assert "Design API call failed" in result["error_message"]

    @patch('design_agent.requests.post')
    def test_get_design_network_error(self, mock_post):
        """Test handling of network connectivity errors"""
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError("Connection failed")
        
        result = self.agent.get_design(
            intent="events",
            type_value="Events"
        )
        
        # Verify network error handled gracefully
        assert result["error"] == True
        assert "Design API call failed" in result["error_message"]

    @patch('design_agent.requests.post')
    def test_get_design_with_empty_message(self, mock_post):
        """Test UI design generation with minimal user context"""
        # Mock successful response for empty message
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"design": "minimal"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.agent.get_design(
            intent="plans",
            type_value="Plans",
            message=""  # Empty message
        )
        
        # Verify minimal design generated successfully
        assert result["error"] == False
        assert "design" in result["results"]

    @patch('design_agent.requests.post')
    def test_get_design_unexpected_error(self, mock_post):
        """Test handling of unexpected exceptions during design generation"""
        mock_post.side_effect = ValueError("Unexpected error")
        
        result = self.agent.get_design(
            intent="events",
            type_value="Events"
        )
        
        # Verify unexpected errors handled gracefully
        assert result["error"] == True
        assert "Unexpected error" in result["error_message"]
        assert result["context"] == "unexpected_error"

    def test_initialization_with_env_var(self):
        """Test agent initialization using environment variable for API URL"""
        with patch.dict('os.environ', {'DESIGN_API_URL': 'http://env-design.com'}):
            agent = DesignAgent()
            # Verify environment variable used for design API URL
            assert agent.design_api_url == 'http://env-design.com'

    @patch('design_agent.requests.post')
    def test_get_design_request_payload_tracking(self, mock_post):
        """Test proper request payload construction and tracking"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"templates": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.agent.get_design(
            intent="billing",
            type_value="Billing",
            message="Show payment history"
        )
        
        # Verify request payload properly tracked in response
        assert result["error"] == False
        assert result["request_payload"]["intent"] == "billing"
        assert result["request_payload"]["type"] == "Billing"
        assert result["request_payload"]["message"] == "Show payment history"