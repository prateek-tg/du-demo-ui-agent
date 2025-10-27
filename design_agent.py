#!/usr/bin/env python3
"""
Design Agent - Simplified
Calls external design API with user-provided intent, type, and message
"""

import requests
from typing import Dict, Any
from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()

class DesignAgent:
    """
    Simplified design agent for retrieving UI/UX design data from external services.
    
    This agent provides a clean interface for calling external design APIs with
    user-provided parameters. It handles HTTP requests, error management, and
    response processing for design-related data retrieval in telecom applications.
    
    Attributes:
        design_api_url (str): External design API endpoint URL
        
    Typical Use Cases:
        - Retrieve UI layouts for different intent categories
        - Get design templates for telecom service pages
        - Fetch styling information based on content type
        - Obtain personalized design elements based on user context
    """
    
    def __init__(self, design_api_url: str = None):
        """
        Initialize the Design Agent with API configuration.
        
        Args:
            design_api_url (str, optional): External design API endpoint URL.
                If not provided, loads from DESIGN_API_URL environment variable or .env file.
        Note:
            The default API endpoint is configured for telecom design services
            and expects intent, type, and optional message parameters.
        """
        # Load design API URL from environment variable or argument
        self.design_api_url = design_api_url or os.getenv("DESIGN_API_URL")
    
    def get_design(self, intent: str, type_value: str, message: str = "") -> Dict[str, Any]:
        """
        Retrieve design data from external API using user-provided parameters.
        
        This method sends a POST request to the external design API with the
        specified intent, type, and optional message to retrieve appropriate
        design data, templates, or styling information.
        
        Args:
            intent (str): The intent category for design retrieval
                (e.g., "events", "usage", "plans", "billing")
            type_value (str): The specific type or category of design
                (e.g., "Events", "Usage", "Plans", "Billing")
            message (str, optional): Additional context message that may
                influence design selection or personalization
                
        Returns:
            Dict[str, Any]: Design API response containing:
                - On success: Design data, templates, styling information
                - On failure: Error dictionary with error=True, message, request_payload
                
        Examples:
            >>> agent.get_design("events", "Events", "Show upcoming concerts")
            {
                "templates": [...],
                "styles": {...},
                "layout": {...}
            }
            
            >>> agent.get_design("invalid", "Invalid")
            {
                "error": True,
                "message": "Design API call failed: HTTP 404",
                "request_payload": {"intent": "invalid", "type": "Invalid"}
            }
            
        Note:
            - Uses 30-second timeout for API calls
            - All API interactions are logged to console
            - Message parameter is optional and only included if non-empty
            - Handles both network errors and unexpected exceptions gracefully
        """
        try:
            # Step 1: Build payload for API request
            payload = {
                "intent": intent,         # Required: Intent category for design matching
                "type": type_value       # Required: Specific design type or template category
            }
            
            # Step 2: Add optional message parameter if provided
            if message and message.strip():
                payload["message"] = message.strip()  # Context for design personalization
                
            # Step 3: Log the outgoing API request for debugging
            logger.debug(f"Calling design API with payload: {payload}")
            
            # Step 4: Make HTTP POST request to external design API
            response = requests.post(
                self.design_api_url,                              # Target design service endpoint
                json=payload,                                     # Request body as JSON
                timeout=30,                                       # 30-second timeout to prevent hanging
                headers={"Content-Type": "application/json"}     # Specify JSON content type
            )
            
            # Step 5: Log response status for monitoring
            logger.info(f"API Response Status: {response.status_code}")
            
            # Step 6: Check for HTTP errors (4xx, 5xx status codes)
            response.raise_for_status()  # Raises RequestException for HTTP errors
            
            # Step 7: Parse successful JSON response
            result = response.json()
            logger.success("Design API call successful")
            
            # Step 8: Return successful response with structured format
            return {
                "error": False,
                "message": "Design API call successful",
                "results": result,                # Actual design data from API
                "context": "success",
                "request_payload": payload        # Echo back the request for debugging
            }
        except requests.exceptions.RequestException as e:
            # Handle network-related errors (timeout, connection issues, HTTP errors)
            logger.error(f"API call failed: {str(e)}")
            return {
                "error": True,
                "error_message": f"Design API call failed: {str(e)}",
                "context": "network_error",           # Indicates network/HTTP issue
                "results": {},
                "request_payload": payload            # Include original request for debugging
            }
        except Exception as e:
            # Handle any unexpected errors (JSON parsing, etc.)
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "error": True,
                "error_message": f"Unexpected error: {str(e)}",
                "context": "unexpected_error",
                "results": {},
                "request_payload": payload
            }

