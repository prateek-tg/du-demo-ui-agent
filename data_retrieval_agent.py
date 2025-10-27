#!/usr/bin/env python3
"""
Agent 2: Data Retrieval Agent
Responsible only for retrieving data from API based on intent
"""

import requests
from typing import Dict, Any, List
from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()

class DataRetrievalAgent:
    """
    Data retrieval agent for telecom services with conversational memory.
    
    This agent is responsible for retrieving data from external APIs based on
    classified user intents. It maintains conversational memory to track API
    interactions and provides contextual responses. The agent handles multiple
    intents, API failures, and conversational responses gracefully.
    
    Attributes:
        api_url (str): External API endpoint for data retrieval
        conversation_memory (deque): Short-term memory storing last 5 interactions
        intent_to_query (Dict[str, str]): Mapping of intents to API query strings
        
    Supported Data Types:
        - Events: Entertainment and promotional events
        - Usage: Data consumption and billing usage
        - Plans: Available telecom plans and pricing
        - Billing: Payment history and billing information
        - Trending Spots: Popular locations and hot spots
        - Sports Events: Sports-related events and activities
    """
    
    def __init__(self, api_url: str = None):
        """
        Initialize the Data Retrieval Agent.
        
        Args:
            api_url (str, optional): External API endpoint URL. If not provided,
                loads from DATA_RETRIEVAL_API_URL environment variable or .env file.
        """
        # Load data retrieval API URL from environment variable or argument
        self.api_url = api_url or os.getenv("DATA_RETRIEVAL_API_URL")
        # Mapping intents to optimal API queries
        self.intent_to_query = {
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


    def retrieve_data(self, intent_result: dict, user_message: str = "") -> Dict[str, Any]:
        """
        Retrieve data from external APIs based on classified user intent.
        
        This is the main method that processes intent classification results and
        retrieves appropriate data from external services. It handles multiple
        scenarios including conversational responses, single/multiple intents,
        API failures, and data merging.
        
        Args:
            intent_result (dict): Result from intent classification containing:
                - intent: List of detected intents or None
                - inappropriate: Boolean flag for vulgar language
                - conversational_response: Text for non-intent queries
                - confidence: Classification confidence score
            user_message (str, optional): Original user message for context
        Returns:
            Dict[str, Any]: Structured response containing:
                - error (bool): True if API call failed
                - message (str): Success message or error description
                - results (Dict): Retrieved data organized by data type
                - queries_used (List[str], optional): API queries for debugging
        """
        try:
            # Step 1: Handle conversational responses from intent classification
            # When LLM provides guidance/clarification instead of detecting intent
            if intent_result.get("conversational_response"):
                return {
                    "error": False,
                    "message": intent_result["conversational_response"],  # Pass through LLM response
                    "results": {},                                        # No data to retrieve
                    "context": "conversational_response"
                }
            
            # Step 2: Handle cases where no intent was detected
            # Provide helpful guidance about available services
            intents = intent_result.get("intent", [])
            if not intents:
                return {
                    "error": False,
                    "message": "I can help you with plans, usage, events, billing, and trending spots. What would you like to know?",
                    "results": {},
                    "context": "no_intent"
                }
            
            # Step 3: Process all detected intents and retrieve data
            # Handle both single and multiple intent scenarios
            all_data = {}                    # Merged results from all API calls
            queries_used = []               # Track queries for debugging/transparency
            
            for intent in intents:
                # Step 3a: Map intent to appropriate API query
                query = self._get_query_for_intent(intent)
                queries_used.append(f"{intent}: {query}")
                
                # Step 3b: Make API call for this specific intent
                data = self._call_api(query)
                
                # Step 3c: Handle API call failures
                if data.get("error"):
                    logger.error(f"API error for intent '{intent}': {data.get('error_message')}")
                    return {
                        "error": True,
                        "error_message": data.get("error_message", "API error"),
                        "intent": intent,                           # Which intent failed
                        "query_used": query,                        # Which query failed
                        "context": "api_call_failed"
                    }
                
                # Step 3d: Merge successful API results
                # Combine data from multiple intents into single response
                if "results" in data:
                    for key, value in data["results"].items():
                        if key not in all_data:
                            # New data type - add directly
                            all_data[key] = value
                        elif isinstance(value, list) and isinstance(all_data[key], list):
                            # Both are lists - extend existing list
                            all_data[key].extend(value)
                        # Note: Non-list conflicts are overridden (last intent wins)
            
            # Step 4: Return successful multi-intent response
            return {
                "error": False,
                "message": "Success",
                "results": all_data,            # Merged data from all intents
                "queries_used": queries_used,   # All queries executed for transparency
                "context": "success"
            }
        except Exception as e:
            logger.error(f"Data retrieval failed: {e}")
            return {
                "error": True,
                "error_message": f"Data retrieval failed: {str(e)}",
                "context": "exception"
            }

    def _get_query_for_intent(self, intent: str) -> str:
        """
        Convert intent category to optimized API query string.
        
        Args:
            intent (str): Intent category from intent classification
            
        Returns:
            str: Optimized query string for external API call
            
        Note:
            Maps user intents to API-specific query terms that yield
            better results from the external data service. Returns
            "available plans" as default fallback for unknown intents.
        """
    # Map intent to API query string, fallback to 'available plans'
        return self.intent_to_query.get(intent, "available plans")

    def _call_api(self, query: str) -> Dict[str, Any]:
        """
        Execute HTTP GET request to external data API.
        
        Args:
            query (str): Query string to send to the external API
            
        Returns:
            Dict[str, Any]: API response data or error information containing:
                - On success: Parsed JSON response from external API
                - On failure: Error dictionary with error=True, error_message, query_used
                
        Raises:
            Does not raise exceptions - all errors are captured and returned
            as structured error dictionaries.
            
        Note:
            - Uses 30-second timeout for API calls
            - Handles both network errors and unexpected exceptions
            - Query parameter is sent as 'message' to the external API
            - All errors include the original query for debugging
        """
        try:
            # Step 1: Execute HTTP GET request to external data service
            response = requests.get(
                self.api_url,                         # External data service endpoint
                params={"message": query},            # Send query as 'message' parameter
                timeout=30                            # 30-second timeout to prevent hanging
            )
            
            # Step 2: Check for HTTP errors (4xx, 5xx status codes)
            response.raise_for_status()               # Raises RequestException for HTTP errors
            
            # Step 3: Parse and return successful JSON response
            return response.json()                    # Return data directly from external API
            
        except requests.exceptions.RequestException as e:
            # Handle network-related errors (timeout, connection, HTTP errors)
            logger.error(f"Network/API error: {e}")
            return {
                "error": True,
                "error_message": f"API call failed: {str(e)}",
                "query_used": query,                  # Include failed query for debugging
                "context": "network_error"
            }
        except Exception as e:
            # Handle unexpected errors (JSON parsing, etc.)
            logger.error(f"Unexpected error: {e}")
            return {
                "error": True,
                "error_message": f"Unexpected error: {str(e)}",
                "query_used": query,                  # Include failed query for debugging
                "context": "unexpected_error"
            }

    def get_query_info(self, intent: str) -> Dict:
        """
        Get detailed information about API query mapping for a specific intent.
        
        Args:
            intent (str): Intent category to get query information for
            
        Returns:
            Dict: Information dictionary containing:
                - intent (str): The input intent category
                - query (str): Mapped API query string
                - api_url (str): External API endpoint URL
                
        Example:
            >>> agent.get_query_info("usage")
            {
                "intent": "usage",
                "query": "usage data", 
                "api_url": "http://..."
            }
            
        Note:
            Useful for debugging, system monitoring, and understanding
            how intents are translated to API queries.
        """
        # Return mapping info for debugging and monitoring
        query = self._get_query_for_intent(intent)
        return {
            "intent": intent,
            "query": query,
            "api_url": self.api_url
        }

