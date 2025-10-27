#!/usr/bin/env python3
"""
Main System: Orchestrates the 2-agent architecture
Agent 1: Intent Classification
Agent 2: Data Retrieval
"""

from src.intent_classification_agent import IntentClassificationAgent
from src.data_retrieval_agent import DataRetrievalAgent
from typing import Dict, Any
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

class Orchestrator:
    """
    Orchestrator for the two-agent architecture in telecom service processing.
    
    This system coordinates between the Intent Classification Agent and Data Retrieval
    Agent to provide a complete pipeline for processing user queries. It manages the
    flow from intent detection through data retrieval and response generation.
    
    Architecture:
        - Agent 1 (Intent): Classifies user queries into intent categories
        - Agent 2 (Data): Retrieves relevant data from external APIs
        - System: Orchestrates the flow and manages error handling
        
    Attributes:
        intent_agent (IntentClassificationAgent): Agent for intent classification
        data_agent (DataRetrievalAgent): Agent for data retrieval
        
    Features:
        - Handles multiple intents in single queries
        - Manages inappropriate language detection and response
        - Provides conversational responses for unclear queries
        - Comprehensive error handling and logging
    """
    
    def __init__(self, openai_api_key: str = None, api_url: str = None):
        """
        Initialize the Two-Agent System with both agents.
        
        Args:
            openai_api_key (str, optional): OpenAI API key for intent classification.
                If not provided, attempts to load from OPENAI_API_KEY environment variable.
            api_url (str, optional): External data API endpoint URL.
                If not provided, uses default telecom data service endpoint.
                
        Note:
            Both agents are initialized with their respective configurations and
            are ready to process queries immediately after initialization.
        """
        # Initialize both agents
        self.intent_agent = IntentClassificationAgent(openai_api_key)
        self.data_agent = DataRetrievalAgent(api_url)
        
        # Initialize conversation history storage (last 5 conversations)
        self.conversation_history = []
    
    def process_query(self, user_message: str) -> Dict[str, Any]:
        """
        Process user query through the complete two-agent pipeline.
        
        This is the main method that orchestrates the entire query processing
        workflow, from intent classification through data retrieval to final
        response generation. Handles all scenarios including single/multiple
        intents, conversational responses, and error conditions.
        
        Args:
            user_message (str): The user's input message to process
            
        Returns:
            Dict[str, Any]: Comprehensive response containing:
                - error (bool): True if processing failed
                - message (str): Success message, conversational response, or error description
                - results (Dict): Retrieved data organized by type (events, usage, etc.)
                - queries_used (List[str], optional): API queries used for debugging
                
        Processing Pipeline:
            1. Intent Classification: Analyze user message for intent(s)
            2. Response Handling: Process conversational or inappropriate responses
            3. Data Retrieval: Fetch data for detected intents from external APIs
            4. Response Assembly: Structure final response with all retrieved data
            
        Examples:
            >>> system.process_query("show my data usage")
            {
                "error": False,
                "message": "Success",
                "results": {"usage": [...]},
                "queries_used": ["usage: usage data"]
            }
            
            >>> system.process_query("hello there")
            {
                "error": False,
                "message": "I can help you with telecom services...",
                "results": {}
            }
            
        Note:
            - System handles errors gracefully without crashing
            - User message is passed to data agent for additional context
        """
        try:
            # Step 1: Classify intent using Agent 1 (returns dict with intent, inappropriate flag, etc.)
            intent_result = self.intent_agent.classify_intent(user_message)
            # Step 2: Retrieve data using Agent 2 (handles all cases) - pass user message for abusive word check
            data = self.data_agent.retrieve_data(intent_result, user_message)
            # If any error in data retrieval, propagate with uniform structure
            if data.get("error"):
                logger.error(f"Data retrieval error: {data.get('error_message', 'Unknown error')}")
                return {
                    "error": True,
                    "error_message": data.get("error_message", "Data retrieval error"),
                    "context": data.get("context", "data_agent"),
                    "results": data.get("results", {}),
                    "queries_used": data.get("queries_used", []),
                }
            # Return the structured response
            return {
                "error": False,
                "message": data.get("message", "Success"),
                "results": data.get("results", {}),
                "queries_used": data.get("queries_used", []),
                "context": data.get("context", "success")
            }
        except Exception as e:
            logger.error(f"System error: {e}")
            return {
                "error": True,
                "error_message": f"System error: {str(e)}",
                "context": "exception",
                "results": {},
                "queries_used": []
            }
    
    def get_system_info(self) -> Dict:
        """
        Retrieve comprehensive information about system configuration and status.
        
        Returns:
            Dict: System information containing:
                - intent_agent (str): Description of intent classification agent
                - data_agent (str): Description of data retrieval agent
                - api_url (str): External data API endpoint URL
                - available_intents (List[str]): List of supported intent categories
        """
        return {
            "intent_agent": "IntentClassificationAgent with OpenAI GPT-4o-mini",
            "data_agent": "DataRetrievalAgent with API integration",
            "api_url": self.data_agent.api_url,
            "available_intents": self.intent_agent.valid_intents
        }
    
    def add_conversation(self, user_message: str, response: Dict[str, Any]) -> None:
        """
        Add a conversation to the history, maintaining only the last 5 conversations.
        
        Args:
            user_message (str): The user's input message
            response (Dict[str, Any]): The system's response
        """
        from datetime import datetime
        
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "response": response
        }
        
        # Add to conversation history
        self.conversation_history.append(conversation)
        
        # Keep only the last 5 conversations
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
    
    def get_conversation_history(self) -> Dict:
        """
        Retrieve and categorize conversation history for API response.
        
        This method transforms the stored conversation history into three distinct
        categories for better organization and analysis:
        - Intent Classifications: All intent detection results from query processing
        - API Calls: Data retrieval operations with external APIs 
        - Design Calls: UI/form generation requests through design agent
        
        Returns:
            Dict: Categorized conversation history containing:
                - intent_classifications (List): Intent detection results with confidence
                - api_calls (List): Data retrieval operations with query details
                - design_calls (List): Design generation requests with parameters
                
        Note:
            - Single conversation may appear in multiple categories
            - Design calls are identified by agent_type="design" field
            - Only conversations with actual data are included in api_calls
        """
        # Initialize category containers
        intent_classifications = []
        api_calls = []
        design_calls = []
        
        # Process each stored conversation and categorize based on content
        for conv in self.conversation_history:
            # Category 1: Design Agent Calls
            # Check for design agent responses (identified by agent_type="design")
            if conv["response"].get("agent_type") == "design":
                design_info = {
                    "timestamp": conv["timestamp"],
                    "user_message": conv["user_message"],
                    "intent": conv["response"].get("intent"),           # Design intent (e.g., "bill_payment")
                    "design_type": conv["response"].get("design_type"), # UI type (e.g., "payment_form")
                    "error": conv["response"].get("error", False),      # Whether design generation failed
                    "results_available": bool(conv["response"].get("results"))  # Whether design data returned
                }
                design_calls.append(design_info)
            
            # Category 2: Intent Classifications
            # Extract intent detection results from query processing (not design calls)
            elif "intent" in conv["response"]:
                intent_info = {
                    "timestamp": conv["timestamp"],
                    "user_message": conv["user_message"],
                    "intent": conv["response"]["intent"],                       # Detected intent (e.g., "data_usage")
                    "confidence": conv["response"].get("confidence", 0.0),      # Classification confidence score
                    "inappropriate": conv["response"].get("inappropriate", False)  # Language appropriateness flag
                }
                intent_classifications.append(intent_info)
            
            # Category 3: API Calls (Data Retrieval Operations)
            # Extract successful data retrieval operations (exclude design results)
            if ("results" in conv["response"] and 
                conv["response"]["results"] and 
                conv["response"].get("agent_type") != "design"):
                
                api_call_info = {
                    "timestamp": conv["timestamp"],
                    "user_message": conv["user_message"],
                    "queries_used": conv["response"].get("queries_used", []),   # External API queries executed
                    "results_count": len(conv["response"]["results"])          # Number of data items retrieved
                }
                api_calls.append(api_call_info)
        
        return {
            "intent_classifications": intent_classifications,
            "api_calls": api_calls,
            "design_calls": design_calls
        }
    
    def clear_conversation_memory(self) -> None:
        """
        Clear conversation memory.
        """
        self.conversation_history.clear()
        logger.info("Conversation memory cleared")
        pass
    
def main():
    """
    Command-line interface for testing the Two-Agent System.
    
    Provides an interactive CLI for testing query processing capabilities.
    Initializes the system, checks for required environment variables,
    and provides a simple loop for user input and response display.
    
    Environment Requirements:
        - OPENAI_API_KEY: Required for intent classification
        
    Usage:
        python orchestrator.py
        
    Commands:
        - Any text: Process as query through both agents
        - 'quit', 'exit', 'q': Exit the program
        - Ctrl+C: Force exit
        
    Note:
        This is primarily for development and testing purposes.
        Production deployments should use the FastAPI interface.
    """
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Error: Set OPENAI_API_KEY in .env file")
        return
    
    # Initialize the system
    system = Orchestrator()
    
    logger.info("2-Agent System Ready")
    logger.info("Agent 1: Intent Classification")
    logger.info("Agent 2: Data Retrieval")
    logger.info("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Process through both agents and return only the data
            result = system.process_query(user_input)
            logger.info(result)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()