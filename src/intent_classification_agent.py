#!/usr/bin/env python3
"""
Agent 1: Intent Classification Agent
Responsible only for understanding user intent
"""

import openai
import os
from dotenv import load_dotenv
from typing import Dict, List
from loguru import logger

load_dotenv()

class IntentClassificationAgent:
    """
    LLM-based intent classification agent for telecom services.
    
    This agent uses OpenAI's GPT model to classify user queries into predefined
    intent categories for telecom services. It handles vulgar language detection,
    ambiguous queries, and provides conversational responses when appropriate.
    
    Attributes:
        client (openai.OpenAI): OpenAI API client for LLM interactions
        valid_intents (List[str]): List of supported intent categories
        
    Supported Intents:
        - events: Entertainment and event offers
        - usage: Data usage information
        - billing: Billing information and payment history
        - recommended_plans: Personalized plan recommendations
        - current_plan: Current plan details
        - plans: Available plans and pricing
        - top_hots: Trending spots and popular locations
        - special_spots: Special VIP locations
        - sports_events: Sports events and games
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the Intent Classification Agent.
        
        Args:
            openai_api_key (str, optional): OpenAI API key. If not provided,
                will attempt to load from OPENAI_API_KEY environment variable.
        """
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        # List of valid intents
        self.valid_intents = [
            "events", "usage", "billing", "recommended_plans", 
            "current_plan", "plans", "top_hots", "special_spots", "sports_events"
        ]

    def classify_intent(self, user_message: str) -> dict:
        """
        Classify user message into appropriate intent category using LLM.
        
        This method uses OpenAI's GPT model to analyze user queries and determine
        the most appropriate intent category. It handles various scenarios including
        clear intents, ambiguous queries, inappropriate language, and out-of-scope
        requests.
        
        Args:
            user_message (str): The user's input message to classify
            
        Returns:
            dict: Classification result containing:
                - intent (List[str] | None): List of detected intents, None if no intent
                - inappropriate (bool): True if vulgar/offensive language detected
                - conversational_response (str | None): Response text for non-intent queries
                - confidence (float): Confidence score (1.0 for intents, 0 for conversational)
                
        Examples:
            >>> agent.classify_intent("show my data usage")
            {
                "intent": ["usage"],
                "inappropriate": False,
                "conversational_response": None,
                "confidence": 1.0
            }
            
            >>> agent.classify_intent("what's the weather like?")
            {
                "intent": None,
                "inappropriate": False,
                "conversational_response": "I can help with telecom services...",
                "confidence": 0
            }
            
        Note:
            - Temperature is set to 0.2 for consistent, focused responses
            - Max tokens limited to 50 to ensure concise outputs
            - Debug information is printed to console
        """
        try:
            # Get model name from environment variable
            model_name = os.getenv("OPENAI_MODEL")
            # Send user message to OpenAI LLM for intent classification
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "system",
                    "content": """You are an intent classification agent. Your task is to analyze the user's query and identify the correct intent from the provided list.

**Your Core Principle:** Your primary goal is to correctly identify the user's single, primary request. If you cannot do this with high confidence, you must respond in a way that seeks clarification or politely indicates the limits of your understanding. Your response should be context-aware, concise, and professional.

**Known Intents (Only output these if they are a perfect match):**
- events
- usage
- billing
- recommended_plans
- current_plan
- plans
- top_hots
- special_spots
- sports_events

**Decision Framework:**

1.  **Single Clear Intent:** If the user's query has one clear, primary intent that matches a known intent, output **only the intent name**. Ignore minor typos and spelling mistakes.
    *   *Example: "show my data usage" → `usage`*

2.  **Ambiguity & Multiple Intents:** If the query is ambiguous or contains multiple distinct intents, **do not guess**. Your response must ask the user to specify what they want by re-entering their query with a single, clear request. You may list the intents you detected to guide them.

3.  **Unclear or Unknown Intent:** If the query is nonsensical, out of scope, or does not match any known intent, politely indicate you didn't understand and ask the user to rephrase their question, optionally hinting at what you *can* help with.

4.  **Offensive Language:** If the query contains vulgar or offensive language but the primary intent is clear, you should:
    *   **First,** classify the intent correctly (output the intent name).
    *   **Second,** politely and professionally address the use of inappropriate language. Do not refuse help, but do set a boundary.

**Instruction:** Analyze the following user query. Apply the framework above to decide whether to output a single intent name or to generate a response that asks for a more specific, re-phrased query.
                    """
                }, {
                    "role": "user",
                    "content": user_message
                }],
                temperature=0.2,
                max_tokens=50
            )
            # Step 1: Extract LLM response text and clean it
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Query: '{user_message}' -> LLM Response: '{response_text}'")
            
            # Step 2: Normalize response for intent matching (lowercase for consistency)
            response_lower = response_text.lower().strip()
            
            # Step 3: Check if response is a direct intent match
            if response_lower in self.valid_intents:
                # Direct intent detected - return with high confidence
                return {
                    "error": False,
                    "intent": [response_lower],               # Single intent in list format
                    "inappropriate": False,                   # No inappropriate language detected
                    "conversational_response": None,          # No conversation needed
                    "confidence": 1.0,                       # Maximum confidence for exact match
                    "context": "intent_detected"
                }
            
            # Step 4: Check for inappropriate language with valid intent
            # LLM may respond with intent + politeness message (e.g., "usage - please be polite")
            for intent in self.valid_intents:
                if (intent in response_lower and 
                    ("polite" in response_lower or "language" in response_lower)):
                    # Intent detected but inappropriate language was used
                    return {
                        "error": False,
                        "intent": [intent],                   # Extract the valid intent
                        "inappropriate": True,                # Flag inappropriate language
                        "conversational_response": None,      # No additional conversation needed
                        "confidence": 1.0,                   # High confidence in intent despite language
                        "context": "inappropriate_language"
                    }
            
            # Step 5: No intent detected - treat as conversational response
            # LLM provided guidance, clarification, or out-of-scope response
            return {
                "error": False,
                "intent": None,                          # No telecom intent detected
                "inappropriate": False,                  # No inappropriate language
                "conversational_response": response_text, # LLM's helpful response to user
                "confidence": 0,                         # Zero confidence since no intent
                "context": "conversational_response"
            }
            
        except Exception as e:
            # Step 6: Handle LLM API errors (network, quota, authentication, etc.)
            logger.error(f"LLM classification failed: {e}")
            return {
                "error": True,
                "error_message": f"LLM classification failed: {str(e)}",
                "intent": None,
                "inappropriate": False,
                "conversational_response": "I'm having trouble understanding that. Could you please rephrase your question?",
                "confidence": 0,
                "context": "exception"
            }
