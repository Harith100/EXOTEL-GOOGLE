import requests
import json
import logging
import os 
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "gemini-1.5-flash"  # or gemini-1.5-pro
        self.generation_url = f"{self.base_url}/models/{self.model}:generateContent"
        
        # System prompt for Malayalam conversation
        self.system_prompt = """നിങ്ങൾ ഒരു സഹായകനായ AI അസിസ്റ്റന്റ് ആണ്. നിങ്ങൾ മലയാളത്തിൽ സംസാരിക്കുകയും ഉപയോക്താക്കളെ സഹായിക്കുകയും ചെയ്യും.

നിർദ്ദേശങ്ങൾ:
1. എല്ലായ്പ്പോഴും മലയാളത്തിൽ മറുപടി നൽകുക
2. വൈകാരികമായി സഹായകമായിരിക്കുക  
3. ഹ്രസ്വവും വ്യക്തവുമായ ഉത്തരങ്ങൾ നൽകുക
4. സങ്കീർണ്ണമായ ചോദ്യങ്ങൾക്ക് വിശദമായ വിശദീകരണം നൽകുക
5. സാംസ്കാരിക സന്ദർഭം മനസ്സിലാക്കുക
6. മര്യാദയുള്ളതും ബഹുമാനപ്പെട്ടതുമായ ഭാഷ ഉപയോഗിക്കുക

നിങ്ങൾ ചെയ്യേണ്ടത്:
- ചോദ്യങ്ങൾക്ക് ഉത്തരം നൽകുക
- വിവരങ്ങൾ വിശദീകരിക്കുക  
- സാധാരണ സഹായം നൽകുക
- സാങ്കേതിക സഹായം നൽകുക

നിങ്ങൾ ചെയ്യരുത്:
- ദോഷകരമായ ഉള്ളടക്കം സൃഷ്ടിക്കരുത്
- വ്യക്തിപരമായ വിവരങ്ങൾ ആവശ്യപ്പെടരുത്
- അനുചിതമായ ഉത്തരങ്ങൾ നൽകരുത്"""

    def get_response(self, user_input: str, context: List[Dict] = None) -> Optional[str]:
        """
        Get response from Gemini API
        
        Args:
            user_input: User's message in Malayalam
            context: Previous conversation context
            
        Returns:
            AI response in Malayalam or None if error
        """
        try:
            # Build conversation context
            messages = []
            
            # Add system prompt
            messages.append({
                "role": "user",
                "parts": [{"text": self.system_prompt}]
            })
            messages.append({
                "role": "model", 
                "parts": [{"text": "മനസ്സിലായി. ഞാൻ മലയാളത്തിൽ സഹായകമായി മറുപടി നൽകും."}]
            })
            
            # Add conversation history
            if context:
                for exchange in context[-3:]:  # Last 3 exchanges only
                    messages.append({
                        "role": "user",
                        "parts": [{"text": exchange.get("user", "")}]
                    })
                    messages.append({
                        "role": "model",
                        "parts": [{"text": exchange.get("assistant", "")}]
                    })
            
            # Add current user input
            messages.append({
                "role": "user",
                "parts": [{"text": user_input}]
            })
            
            # Prepare request payload
            payload = {
                "contents": messages,
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                    "stopSequences": []
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            # Make API request
            response = requests.post(
                f"{self.generation_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    
                    if 'content' in candidate and 'parts' in candidate['content']:
                        response_text = candidate['content']['parts'][0]['text'].strip()
                        
                        if response_text:
                            logger.info(f"Gemini response: {response_text[:100]}...")
                            return response_text
                        else:
                            logger.warning("Empty response from Gemini")
                            return "ക്ഷമിക്കണം, എനിക്ക് മറുപടി നൽകാൻ കഴിഞ്ഞില്ല. വീണ്ടും ചോദിക്കാമോ?"
                    else:
                        logger.error("Invalid response structure from Gemini")
                        return "ക്ഷമിക്കണം, എന്തോ പ്രശ്നം സംഭവിച്ചു."
                else:
                    logger.error("No candidates in Gemini response")
                    return "ക്ഷമിക്കണം, എനിക്ക് മറുപടി നൽകാൻ കഴിഞ്ഞില്ല."
                    
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return "ക്ഷമിക്കണം, സേവനത്തിൽ പ്രശ്നം സംഭവിച്ചു."
                
        except requests.exceptions.Timeout:
            logger.error("Gemini API timeout")
            return "ക്ഷമിക്കണം, പ്രതികരണം വൈകി. വീണ്ടും ശ്രമിക്കാമോ?"
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request error: {str(e)}")
            return "ക്ഷമിക്കണം, കണക്ഷൻ പ്രശ്നം സംഭവിച്ചു."
        except Exception as e:
            logger.error(f"Gemini processing error: {str(e)}")
            return "ക്ഷമിക്കണം, എന്തോ പ്രശ്നം സംഭവിച്ചു."
    
    def get_simple_response(self, user_input: str) -> Optional[str]:
        """
        Get a simple response without context
        
        Args:
            user_input: User's message
            
        Returns:
            AI response or None if error
        """
        return self.get_response(user_input, context=None)
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection successful
        """
        try:
            test_response = self.get_simple_response("നമസ്കാരം")
            return test_response is not None
            
        except Exception as e:
            logger.error(f"Gemini connection test failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model
        
        Returns:
            Model information
        """
        try:
            model_url = f"{self.base_url}/models/{self.model}?key={self.api_key}"
            response = requests.get(model_url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get model info: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}