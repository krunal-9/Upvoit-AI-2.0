import os
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))
        
        # Provider-specific settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.grok_api_key = os.getenv("GROK_API_KEY")
        self.grok_base_url = os.getenv("GROK_BASE_URL", "https://api.grok.x.ai/v1") # Example base URL
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        # Model names
        self.fast_model_name = os.getenv("FAST_MODEL_NAME", "gpt-4o-mini")
        self.smart_model_name = os.getenv("SMART_MODEL_NAME", "gpt-4o")

    def get_llm(self, model_type: str = "smart"):
        """
        Factory method to get the LLM instance based on provider and model type.
        model_type: "fast" or "smart"
        """
        model_name = self.fast_model_name if model_type == "fast" else self.smart_model_name
        
        if self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            return ChatOpenAI(
                model=model_name,
                temperature=self.temperature,
                api_key=self.openai_api_key
            )
            
        elif self.provider == "gemini":
            if not self.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError("Please install langchain-google-genai to use Gemini models: pip install langchain-google-genai")

            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                google_api_key=self.google_api_key,
                convert_system_message_to_human=True # Sometimes needed for Gemini
            )
            
        elif self.provider == "grok":
            if not self.grok_api_key:
                raise ValueError("GROK_API_KEY is not set.")
            return ChatOpenAI(
                model=model_name,
                temperature=self.temperature,
                api_key=self.grok_api_key,
                base_url=self.grok_base_url
            )

        elif self.provider == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is not set.")
            return ChatOpenAI(
                model=model_name,
                temperature=self.temperature,
                api_key=self.deepseek_api_key,
                base_url=self.deepseek_base_url
            )
            
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

# Global instance for easy import
model_config = ModelConfig()

def get_fast_llm():
    return model_config.get_llm("fast")

def get_smart_llm():
    return model_config.get_llm("smart")
