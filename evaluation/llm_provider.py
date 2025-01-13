import os
from dotenv import load_dotenv
import json
import boto3
from typing import Dict, Any
from openai import AzureOpenAI, OpenAI
import aiohttp

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file {config_path}")

class LLMProvider:
    def __init__(self):
        load_dotenv()  # Load environment variables
        self.config = load_config('config.json')
        self.api_endpoint = self.config.get('api_endpoint')
        self.api_key = self.config.get('api_key')
        self.provider = os.getenv("LLM_PROVIDER", "azure")  # Default to azure if not specified
        
        # Initialize the appropriate client based on provider
        if self.provider == "azure":
            self._setup_azure()
        elif self.provider == "openai":
            self._setup_openai()
        elif self.provider == "bedrock":
            self._setup_bedrock()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def generate_completion(self, prompt: str) -> str:
        """Generate a completion using the configured LLM provider."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Please provide numerical scores or brief evaluations as requested."
            },
            {
                "role": "user",
                "content": str(prompt)  # Ensure prompt is a string
            }
        ]

        try:
            if self.provider == "azure":
                return await self._azure_completion(messages)
            elif self.provider == "openai":
                return await self._openai_completion(messages)
            elif self.provider == "bedrock":
                return await self._bedrock_completion(messages)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            print(f"Error in generate_completion: {str(e)}")
            raise

    def _setup_azure(self):
        azure_endpoint = os.getenv("AZURE_API_BASE")
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_api_version = os.getenv("AZURE_API_VERSION")
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.model_name = os.getenv("AZURE_MODEL_NAME", "gpt-4")
        self.model_version = os.getenv("AZURE_MODEL_VERSION", "0613")
        self.embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        
        if not all([azure_endpoint, azure_api_key, azure_api_version, self.deployment_name]):
            raise ValueError("Missing required Azure OpenAI configuration in environment variables")
            
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version
        )

    def _setup_openai(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
        self.model_version = os.getenv("OPENAI_MODEL_VERSION", "0613")
        
        if not openai_api_key:
            raise ValueError("Missing OpenAI API key in environment variables")
            
        self.client = OpenAI(
            api_key=openai_api_key
        )

    def _setup_bedrock(self):
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")
        self.model_id = os.getenv("AWS_MODEL_ID", "anthropic.claude-v2")
        
        if not all([aws_access_key, aws_secret_key, aws_region]):
            raise ValueError("Missing required AWS credentials in environment variables")
            
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

    async def _azure_completion(self, messages: list) -> str:
        try:
            print(f"Original messages: {messages}")  # Debug print
            
            # Format messages according to Azure OpenAI requirements
            formatted_messages = []
            for msg in messages:
                formatted_message = {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": str(msg["content"])  # Ensure content is a string
                        }
                    ]
                }
                formatted_messages.append(formatted_message)
            
            print(f"Sending formatted messages to Azure: {formatted_messages}")  # Debug print
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=formatted_messages,
                temperature=0,
                max_tokens=150
            )
            
            print(f"Response from Azure: {response}")  # Debug print
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Azure completion error: {str(e)}")
            print(f"Full error details: {e.__dict__}")  # Debug print
            raise

    async def _openai_completion(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=4000  # Increased for evaluation responses
        )
        return response.choices[0].message.content.strip()

    async def _bedrock_completion(self, messages: list) -> str:
        # Format messages for Bedrock (example for Claude)
        prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 4000,  # Increased for evaluation responses
                "temperature": 0
            })
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['completion'].strip() 