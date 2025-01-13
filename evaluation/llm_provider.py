import os
from dotenv import load_dotenv
import json
import boto3
from typing import Dict, Any
from openai import AzureOpenAI, OpenAI
import aiohttp

class LLMProvider:
    def __init__(self):
        self.config = load_config('config.json')
        self.api_endpoint = self.config.get('api_endpoint')
        self.api_key = self.config.get('api_key')

    async def generate_completion(self, prompt: str) -> str:
        """Generate a completion using the configured LLM provider."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Please provide numerical scores or brief evaluations as requested."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        if self.provider == "azure":
            return await self._azure_completion(messages)
        elif self.provider == "openai":
            return await self._openai_completion(messages)
        elif self.provider == "bedrock":
            return await self._bedrock_completion(messages)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

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
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=0,
            max_tokens=4000  # Increased for evaluation responses
        )
        return response.choices[0].message.content.strip()

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