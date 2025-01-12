import os
from dotenv import load_dotenv
import json
import boto3
from typing import Dict, Any
from openai import AzureOpenAI, OpenAI

class LLMProvider:
    def __init__(self, config_path: str = "llm.config.json"):
        # Load environment variables
        load_dotenv()
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.provider = self.config.get("default_provider", "azure")
        self._setup_provider()

    def _setup_provider(self):
        if self.provider == "azure":
            self._setup_azure()
        elif self.provider == "openai":
            self._setup_openai()
        elif self.provider == "bedrock":
            self._setup_bedrock()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _setup_azure(self):
        azure_endpoint = os.getenv("AZURE_API_BASE")
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_api_version = os.getenv("AZURE_API_VERSION")
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        
        if not all([azure_endpoint, azure_api_key, azure_api_version, self.deployment_name]):
            raise ValueError("Missing required Azure OpenAI configuration in environment variables")
            
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version
        )

    def _setup_openai(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Missing OpenAI API key in environment variables")
            
        self.client = OpenAI(
            api_key=openai_api_key
        )
        self.model = self.config["openai"]["model"]
        self.embedding_model = "text-embedding-3-small"

    def _setup_bedrock(self):
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")
        
        if not all([aws_access_key, aws_secret_key, aws_region]):
            raise ValueError("Missing required AWS credentials in environment variables")
            
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        self.model_id = self.config["bedrock"]["model_id"]

    async def _azure_completion(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=0,
            max_tokens=5
        )
        return response.choices[0].message.content.strip()

    async def _openai_completion(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=5
        )
        return response.choices[0].message.content.strip()

    async def _bedrock_completion(self, messages: list) -> str:
        # Format messages for Bedrock (example for Claude)
        prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 5,
                "temperature": 0
            })
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['completion'].strip()

    async def generate_completion(self, messages: list) -> str:
        if self.provider == "azure":
            return await self._azure_completion(messages)
        elif self.provider == "openai":
            return await self._openai_completion(messages)
        elif self.provider == "bedrock":
            return await self._bedrock_completion(messages) 