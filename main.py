import sqlite3
import json
import requests
from dataclasses import dataclass
from typing import List, Optional
import csv
import os

@dataclass
class Config:
    api_endpoint: str
    api_key: Optional[str]
    auth_type: Optional[str]  # Add this line
    response_mapping: dict  # Defines how to extract response from API JSON
    chunks_mapping: dict   # Defines how to extract chunks from API JSON
    ground_truth_path: str
    request_config: dict

class Database:
    def __init__(self, db_path: str = "rag_evaluator.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ground_truth (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_responses (
                id INTEGER PRIMARY KEY,
                ground_truth_id INTEGER,
                response TEXT NOT NULL,
                chunks TEXT,
                evaluation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ground_truth_id) REFERENCES ground_truth (id)
            )
        """)
        self.conn.commit()

class RAGEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database()
    
    def load_ground_truth(self):
        with open(self.config.ground_truth_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Print column names for debugging
            print("CSV Columns:", reader.fieldnames)
            
            cursor = self.db.conn.cursor()
            for row in reader:
                cursor.execute(
                    "INSERT INTO ground_truth (question, answer) VALUES (?, ?)",
                    (row['Question'], row['Answer'])
                )
            self.db.conn.commit()
    
    def call_rag_api(self, question: str) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.config.api_key  # Direct API key without 'Bearer'
        }
        
        # Start with additional params from config
        payload = self.config.request_config.get('additional_params', {})
        
        # Add messages
        messages = []
        if system_message := self.config.request_config.get('system_message'):
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": question})
        payload["messages"] = messages
        
        # Add other configured parameters
        for key, value in self.config.request_config.items():
            if key not in ['system_message', 'additional_params']:
                payload[key] = value
        
        # Update use_context to be the context value
        if 'use_context' in payload:
            payload['context'] = payload.pop('use_context')
        
        try:
            print(f"Calling RAG API with payload: {payload}")  # Debug print
            response = requests.post(
                self.config.api_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling RAG API: {str(e)}")
            print(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
            raise

    def extract_response_and_chunks(self, api_response: dict) -> tuple[str, List[str]]:
        def get_nested_value(data: dict, path: List[str | int]) -> any:
            current = data
            for key in path:
                if isinstance(current, (dict, list)):
                    current = current[key]
                else:
                    raise ValueError(f"Cannot navigate further with key {key} in {current}")
            return current

        # Extract response
        try:
            response_data = get_nested_value(api_response, self.config.response_mapping['path'])
            if self.config.response_mapping['is_array']:
                response = ' '.join(response_data)
            else:
                response = str(response_data)
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Failed to extract response using mapping: {str(e)}")

        # Extract chunks
        try:
            chunks_data = get_nested_value(api_response, self.config.chunks_mapping['path'])
            if not self.config.chunks_mapping['is_array']:
                chunks_data = [chunks_data]
            
            chunks = []
            text_field = self.config.chunks_mapping['text_field']
            for chunk in chunks_data:
                if isinstance(chunk, dict) and text_field in chunk:
                    chunks.append(chunk[text_field])
                else:
                    raise ValueError(f"Chunk missing required field: {text_field}")
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Failed to extract chunks using mapping: {str(e)}")

        return response, chunks

    def process_ground_truth(self):
        cursor = self.db.conn.cursor()
        
        # Get all ground truth entries
        cursor.execute("SELECT id, question FROM ground_truth")
        ground_truth_entries = cursor.fetchall()
        
        for gt_id, question in ground_truth_entries:
            try:
                # Call RAG API and extract response/chunks
                api_response = self.call_rag_api(question)
                response, chunks = self.extract_response_and_chunks(api_response)
                
                # Store the results
                cursor.execute(
                    """
                    INSERT INTO rag_responses 
                    (ground_truth_id, response, chunks) 
                    VALUES (?, ?, ?)
                    """,
                    (gt_id, response, json.dumps(chunks))
                )
                self.db.conn.commit()
                
                print(f"Processed question {gt_id}: {question[:50]}...")
                
            except Exception as e:
                print(f"Error processing question {gt_id}: {str(e)}")
                continue
        
        print("Finished processing all ground truth questions")

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return Config(**config_data)

def main():
    config = load_config('config.json')
    evaluator = RAGEvaluator(config)
    
    # Load ground truth data if not already loaded
    cursor = evaluator.db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ground_truth")
    if cursor.fetchone()[0] == 0:
        print("Loading ground truth data...")
        evaluator.load_ground_truth()
    
    # Process all questions
    print("Starting to process questions through RAG system...")
    evaluator.process_ground_truth()

if __name__ == "__main__":
    main()
