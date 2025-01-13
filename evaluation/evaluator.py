from typing import List, Dict
import json
import asyncio
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from metrics import RAGEvaluator  # Try local import first
except ImportError:
    try:
        from evaluation.metrics import RAGEvaluator  # Try package import
    except ImportError:
        from .metrics import RAGEvaluator  # Try relative import

class EvaluationManager:
    def __init__(self, db_connection, llm_provider):
        self.db = db_connection
        self.evaluator = RAGEvaluator(llm_provider=llm_provider)
    
    async def evaluate_response(self, ground_truth_id: int) -> Dict:
        print(f"Starting evaluation for ground truth ID: {ground_truth_id}")
        cursor = self.db.cursor()
        
        # Get the question, ground truth, response, and chunks
        cursor.execute("""
            SELECT 
                gt.question,
                gt.answer,
                rr.response,
                rr.chunks
            FROM ground_truth gt
            JOIN rag_responses rr ON gt.id = rr.ground_truth_id
            WHERE gt.id = ?
        """, (ground_truth_id,))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No response found for ground truth ID {ground_truth_id}")
            
        question, ground_truth, response, chunks = row
        print(f"Retrieved data for evaluation:")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Response: {response}")
        print(f"Chunks: {chunks}")
        
        chunks_list = json.loads(chunks) if chunks else []
        
        results = {
            "response_evaluation": {
                "relevance": await self.evaluator.calculate_relevance(
                    question, response
                ),
                "completeness": await self.evaluator.calculate_completeness(
                    ground_truth, response
                ),
                "consistency": await self.evaluator.calculate_consistency(
                    ground_truth, response
                ),
                "fluency": await self.evaluator.calculate_fluency(
                    response
                ),
                "rouge_scores": self.evaluator.calculate_rouge_scores(
                    response, ground_truth
                ),
                "cosine_similarity": self.evaluator.calculate_cosine_similarity(
                    ground_truth, response
                ),
                "ai_evaluation": await self.evaluator.evaluate_response_with_ai(
                    question, ground_truth, response
                )
            },
            "chunks_evaluation": []
        }
        
        # Evaluate each chunk
        for chunk in chunks_list:
            chunk_eval = {
                "cosine_similarity": self.evaluator.calculate_cosine_similarity(
                    ground_truth, chunk
                ),
                "rouge_scores": self.evaluator.calculate_rouge_scores(
                    chunk, ground_truth
                ),
                "completeness": await self.evaluator.evaluate_chunk_completeness(
                    chunk, ground_truth
                )
            }
            results["chunks_evaluation"].append(chunk_eval)
        
        return results 