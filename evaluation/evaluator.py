from typing import List, Dict
import json
import asyncio
from .metrics import RAGEvaluator

class EvaluationManager:
    def __init__(self, db_connection, openai_api_key: str):
        self.db = db_connection
        self.evaluator = RAGEvaluator(openai_api_key)
    
    async def evaluate_response(self, ground_truth_id: int) -> Dict:
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
            return {"error": "Response not found"}
            
        question, ground_truth, response, chunks_json = row
        chunks = json.loads(chunks_json)
        
        # Calculate metrics
        results = {
            "response_metrics": {
                "cosine_similarity": self.evaluator.calculate_cosine_similarity(
                    ground_truth, response
                ),
                "rouge_scores": self.evaluator.calculate_rouge_scores(
                    response, ground_truth
                ),
                "ai_evaluation": await self.evaluator.evaluate_response_with_ai(
                    question, ground_truth, response
                )
            },
            "chunks_evaluation": []
        }
        
        # Evaluate each chunk
        for chunk in chunks:
            chunk_eval = {
                "chunk": chunk,
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
        
        # Store evaluation results
        cursor.execute("""
            UPDATE rag_responses
            SET evaluation = ?
            WHERE ground_truth_id = ?
        """, (json.dumps(results), ground_truth_id))
        
        self.db.commit()
        return results 