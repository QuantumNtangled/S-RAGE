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
        self.evaluator = RAGEvaluator(llm_provider)
    
    async def evaluate_response(self, ground_truth_id: int) -> Dict:
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT gt.question, gt.answer, rr.response, rr.chunks
                FROM ground_truth gt
                JOIN rag_responses rr ON gt.id = rr.ground_truth_id
                WHERE gt.id = ?
            """, (ground_truth_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No data found for ground_truth_id {ground_truth_id}")
            
            question, ground_truth, response, chunks_json = row
            chunks = json.loads(chunks_json) if chunks_json else []
            
            # Debug print for AI evaluation inputs
            print("\nStarting AI evaluation with:")
            print(f"Question: {question[:100]}...")
            print(f"Ground Truth: {ground_truth[:100]}...")
            print(f"Response: {response[:100]}...")
            
            # First evaluate the main response
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
                    "semantic_similarity": self.evaluator.calculate_semantic_similarity(
                        ground_truth, response
                    ),
                    "ai_evaluation": await self.evaluator.evaluate_response_with_ai(
                        question=question,
                        ground_truth=ground_truth,
                        response=response  # Pass the actual response, not chunks
                    )
                },
                "chunks_evaluation": []
            }
            
            # Then evaluate chunks separately
            for chunk in chunks:
                chunk_eval = {
                    "chunk": chunk,
                    "semantic_similarity": self.evaluator.calculate_semantic_similarity(
                        ground_truth, chunk
                    ),
                    "completeness": await self.evaluator.evaluate_chunk_completeness(
                        chunk, ground_truth
                    )
                }
                results["chunks_evaluation"].append(chunk_eval)
            
            # Debug print for AI evaluation score
            print(f"\nAI Evaluation Score: {results['response_evaluation']['ai_evaluation']}")
            
            # Store in database
            cursor.execute("""
                UPDATE rag_responses 
                SET evaluation = ? 
                WHERE ground_truth_id = ?
            """, (json.dumps(results), ground_truth_id))
            self.db.commit()
            
            return results
            
        except Exception as e:
            print(f"Error in evaluate_response: {str(e)}")
            raise 