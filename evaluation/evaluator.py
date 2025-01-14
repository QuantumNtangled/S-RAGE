from typing import List, Dict
import json
import asyncio
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from metrics import RAGEvaluator  # Try local import first
    from llm_provider import MainEvalProvider  # Add this import
except ImportError:
    try:
        from evaluation.metrics import RAGEvaluator  # Try package import
        from evaluation.llm_provider import MainEvalProvider  # And this one
    except ImportError:
        from .metrics import RAGEvaluator  # Try relative import
        from .llm_provider import MainEvalProvider  # And this one

class EvaluationManager:
    def __init__(self, db_connection, llm_provider):
        self.db = db_connection
        self.evaluator = RAGEvaluator(llm_provider)
        self.main_eval_provider = MainEvalProvider()  # Create the main eval provider
    
    async def evaluate_response(self, ground_truth_id: int) -> Dict:
        try:
            cursor = self.db.cursor()
            # First get just the main evaluation data
            cursor.execute("""
                SELECT gt.question, gt.answer, rr.response
                FROM ground_truth gt
                JOIN rag_responses rr ON gt.id = rr.ground_truth_id
                WHERE gt.id = ?
            """, (ground_truth_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No data found for ground_truth_id {ground_truth_id}")
            
            question, ground_truth, response = row
            
            # Use main_eval_provider for AI evaluation
            ai_score = await self.evaluator.evaluate_response_with_ai(
                question=question,
                ground_truth=ground_truth,
                response=response
            )
            print(f"\nMain AI Evaluation Score (before storing): {ai_score} (type: {type(ai_score)})")
            
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
                    "ai_evaluation": ai_score  # This should now be a string
                }
            }
            
            # Now get and evaluate chunks separately
            cursor.execute("""
                SELECT chunks FROM rag_responses WHERE ground_truth_id = ?
            """, (ground_truth_id,))
            chunks_row = cursor.fetchone()
            chunks = json.loads(chunks_row[0]) if chunks_row and chunks_row[0] else []
            
            chunks_evaluation = []
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
                chunks_evaluation.append(chunk_eval)
            
            results["chunks_evaluation"] = chunks_evaluation
            
            print(f"\nFinal results before DB storage:")
            print(f"AI Evaluation in results: {results['response_evaluation']['ai_evaluation']} (type: {type(results['response_evaluation']['ai_evaluation'])})")
            
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