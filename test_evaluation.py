from evaluation.evaluator import EvaluationManager
from evaluation.llm_provider import LLMProvider
import sqlite3

def main():
    # Initialize components
    db = sqlite3.connect('rag_evaluator.db')
    llm_provider = LLMProvider()
    
    # Create evaluator
    evaluator = EvaluationManager(
        db_connection=db,
        llm_provider=llm_provider
    )
    
    print("Evaluation manager initialized successfully!")

if __name__ == "__main__":
    main() 