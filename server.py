from flask import Flask, jsonify, request, send_from_directory
import os
from main import RAGEvaluator, load_config
from evaluation.evaluator import EvaluationManager
from evaluation.llm_provider import LLMProvider
from dotenv import load_dotenv

app = Flask(__name__, static_folder='static')
config = load_config('config.json')
evaluator = RAGEvaluator(config)

# Load environment variables
load_dotenv()

# Initialize the LLM provider
llm_provider = LLMProvider()

# Update evaluator initialization
evaluator = EvaluationManager(
    db_connection=evaluator.db,
    llm_provider=llm_provider
)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/process', methods=['POST'])
def process_questions():
    evaluator.process_ground_truth()
    return jsonify({"status": "success"})

@app.route('/api/results', methods=['GET'])
def get_results():
    cursor = evaluator.db.cursor()
    cursor.execute("""
        SELECT
            gt.question,
            gt.answer,
            rr.response,
            rr.chunks,
            rr.evaluation,
            rr.timestamp
        FROM ground_truth gt
        LEFT JOIN rag_responses rr ON gt.id = rr.ground_truth_id
        ORDER BY gt.id DESC
    """)
    
    results = []
    for row in cursor.fetchall():
        results.append({
            "question": row[0],
            "ground_truth": row[1],
            "response": row[2],
            "chunks": row[3],
            "evaluation": row[4],
            "timestamp": row[5]
        })
    
    return jsonify(results)

@app.route('/api/evaluate/<int:ground_truth_id>', methods=['POST'])
async def evaluate_response(ground_truth_id):
    try:
        results = await evaluator.evaluate_response(ground_truth_id)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True) 