from flask import Flask, jsonify, request, send_from_directory
import os
from main import RAGEvaluator, load_config
from evaluation.evaluator import EvaluationManager
from evaluation.llm_provider import LLMProvider

app = Flask(__name__, static_folder='static')
config = load_config('config.json')
evaluator = RAGEvaluator(config)

# Initialize the LLM provider
llm_provider = LLMProvider()

# Update evaluator initialization
evaluator = EvaluationManager(
    evaluator.db.conn,
    llm_provider
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
    cursor = evaluator.db.conn.cursor()
    cursor.execute("""
        SELECT 
            gt.question,
            gt.answer,
            rr.response,
            rr.chunks,
            rr.timestamp
        FROM ground_truth gt
        LEFT JOIN rag_responses rr ON gt.id = rr.ground_truth_id
        ORDER BY gt.id
    """)
    
    results = []
    for row in cursor.fetchall():
        results.append({
            "question": row[0],
            "ground_truth": row[1],
            "response": row[2],
            "chunks": row[3],
            "timestamp": row[4]
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