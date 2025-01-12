from flask import Flask, jsonify, request, render_template, send_from_directory, g
from evaluation.evaluator import EvaluationManager
from evaluation.llm_provider import LLMProvider
from main import RAGEvaluator, load_config, Database
import os
from dotenv import load_dotenv
import sqlite3

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load configuration
config = load_config('config.json')

# Initialize the LLM provider
llm_provider = LLMProvider()

# Database configuration
DATABASE = 'rag_evaluation.db'  # Update this to match your database path

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

def get_evaluator():
    if 'evaluator' not in g:
        g.evaluator = EvaluationManager(
            db_connection=get_db(),
            llm_provider=llm_provider
        )
    return g.evaluator

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/results', methods=['GET'])
def get_results():
    db = get_db()
    cursor = db.cursor()
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
        evaluator = get_evaluator()
        results = await evaluator.evaluate_response(ground_truth_id)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 