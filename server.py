from flask import Flask, jsonify, request, render_template, send_from_directory, g
from evaluation.evaluator import EvaluationManager
from evaluation.llm_provider import LLMProvider
from main import RAGEvaluator, load_config, Database
import os
from dotenv import load_dotenv
import sqlite3
import json

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load configuration
config = load_config('config.json')

# Initialize the LLM provider
llm_provider = LLMProvider()  # Remove the auth_type parameter

# Database configuration
DATABASE = 'rag_evaluator.db'  # Update this to match your database path

def init_db():
    with app.app_context():
        db = get_db()
        # Create tables
        db.execute("""
            CREATE TABLE IF NOT EXISTS ground_truth (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL
            )
        """)
        
        db.execute("""
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
        db.commit()

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
            gt.id,
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
        print(f"Chunks from DB: {row[4]}")
        print(f"Evaluation from DB: {row[5]}")
        
        results.append({
            "id": row[0],
            "question": row[1],
            "ground_truth": row[2],
            "response": row[3],
            "chunks": json.loads(row[4]) if row[4] else [],
            "evaluation": json.loads(row[5]) if row[5] else None,
            "timestamp": row[6]
        })
    
    return jsonify(results)

@app.route('/api/evaluate/<int:ground_truth_id>', methods=['POST'])
def evaluate_response(ground_truth_id):
    try:
        print(f"Starting evaluation for ground_truth_id: {ground_truth_id}")
        evaluator = get_evaluator()
        results = evaluator.evaluate_response(ground_truth_id)
        print(f"Evaluation results: {results}")
        
        # Store evaluation results in database
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            UPDATE rag_responses 
            SET evaluation = ? 
            WHERE ground_truth_id = ?
        """, (json.dumps(results), ground_truth_id))
        db.commit()
        
        return jsonify(results)
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Initialize database before running the app
init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 