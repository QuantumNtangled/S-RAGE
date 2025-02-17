from flask import Flask, jsonify, request, render_template, send_from_directory, g
from evaluation.evaluator import EvaluationManager
from evaluation.llm_provider import LLMProvider
from main import RAGEvaluator, load_config, Database
import os
from dotenv import load_dotenv
import sqlite3
import json
import asyncio
from functools import partial, wraps

app = Flask(__name__)

# Load environment variables and config silently
load_dotenv()
config = load_config('config.json')
llm_provider = LLMProvider()
DATABASE = 'rag_evaluator.db'

def init_db():
    with app.app_context():
        db = get_db()
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
        # Log evaluation data for debugging
        if row[5]:  # evaluation column
            print("\nEvaluation data from database:")
            print(f"Ground Truth ID: {row[0]}")
            try:
                eval_data = json.loads(row[5])
                if 'response_evaluation' in eval_data:
                    print("AI Evaluation Score:", eval_data['response_evaluation'].get('ai_evaluation', 'Not found'))
            except json.JSONDecodeError as e:
                print(f"Error parsing evaluation JSON: {e}")
        
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

def run_async(func):
    """Decorator to run async functions in sync Flask routes"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

@app.route('/api/evaluate/<int:ground_truth_id>', methods=['POST'])
@run_async
async def evaluate_response(ground_truth_id):
    try:
        print(f"\nStarting AI evaluation for ground_truth_id: {ground_truth_id}")
        evaluator = get_evaluator()
        results = await evaluator.evaluate_response(ground_truth_id)
        
        # Log AI evaluation results
        if results and 'response_evaluation' in results:
            print("AI Evaluation Results:")
            print(f"Score: {results['response_evaluation'].get('ai_evaluation', 'Not found')}")
        
        results = await resolve_coroutines(results)
        
        # Store evaluation results
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
        print(f"Error in AI evaluation: {str(e)}")
        return jsonify({"error": str(e)}), 500

async def resolve_coroutines(obj):
    """Recursively resolve any coroutines in the object"""
    if isinstance(obj, dict):
        return {key: await resolve_coroutines(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [await resolve_coroutines(item) for item in obj]
    elif asyncio.iscoroutine(obj):
        return await obj
    return obj

# Initialize database before running the app
init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 