import sqlite3

def check_database():
    try:
        conn = sqlite3.connect('rag_evaluator.db')
        cursor = conn.cursor()
        
        # Check table schemas
        print("\nTable Schemas:")
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schemas = cursor.fetchall()
        for schema in schemas:
            print(f"\n{schema[0]}")
            
        # Check ground_truth table
        print("\nGround Truth Table:")
        cursor.execute("SELECT COUNT(*) FROM ground_truth")
        count = cursor.fetchone()
        print(f"Total records: {count[0]}")
        
        cursor.execute("SELECT id, question, substr(answer, 1, 100) FROM ground_truth LIMIT 3")
        ground_truth_rows = cursor.fetchall()
        for row in ground_truth_rows:
            print(f"\nID: {row[0]}")
            print(f"Question: {row[1]}")
            print(f"Answer (truncated): {row[2]}...")
            
        # Check rag_responses table
        print("\nRAG Responses Table:")
        cursor.execute("SELECT COUNT(*) FROM rag_responses")
        count = cursor.fetchone()
        print(f"Total records: {count[0]}")
        
        cursor.execute("""
            SELECT r.id, r.ground_truth_id, substr(r.response, 1, 100), 
                   substr(g.answer, 1, 100) as ground_truth_answer
            FROM rag_responses r
            LEFT JOIN ground_truth g ON r.ground_truth_id = g.id
            LIMIT 3
        """)
        response_rows = cursor.fetchall()
        for row in response_rows:
            print(f"\nResponse ID: {row[0]}")
            print(f"Ground Truth ID: {row[1]}")
            print(f"Response (truncated): {row[2]}...")
            print(f"Ground Truth Answer (truncated): {row[3]}...")
            
        cursor.execute("SELECT id, evaluation FROM rag_responses WHERE evaluation IS NOT NULL LIMIT 1")
        eval_row = cursor.fetchone()
        if eval_row:
            print("\nSample Evaluation Data:")
            print(f"ID: {eval_row[0]}")
            print(f"Evaluation: {eval_row[1]}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_database() 