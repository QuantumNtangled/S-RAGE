import sqlite3

def check_database():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Check ground_truth table
        print("\nGround Truth Table:")
        cursor.execute("SELECT * FROM ground_truth")
        ground_truth_rows = cursor.fetchall()
        for row in ground_truth_rows:
            print(f"ID: {row[0]}")
            print(f"Answer: {row[1]}\n")
            
        # Check responses table with joins
        print("\nResponses with Ground Truth:")
        cursor.execute("""
            SELECT r.id, r.ground_truth_id, r.question, g.answer
            FROM responses r
            LEFT JOIN ground_truth g ON r.ground_truth_id = g.id
        """)
        response_rows = cursor.fetchall()
        for row in response_rows:
            print(f"Response ID: {row[0]}")
            print(f"Ground Truth ID: {row[1]}")
            print(f"Question: {row[2]}")
            print(f"Ground Truth Answer: {row[3]}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_database() 