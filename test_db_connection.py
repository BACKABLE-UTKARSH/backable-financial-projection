import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

try:
    print("Testing database connection...")
    conn = psycopg2.connect(
        host=os.getenv('PROJECTIONS_DB_HOST', 'memberchat-db.postgres.database.azure.com'),
        dbname=os.getenv('PROJECTIONS_DB_NAME', 'BACKABLE-FINANCIAL-PROJECTION'),
        user=os.getenv('PROJECTIONS_DB_USER', 'backable'),
        password=os.getenv('PROJECTIONS_DB_PASSWORD', 'Utkar$h007'),
        port=int(os.getenv('PROJECTIONS_DB_PORT', '5432')),
        connect_timeout=5
    )
    print("SUCCESS: Database connection established")
    conn.close()
    print("Connection closed successfully")
except Exception as e:
    print(f"ERROR: Database connection failed - {e}")
