import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
PROJECTIONS_DB_HOST = os.getenv("PROJECTIONS_DB_HOST", "memberchat-db.postgres.database.azure.com")
PROJECTIONS_DB_NAME = os.getenv("PROJECTIONS_DB_NAME", "BACKABLE-FINANCIAL-PROJECTION")
PROJECTIONS_DB_USER = os.getenv("PROJECTIONS_DB_USER", "backable")
PROJECTIONS_DB_PASSWORD = os.getenv('PROJECTIONS_DB_PASSWORD', "Utkar$h007")
PROJECTIONS_DB_PORT = int(os.getenv("PROJECTIONS_DB_PORT", "5432"))

client_id = input("Enter client ID to clear cache (or 'all' for all clients): ").strip()

try:
    conn = psycopg2.connect(
        host=PROJECTIONS_DB_HOST,
        dbname=PROJECTIONS_DB_NAME,
        user=PROJECTIONS_DB_USER,
        password=PROJECTIONS_DB_PASSWORD,
        port=PROJECTIONS_DB_PORT
    )

    with conn.cursor() as cur:
        if client_id.lower() == 'all':
            cur.execute("DELETE FROM financial_projections_cache")
            conn.commit()
            print(f"✅ Cleared ALL cached projections")
        else:
            cur.execute("DELETE FROM financial_projections_cache WHERE client_id = %s", (client_id,))
            conn.commit()
            count = cur.rowcount
            print(f"✅ Cleared {count} cached projections for client {client_id}")

    conn.close()
    print("✅ Cache cleared successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
