import os
import sqlite3
import logging
from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_local_db():
    """Create a local SQLite database with sample data"""
    try:
        # create a database connection
        con = sqlite3.connect("tutorial.db")
        cur = con.cursor()

        # create a table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Check if data already exists
        cur.execute("SELECT COUNT(*) FROM products")
        count = cur.fetchone()[0]
        
        if count == 0:
            # insert sample data
            sample_products = [
                ("Laptop", 1000, "Electronics", "High-performance laptop for work and gaming"),
                ("Phone", 500, "Electronics", "Smartphone with latest features"),
                ("Tablet", 300, "Electronics", "Portable tablet for entertainment"),
                ("Keyboard", 100, "Accessories", "Mechanical keyboard for typing"),
                ("Mouse", 50, "Accessories", "Wireless optical mouse"),
                ("Monitor", 300, "Electronics", "24-inch HD monitor"),
                ("Speaker", 100, "Audio", "Bluetooth speaker with great sound"),
                ("Headphones", 50, "Audio", "Noise-cancelling headphones"),
                ("Printer", 300, "Office", "All-in-one printer and scanner"),
                ("Scanner", 100, "Office", "Document scanner for digitization"),
                ("Projector", 50, "Electronics", "Portable projector for presentations"),
                ("UPS", 300, "Electronics", "Uninterruptible power supply")
            ]
            
            cur.executemany(
                "INSERT INTO products (name, price, category, description) VALUES (?, ?, ?, ?)",
                sample_products
            )
            
            logger.info(f"Created {len(sample_products)} sample products")
        else:
            logger.info(f"Database already contains {count} products")

        con.commit()
        con.close()
        
        return True
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        raise Exception(f"Failed to create SQLite database: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise Exception(f"Failed to create database: {e}")

def test_database_connection(database_uri):
    """Test database connection before creating toolkit"""
    try:
        engine = create_engine(database_uri)
        
        # Test connection
        with engine.connect() as connection:
            # Try to execute a simple query
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
            
        logger.info("Database connection test successful")
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"Database connection failed: {e}")
        raise Exception(f"Database connection failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error testing connection: {e}")
        raise Exception(f"Connection test failed: {e}")

def create_database_toolkit(llm, database_uri):
    """Create a database toolkit with error handling"""
    try:
        logger.info(f"Creating database toolkit for: {database_uri}")
        
        # Test connection first
        test_database_connection(database_uri)
        
        # Create database instance
        database = SQLDatabase.from_uri(database_uri)
        
        # Create toolkit
        toolkit = SQLDatabaseToolkit(db=database, llm=llm)
        
        logger.info("Database toolkit created successfully")
        return toolkit
        
    except Exception as e:
        logger.error(f"Failed to create database toolkit: {e}")
        raise Exception(f"Failed to create database toolkit: {e}")

def get_database_info(database_uri):
    """Get information about the database schema and tables"""
    try:
        engine = create_engine(database_uri)
        database = SQLDatabase.from_uri(database_uri)
        
        # Get table names
        tables = database.get_table_names()
        
        # Get schema information
        schema_info = {}
        for table in tables:
            try:
                schema = database.get_table_info(table)
                schema_info[table] = schema
            except Exception as e:
                logger.warning(f"Could not get schema for table {table}: {e}")
                schema_info[table] = "Schema information unavailable"
        
        return {
            "tables": tables,
            "schema": schema_info,
            "connection_string": database_uri
        }
        
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        raise Exception(f"Failed to get database info: {e}")

def validate_query(query):
    """Basic validation for SQL queries"""
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    # Basic security checks
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    query_upper = query.upper()
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False, f"Query contains potentially dangerous keyword: {keyword}"
    
    return True, "Query is valid"



