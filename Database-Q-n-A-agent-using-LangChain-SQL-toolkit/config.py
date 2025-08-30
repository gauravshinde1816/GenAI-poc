"""
Configuration file for Database Q&A Agent
Copy this file to .env and fill in your actual values
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Database Configuration
DEFAULT_DATABASE = "SQLite"  # Options: "SQLite", "MySQL"

# MySQL Configuration (optional)
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}

# AI Model Configuration
AI_MODEL_CONFIG = {
    "model": "gemma2-9b-it",
    "temperature": 0.1,
    "max_tokens": 4096
}

# Application Configuration
APP_CONFIG = {
    "page_title": "Database Q&A Agent",
    "page_icon": "🗄️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Validation
def validate_config():
    """Validate that required configuration is present"""
    errors = []
    
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is required. Please set it in your .env file")
    
    if DEFAULT_DATABASE == "MySQL":
        required_mysql_fields = ["user", "password", "database"]
        for field in required_mysql_fields:
            if not MYSQL_CONFIG.get(field):
                errors.append(f"MYSQL_{field.upper()} is required for MySQL connections")
    
    return errors

def get_config_status():
    """Get the current configuration status"""
    errors = validate_config()
    
    if errors:
        return {
            "status": "error",
            "errors": errors,
            "ready": False
        }
    
    return {
        "status": "ready",
        "errors": [],
        "ready": True
    } 