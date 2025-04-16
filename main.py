# main.py
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

import cx_Oracle
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection details from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "1521"),
    "service_name": os.getenv("DB_SERVICE_NAME", "ORCLPDB"),
    "user": os.getenv("DB_USER", "TECH_SCHEMA"),
    "password": os.getenv("DB_PASSWORD", "5669"),
    "schema": os.getenv("DB_SCHEMA", "TECH_SCHEMA"),
}

# Mistral API key from environment variables
MISTRAL_API_KEY = "pNe4xqYCsQWRkN8ORRDfJ2vmQjUO6sHx"

# File paths
SCHEMA_FILE = "table_details.json"
VECTOR_INDEX_FILE = "faiss_index.bin"
VECTOR_NAMES_FILE = "table_names.json"
VECTOR_DESC_FILE = "table_descriptions.json"
LAST_FETCH_FILE = "last_fetch.txt"

# Available models
MISTRAL_MODELS = ["mistral-large-latest", "mistral-medium", "mistral-small"]

# Pydantic models for request/response
class NLQueryRequest(BaseModel):
    query: str
    model: str = "mistral-large-latest"

class SQLResponse(BaseModel):
    nl_query: str
    sql_query: str
    relevant_tables: List[Dict[str, Any]]
    execution_time: float

class StatusResponse(BaseModel):
    db_connected: bool
    last_updated: Optional[str] = None
    schema_tables_count: Optional[int] = None

class SchemaVectorSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.table_data = []
        self.table_names = []
    
    def fetch_schema(self):
        # Validate DB configuration
        if not all([DB_CONFIG["user"], DB_CONFIG["password"], DB_CONFIG["schema"]]):
            return None, "Database configuration incomplete. Please check environment variables."
        
        try:
            dsn_tns = cx_Oracle.makedsn(
                DB_CONFIG["host"], 
                DB_CONFIG["port"], 
                service_name=DB_CONFIG["service_name"]
            )
            
            connection = cx_Oracle.connect(
                user=DB_CONFIG["user"], 
                password=DB_CONFIG["password"], 
                dsn=dsn_tns
            )
            
            query = f"""
            SELECT 
                table_name, 
                column_name, 
                data_type, 
                data_length, 
                nullable 
            FROM 
                all_tab_columns 
            WHERE 
                owner = '{DB_CONFIG["schema"]}'
            ORDER BY 
                table_name, column_id
            """
            
            cursor = connection.cursor()
            cursor.execute(query)
            
            tables = {}
            for row in cursor:
                table_name, column_name, data_type, data_length, nullable = row
                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append({
                    'column_name': column_name,
                    'data_type': data_type,
                    'data_length': data_length,
                    'nullable': nullable
                })
            
            cursor.close()
            connection.close()
            
            # Check if any tables were found
            if not tables:
                return None, f"No tables found in schema '{DB_CONFIG['schema']}'"
            
            with open(SCHEMA_FILE, 'w') as json_file:
                json.dump(tables, json_file, indent=4)
            
            with open(LAST_FETCH_FILE, 'w') as f:
                f.write(datetime.now().isoformat())
            
            logger.info(f"Successfully fetched schema with {len(tables)} tables")
            return tables, None
        
        except cx_Oracle.DatabaseError as e:
            error_msg = f"Database error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def load_schema_data(self, schema_json):
        self.table_data = []
        self.table_names = []
        
        for table_name, columns in schema_json.items():
            self.table_names.append(table_name)
            
            column_descriptions = [
                f"{col['column_name']} ({col['data_type']}, {'NULL' if col['nullable'] == 'Y' else 'NOT NULL'})" 
                for col in columns
            ]
            
            table_desc = f"Table {table_name} with columns: {', '.join(column_descriptions)}"
            self.table_data.append(table_desc)
        
        self.build_index()
        
        with open(VECTOR_NAMES_FILE, 'w') as f:
            json.dump(self.table_names, f)
        
        with open(VECTOR_DESC_FILE, 'w') as f:
            json.dump(self.table_data, f)
        
        logger.info(f"Schema data loaded and indexed with {len(self.table_names)} tables")
    
    def build_index(self):
        if not self.table_data:
            self.index = None
            return
        
        table_vectors = self.model.encode(self.table_data)
        faiss.normalize_L2(table_vectors)
        
        vector_dimension = table_vectors.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)
        self.index.add(table_vectors)
        
        faiss.write_index(self.index, VECTOR_INDEX_FILE)
        logger.info(f"Vector index built with dimension {vector_dimension}")
    
    def load_from_disk(self):
        try:
            if os.path.exists(VECTOR_INDEX_FILE) and os.path.exists(VECTOR_NAMES_FILE) and os.path.exists(VECTOR_DESC_FILE):
                self.index = faiss.read_index(VECTOR_INDEX_FILE)
                
                with open(VECTOR_NAMES_FILE, 'r') as f:
                    self.table_names = json.load(f)
                
                with open(VECTOR_DESC_FILE, 'r') as f:
                    self.table_data = json.load(f)
                
                logger.info(f"Vector database loaded from disk with {len(self.table_names)} tables")
                return True
            
            logger.warning("Vector database files not found")
            return False
        except Exception as e:
            error_msg = f"Error loading vector database: {str(e)}"
            logger.error(error_msg)
            return False
    
    def search(self, query, top_k=5):
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.table_names) and scores[0][i] > 0:
                results.append({
                    "table_name": self.table_names[idx],
                    "similarity_score": float(scores[0][i]),
                    "description": self.table_data[idx]
                })
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.info(f"Search for '{query}' returned {len(results)} results")
        return results

def should_fetch_schema():
    if not os.path.exists(LAST_FETCH_FILE):
        return True
    
    try:
        with open(LAST_FETCH_FILE, 'r') as f:
            last_fetch_str = f.read().strip()
            last_fetch = datetime.fromisoformat(last_fetch_str)
            hours_since_update = (datetime.now() - last_fetch).total_seconds() / 3600
            logger.info(f"Hours since last schema update: {hours_since_update:.2f}")
            return hours_since_update > 24
    except Exception as e:
        logger.warning(f"Error checking schema freshness: {str(e)}")
        return True

def generate_sql(schema_text, nl_query, model="mistral-large-latest"):
    # Validate Mistral API key
    if not MISTRAL_API_KEY:
        return None, "Mistral API key not found. Please set the MISTRAL_API_KEY environment variable."
    
    prompt_template = PromptTemplate(
        input_variables=["schema", "query"],
        template="""
        You are an SQL expert. Convert the following natural language query into a detailed SQL query.
        
        Database Schema (including table details):
        {schema}
        
        User Query: {query}
        
        Follow these guidelines:
        1. Use appropriate JOINs based on the table relationships
        2. Include relevant WHERE clauses for filtering
        3. Use aliases for table names to improve readability
        4. Add appropriate aggregate functions (COUNT, SUM, AVG, etc.) if needed
        5. Include ORDER BY, GROUP BY, and HAVING clauses as necessary
        6. Include comments for complex parts of the query
        
        Return ONLY the SQL query without any explanations or code block formatting.
        """
    )
    
    try:
        # Use environment variable for API key
        llm = ChatMistralAI(model=model, api_key=MISTRAL_API_KEY)
        
        chain = LLMChain(llm=llm, prompt=prompt_template)
        sql_query = chain.run({"schema": schema_text, "query": nl_query})
        
        # Remove any existing formatting if present
        sql_query = sql_query.replace("sql", "").replace("", "").strip()
        
        logger.info(f"SQL query generated successfully using {model}")
        return sql_query, None
    except Exception as e:
        error_msg = f"Error generating SQL: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def execute_sql_query(sql_query):
    """Execute the SQL query and return results (for future implementation)"""
    # This is a placeholder function for future functionality
    try:
        dsn_tns = cx_Oracle.makedsn(
            DB_CONFIG["host"], 
            DB_CONFIG["port"], 
            service_name=DB_CONFIG["service_name"]
        )
        
        connection = cx_Oracle.connect(
            user=DB_CONFIG["user"], 
            password=DB_CONFIG["password"], 
            dsn=dsn_tns
        )
        
        cursor = connection.cursor()
        cursor.execute(sql_query)
        
        columns = [col[0] for col in cursor.description]
        results = []
        
        for row in cursor:
            results.append(dict(zip(columns, row)))
        
        cursor.close()
        connection.close()
        
        return results, None
    except Exception as e:
        error_msg = f"Error executing SQL: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

# Initialize FastAPI app
app = FastAPI(
    title="Natural Language to SQL API",
    description="Convert natural language queries to SQL using AI",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize schema search
schema_search = SchemaVectorSearch()
schema_loaded = schema_search.load_from_disk()

# Dependency to ensure schema is loaded
async def get_schema_search():
    if not schema_loaded and not os.path.exists(SCHEMA_FILE):
        # Try to fetch schema if not loaded
        schema_json, error = schema_search.fetch_schema()
        if not error:
            schema_search.load_schema_data(schema_json)
        else:
            raise HTTPException(status_code=503, detail=f"Schema not available: {error}")
    return schema_search

# Background task to update schema
def update_schema_background():
    if should_fetch_schema():
        schema_json, error = schema_search.fetch_schema()
        if not error:
            schema_search.load_schema_data(schema_json)
            logger.info("Schema updated in background task")
        else:
            logger.error(f"Error updating schema in background: {error}")

@app.on_event("startup")
async def startup_event():
    # Check if we need to fetch schema on startup
    if should_fetch_schema():
        schema_json, error = schema_search.fetch_schema()
        if not error:
            schema_search.load_schema_data(schema_json)
        else:
            logger.error(f"Error fetching schema on startup: {error}")

@app.get("/", tags=["Status"])
async def root():
    return {"message": "Natural Language to SQL API is running"}

@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status():
    is_connected = os.path.exists(SCHEMA_FILE)
    tables_count = None
    last_updated = None
    
    if is_connected:
        try:
            with open(SCHEMA_FILE, 'r') as f:
                schema_json = json.load(f)
                tables_count = len(schema_json)
            
            if os.path.exists(LAST_FETCH_FILE):
                with open(LAST_FETCH_FILE, 'r') as f:
                    last_fetch_str = f.read().strip()
                    last_fetch = datetime.fromisoformat(last_fetch_str)
                    last_updated = last_fetch.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
    
    return StatusResponse(
        db_connected=is_connected,
        last_updated=last_updated,
        schema_tables_count=tables_count
    )

@app.post("/update-schema", tags=["Schema"])
async def update_schema():
    schema_json, error = schema_search.fetch_schema()
    
    if error:
        raise HTTPException(status_code=500, detail=f"Error fetching schema: {error}")
    
    schema_search.load_schema_data(schema_json)
    
    return {
        "status": "success",
        "message": "Schema updated successfully",
        "tables_count": len(schema_json)
    }

@app.get("/models", tags=["Models"])
async def get_models():
    return {
        "models": MISTRAL_MODELS
    }

@app.get("/schema-tables", tags=["Schema"])
async def get_schema_tables(schema_search: SchemaVectorSearch = Depends(get_schema_search)):
    try:
        with open(SCHEMA_FILE, 'r') as f:
            schema_json = json.load(f)
        
        return {
            "tables_count": len(schema_json),
            "tables": schema_json
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading schema: {str(e)}")

@app.post("/generate-sql", response_model=SQLResponse, tags=["SQL"])
async def generate_sql_endpoint(
    request: NLQueryRequest, 
    background_tasks: BackgroundTasks,
    schema_search: SchemaVectorSearch = Depends(get_schema_search)
):
    import time
    start_time = time.time()
    
    # Schedule background schema update if needed
    background_tasks.add_task(update_schema_background)
    
    nl_query = request.query
    model = request.model
    
    if model not in MISTRAL_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {', '.join(MISTRAL_MODELS)}")
    
    # Search for relevant tables
    relevant_tables = schema_search.search(nl_query)
    
    if not relevant_tables:
        raise HTTPException(status_code=404, detail="No relevant tables found for your query")
    
    try:
        with open(SCHEMA_FILE, 'r') as f:
            schema_json = json.load(f)
        
        # Prepare schema text from relevant tables
        schema_text = ""
        for table in relevant_tables:
            schema_text += f"Table: {table['table_name']}\nColumns:\n"
            for col in schema_json[table["table_name"]]:
                nullable = "NULL" if col['nullable'] == 'Y' else "NOT NULL"
                schema_text += f"- {col['column_name']} ({col['data_type']}, {nullable})\n"
        
        # Generate SQL query
        sql_query, error = generate_sql(schema_text, nl_query, model)
        
        if error:
            raise HTTPException(status_code=500, detail=f"Error generating SQL: {error}")
        
        execution_time = time.time() - start_time
        
        return SQLResponse(
            nl_query=nl_query,
            sql_query=sql_query,
            relevant_tables=relevant_tables,
            execution_time=execution_time
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute-sql", tags=["SQL"])
async def execute_sql_endpoint(sql_query: str):
    results, error = execute_sql_query(sql_query)
    
    if error:
        raise HTTPException(status_code=500, detail=f"Error executing SQL: {error}")
    
    return {
        "status": "success",
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)