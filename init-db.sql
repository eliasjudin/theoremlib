-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the schema
CREATE SCHEMA IF NOT EXISTS theoremlib;

-- Set up any other database configurations here
ALTER DATABASE theoremlib SET search_path TO theoremlib, public;

-- Create indexes
CREATE INDEX IF NOT EXISTS theorem_embedding_idx ON theorems 
USING ivfflat (theorem_embedding vector_l2_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS proof_embedding_idx ON theorems 
USING ivfflat (proof_embedding vector_l2_ops)
WITH (lists = 100);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS theorem_text_idx ON theorems 
USING gin(to_tsvector('english', theorem_text));

CREATE INDEX IF NOT EXISTS proof_text_idx ON theorems 
USING gin(to_tsvector('english', proof_text));