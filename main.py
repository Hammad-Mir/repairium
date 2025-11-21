import os
import httpx
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, status, File, UploadFile

from llmware.status import Status
from llmware.library import Library
from llmware.resources import CloudBucketManager
from llmware.configs import LLMWareConfig, ChromaDBConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Pydantic Models =============

class LibraryCreate(BaseModel):
    name: str = Field(..., description="Name of the library to create")
    
# class LibraryResponse(BaseModel):
#     name: str
#     embedding_status: dict

class EmbeddingStatusItem(BaseModel):
    embedding_status: str = "no embeddings"
    embedded_blocks: int = 0
    embedding_model: str = "NA"
    embedding_db: str = "NA"
    time_stamp: str = "NA"

class LibraryResponse(BaseModel):
    name: str
    embedding_status: List[EmbeddingStatusItem] = Field(default_factory=list)  # ✅ Correct
    file_count: Optional[int] = None
    created_at: Optional[str] = None
    
class LibraryListResponse(BaseModel):
    libraries: List[str]
    
class FileMetadata(BaseModel):
    filename: str
    # library: str
    # documentUrl: Optional[str] = None
    doc_type: Optional[str] = None
    custom_metadata: Optional[dict] = None

class AddFileRequest(BaseModel):
    filename: str = Field(..., description="Name of the file to add")
    library_name: str = Field(..., description="Name of the library")
    blob_uri: str = Field(..., description="URI/path to the file blob")
    # metadata: FileMetadata = Field(..., description="File metadata")
    
class AddFileResponse(BaseModel):
    filename: str
    library_name: str
    parsing_output: dict
    message: str

class DeleteLibraryResponse(BaseModel):
    library_name: str
    message: str
    
class EmbeddingRequest(BaseModel):
    library_name: str
    embedding_model: str = Field(default="text-embedding-3-small")

# ============= Utility Functions =============

def ensure_gpt2_tokenizer_exists():
    """Ensure GPT2 tokenizer is available for LLMware"""
    local_model_repo = LLMWareConfig().get_model_repo_path()
    gpt2_path = os.path.join(local_model_repo, "gpt2")
    tokenizer_file = os.path.join(gpt2_path, "tokenizer.json")
    
    if not os.path.exists(tokenizer_file):
        logger.info(f"GPT2 tokenizer missing at: {tokenizer_file}")
        logger.info("Attempting to download...")
        
        os.makedirs(local_model_repo, exist_ok=True)
        os.makedirs(gpt2_path, exist_ok=True)
        
        try:
            CloudBucketManager().pull_single_model_from_llmware_public_repo(model_name="gpt2")
            logger.info("GPT2 tokenizer downloaded successfully.")
        except Exception as e:
            logger.error(f"Error downloading GPT2 assets: {e}")
            raise
    else:
        logger.info("GPT2 tokenizer found.")

def install_vector_embeddings(library: Library, embedding_model_name: str, api_key: str):
    """Install vector embeddings for a library"""
    library_name = library.library_name
    vector_db = LLMWareConfig().get_vector_db()
    
    logger.info(f"Starting embedding: library={library_name}, vector_db={vector_db}, model={embedding_model_name}")
    
    library.install_new_embedding(
        embedding_model_name=embedding_model_name,
        vector_db=vector_db,
        batch_size=100,
        model_api_key=api_key
    )
    
    update = Status().get_embedding_status(library_name, embedding_model_name)
    logger.info(f"Embeddings complete: {update}")
    
    return library.get_embedding_status()

# ============= Application Lifecycle =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic"""
    # Startup
    logger.info("Starting up FastAPI application...")
    
    # Configure LLMware
    LLMWareConfig().set_active_db("sqlite")
    LLMWareConfig().set_vector_db("chromadb")
    chroma_path = ChromaDBConfig().get_config("persistent_path")
    print(f"data path {chroma_path}")
    load_dotenv()  # Load environment variables from .env file
    
    # Ensure tokenizer exists
    ensure_gpt2_tokenizer_exists()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# ============= FastAPI Application =============

app = FastAPI(
    title="Repairium RAG API",
    description="API for managing LLMware libraries and embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# ============= Health Check =============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "repairium-rag-api"}

# ============= Library Endpoints =============

@app.post("/libraries", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
async def create_library(library_data: LibraryCreate):
    try:
        library = Library().create_new_library(library_data.name)
        
        # Get embedding status (returns a list)
        embedding_status_raw = library.get_embedding_status() or []
        
        # Parse into typed models
        embedding_status = [
            EmbeddingStatusItem(**item) 
            for item in embedding_status_raw
        ]
        
        return LibraryResponse(
            name=library_data.name,
            embedding_status=embedding_status,  # Now accepts list
            file_count=0,
            created_at=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Error creating library: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create library: {str(e)}"
        )

@app.get("/libraries", response_model=LibraryListResponse)
async def get_all_libraries():
    """Get all libraries"""
    try:
        # Get all libraries from LLMware
        all_libraries = Library().get_all_library_cards()
        library_names = [lib.get("library_name") for lib in all_libraries if lib.get("library_name")]
        
        return LibraryListResponse(libraries=library_names)
    except Exception as e:
        logger.error(f"Error fetching libraries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch libraries: {str(e)}"
        )

@app.get("/libraries/{library_name}", response_model=LibraryResponse)
async def get_library(library_name: str):
    """Get specific library details"""
    try:
        library = Library().load_library(library_name)
        embedding_status_raw = library.get_embedding_status() or []
        # Normalize raw embedding status items into typed EmbeddingStatusItem instances
        embedding_status = [
            EmbeddingStatusItem(**item) for item in embedding_status_raw
        ]
        
        return LibraryResponse(
            name=library_name,
            embedding_status=embedding_status
        )
    except Exception as e:
        logger.error(f"Error fetching library {library_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library not found: {str(e)}"
        )

@app.delete("/libraries/{library_name}", response_model=DeleteLibraryResponse)
async def delete_library(library_name: str):
    """Delete a library"""
    try:
        logger.info(f"Deleting library: {library_name}")
        library = Library().load_library(library_name)
        library.delete_library(library_name, confirm_delete=True)
        
        return DeleteLibraryResponse(
            library_name=library_name,
            message=f"Library '{library_name}' deleted successfully"
        )
    except Exception as e:
        logger.error(f"Error deleting library {library_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete library: {str(e)}"
        )

# ============= File Operations =============

@app.post("/libraries/files", response_model=AddFileResponse)
async def add_file_to_library(request: AddFileRequest):
    """
    Add a file to a library using blob URI and metadata
    Downloads file from URL/MinIO, then processes with LLMware
    """
    temp_file_path = None
    
    try:
        logger.info(f"Adding file to library: {request.library_name}")
        logger.info(f"Blob URI: {request.blob_uri}")
        logger.info(f"File: {request.filename}")
        
        # Load or create library
        try:
            library = Library().load_library(request.library_name)
            logger.info(f"Loaded existing library: {request.library_name}")
        except:
            logger.info(f"Creating new library: {request.library_name}")
            library = Library().create_new_library(request.library_name)
        
        # Download file from blob URI to temporary location
        logger.info(f"Downloading file from: {request.blob_uri}")
        temp_file_path = await download_file_from_uri(
            request.blob_uri, 
            request.filename
        )
        
        # Process file with LLMware using local path
        logger.info(f"Processing file with LLMware: {temp_file_path}")
        parsing_output = library.add_file(temp_file_path)
        
        logger.info(f"File parsing complete: {parsing_output}")
        
        return AddFileResponse(
            filename=request.filename,
            library_name=request.library_name,
            parsing_output=parsing_output or {},
            message="File added successfully to library"
        )
        
    except httpx.HTTPError as e:
        logger.error(f"Failed to download file from {request.blob_uri}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download file from blob storage: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error adding file to library: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add file to library: {str(e)}"
        )
    finally:
        # Cleanup: Remove temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


async def download_file_from_uri(uri: str, filename: str) -> str:
    """
    Download file from URI (HTTP/HTTPS/MinIO/S3) to temporary location
    Returns local file path
    """
    # Create temp directory if it doesn't exist
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, f"llmware_{filename}")
    
    # Download file
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout
        response = await client.get(uri, follow_redirects=True)
        response.raise_for_status()
        
        # Write to temporary file
        with open(local_path, 'wb') as f:
            f.write(response.content)
    
    return local_path

@app.post("/libraries/embed")
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for a library"""
    try:
        logger.info(f"Creating embeddings for library: {request.library_name}")
        
        library = Library().load_library(request.library_name)
        
        # Get OpenAI API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OPENAI_API_KEY not configured"
            )
        
        embedding_status = install_vector_embeddings(
            library=library,
            embedding_model_name=request.embedding_model,
            api_key=api_key
        )
        
        return {
            "library_name": request.library_name,
            "embedding_model": request.embedding_model,
            "embedding_status": embedding_status,
            "message": "Embeddings created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create embeddings: {str(e)}"
        )

# ============= Error Handlers =============

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )