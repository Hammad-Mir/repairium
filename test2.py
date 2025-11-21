import os
import logging
from importlib import util
from dotenv import load_dotenv
from llmware.setup import Setup
from tokenizers import Tokenizer
from llmware.status import Status
from typing import Any, Dict, cast
from llmware.library import Library
from llmware.retrieval import Query
from llmware.models import ModelCatalog
from llmware.resources import CloudBucketManager
from llmware.configs import LLMWareConfig, ChromaDBConfig

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

def ensure_gpt2_tokenizer_exists():
    # 1. Get the base model repo path (e.g., .../llmware/models)
    local_model_repo = LLMWareConfig().get_model_repo_path()

    # 2. Define the specific gpt2 path and file we need
    gpt2_path = os.path.join(local_model_repo, "gpt2")
    tokenizer_file = os.path.join(gpt2_path, "tokenizer.json")

    # 3. Check if the file exists. If not, we need to fix it.
    if not os.path.exists(tokenizer_file):
        print(f"GPT2 tokenizer missing at: {tokenizer_file}")
        print("Attempting to download...")

        # 4. Ensure the base repo directory exists
        if not os.path.exists(local_model_repo):
            os.makedirs(local_model_repo, exist_ok=True)

        # 5. Ensure the 'gpt2' directory exists
        if not os.path.exists(gpt2_path):
            os.makedirs(gpt2_path, exist_ok=True)

        # 6. Download the assets
        try:
            CloudBucketManager().pull_single_model_from_llmware_public_repo(model_name="gpt2")
            print("GPT2 tokenizer downloaded successfully.")
        except Exception as e:
            print(f"Error downloading GPT2 assets: {e}")
            
    else:
        print("GPT2 tokenizer found. Proceeding...")

# test if tokenizer file is available
ensure_gpt2_tokenizer_exists()

# ===============================================================

def setup_library(library_name):

    """ Note: this setup_library method is provided to enable a self-contained example to create a test library """

    #   Step 1 - Create library which is the main 'organizing construct' in llmware
    print ("\nupdate: Creating library: {}".format(library_name))

    library = Library().create_new_library(library_name)

    #   check the embedding status 'before' installing the embedding
    embedding_record = library.get_embedding_status()
    print("embedding record - before embedding ", embedding_record)

    return library

def install_vector_embeddings(library, embedding_model_name):

    """ This method is the core example of installing an embedding on a library.
        -- two inputs - (1) a pre-created library object and (2) the name of an embedding model """

    library_name = library.library_name
    vector_db = LLMWareConfig().get_vector_db()

    print(f"\nupdate: Starting the Embedding: "
          f"library - {library_name} - "
          f"vector_db - {vector_db} - "
          f"model - {embedding_model_name}")

    #   *** this is the one key line of code to create the embedding ***
    library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db,batch_size=100, model_api_key=OPENAI_API_KEY)

    #   note: for using llmware as part of a larger application, you can check the real-time status by polling Status()
    #   --both the EmbeddingHandler and Parsers write to Status() at intervals while processing
    update = Status().get_embedding_status(library_name, embedding_model_name)
    print("update: Embeddings Complete - Status() check at end of embedding - ", update)

    embedding_record = library.get_embedding_status()

    print("\nupdate:  embedding record - ", embedding_record)

    return 0

# ===============================================================

LLMWareConfig().set_active_db("sqlite")
LLMWareConfig().set_vector_db("chromadb")
ChromaDBConfig().get_config("persistent_path")

# ==============================================================

library_name = "rmp_library"

filepath = r'docs/Manuals/Commercial Equipment/RMP (Rugged Mobile Power)'

embedding_model = "text-embedding-3-small"

# ===============================================================

library = setup_library(library_name)

parsing_output = library.add_file(filepath)

print (f"Step 4 - completed parsing - {parsing_output}")

install_vector_embeddings(library, embedding_model)