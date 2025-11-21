
from llmware.configs import LLMWareConfig, ChromaDBConfig
import inspect

print("LLMWareConfig methods:")
# print([m for m in dir(LLMWareConfig) if not m.startswith("_")])

print("\nChromaDBConfig methods:")
print([m for m in dir(ChromaDBConfig) if not m.startswith("_")])

LLMWareConfig().set_active_db("sqlite")

LLMWareConfig().set_vector_db("chromadb")

ChromaDBConfig().set_config("lite", True)
ChromaDBConfig().set_config("db_path", "./chromadb")

try:
    c = LLMWareConfig()
    print("\nCurrent active db:", c.get_active_db())
    print("Current vector db:", c.get_vector_db())
except Exception as e:
    print(e)
