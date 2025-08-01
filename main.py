# main.py
import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.documents import Document  # Import Document
from vector import retriever
from logger import AgentLogger

# Initialize logger for this module
logger = AgentLogger(__name__)

# --- Load environment variables ---
# (This section remains the same)
logger.info("Loading environment variables...")
load_dotenv()

LLM_HOST_CONFIG = os.getenv("LLM_HOST")
if LLM_HOST_CONFIG == "LLM_STUDIO":
    LLM_HOST = os.getenv("LLM_HOST_LLM_STUDIO")
    LLM_PATH = os.getenv("LLM_PATH_LLM_STUDIO")
    MODEL_NAME = os.getenv("OLLAMA_MODEL_LLM_STUDIO")
    logger.info("Using LLM Studio configuration")
else:
    LLM_HOST = os.getenv("LLM_HOST_LLAMA_CPP")
    LLM_PATH = os.getenv("LLM_PATH_LLAMA_CPP")
    MODEL_NAME = os.getenv("OLLAMA_MODEL_LLAMA_CPP")
    logger.info("Using Llama.cpp configuration")

SYSTEM_CONTENT = os.getenv("ROLE_SYSTEM_CONTENT", "You are a helpful assistant.")
PROMPT_FILE = os.getenv("PROMPT_FILE", "prompt.txt")

logger.info(f"LLM Host: {LLM_HOST}")
logger.info(f"Model: {MODEL_NAME}")

# --- Load Prompt Template ---
# (This section remains the same)
logger.info(f"Loading prompt template from {PROMPT_FILE}")
try:
    with open(PROMPT_FILE, encoding="utf-8") as f:
        prompt_template = f.read().strip()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    logger.info("✓ Prompt template loaded successfully")
except FileNotFoundError:
    logger.error(f"❌ Prompt file not found: {PROMPT_FILE}")
    raise

# --- Custom LLM Runnable ---
# (This class remains the same)
class LocalLLM(Runnable):
    def invoke(self, input):
        # The 'input' here is now a fully formed prompt from the previous step in the chain
        prompt_value = input
        prompt_text = prompt_value.to_string()
        
        url = f"{LLM_HOST}/{LLM_PATH}"
        logger.debug(f"Sending request to LLM: {url}")
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": prompt_text}
            ],
        }
        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.debug("✓ LLM response received successfully")
            return content
        except requests.RequestException as e:
            logger.error(f"❌ Failed to get LLM response: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"❌ Unexpected LLM response format: {e}")
            raise

model = LocalLLM()

# --- NEW: Document Formatting Function ---
def format_docs(docs: list[Document]) -> str:
    """
    Formats a list of documents into a single string for the prompt.
    Includes the review content and key metadata for clarity.
    """
    logger.debug(f"Formatting {len(docs)} documents for prompt context.")
    formatted_strings = []
    for doc in docs:
        # Extract metadata with defaults in case they are missing
        title = doc.metadata.get('title', 'N/A')
        rating = doc.metadata.get('rating', 'N/A')
        
        # Create a clean, readable string for each document
        formatted_string = (
            f"Review: {doc.page_content}\n"
            f"Title: {title}\n"
            f"Rating: {rating}"
        )
        formatted_strings.append(formatted_string)
    
    # Join all formatted strings with a clear separator
    return "\n\n---\n\n".join(formatted_strings)

# --- NEW: Idiomatic LCEL RAG Chain ---
# This chain defines the entire RAG process in a clear, readable way.
rag_chain = (
    {"reviews": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
)

def run_query(question: str) -> str:
    """
    Invokes the RAG chain with the user's question.
    """
    logger.info(f"Processing question: '{question[:50]}{'...' if len(question) > 50 else ''}'")
    try:
        response = rag_chain.invoke(question)
        logger.info("✓ Question processed successfully")
        return response
    except Exception as e:
        logger.error(f"❌ Error invoking RAG chain: {e}")
        # Re-raise the exception to be handled in the main loop
        raise

def main():
    logger.info("🚀 Starting agentRAG interactive session")
    logger.info("Type 'q' to quit")
    
    try:
        while True:
            print("\n" + "="*50)
            q = input("Ask your question (q to quit): ")
            if q.strip().lower() == "q":
                logger.info("👋 User requested exit")
                break
            
            if not q.strip():
                logger.warning("Empty question received, please try again.")
                continue
                
            try:
                # Use the new chain invocation function
                response = run_query(q)
                print("\n💡 Response:\n" + response)
            except Exception as e:
                print(f"\n❌ Sorry, there was an error processing your question: {e}\n")
                
    except KeyboardInterrupt:
        logger.info("\n👋 Session interrupted by user")
    except Exception as e:
        logger.critical(f"🚨 Critical error in main loop: {e}")
        raise
    finally:
        logger.info("📝 Session ended")

# --- CLI Loop ---
if __name__ == "__main__":
    main()