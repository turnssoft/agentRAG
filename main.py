# main.py
import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
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
# (This class is now correct)
class LocalLLM(Runnable):
    def invoke(self, prompt_value, config=None):
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

# --- NEW: Refined Document Processing Function ---
def process_retrieved_documents(data: dict) -> dict:
    """
    Takes the retrieved docs and the question, filters the docs,
    formats them, and returns a dictionary ready for the prompt.
    """
    docs = data['docs']
    question = data['question']
    logger.debug(f"Filtering {len(docs)} documents based on question: '{question}'")

    pizza_keywords = ["meat lovers", "volcano", "pepperoni", "margherita", "four cheese"]
    relevant_keyword = next((kw for kw in pizza_keywords if kw in question.lower()), None)
    
    filtered_docs = docs
    if relevant_keyword:
        logger.info(f"Found relevant keyword: '{relevant_keyword}'. Filtering documents...")
        filtered_docs = [
            doc for doc in docs 
            if 'title' in doc.metadata and relevant_keyword in doc.metadata['title'].lower()
        ]
    else:
        logger.info("No specific pizza keyword found. Using all retrieved documents.")

    logger.info(f"Found {len(filtered_docs)} relevant documents after filtering.")

    # Format the filtered documents into a single string for the context
    formatted_context = "\n\n---\n\n".join([
        f"Review: {doc.page_content}\n"
        f"Title: {doc.metadata.get('title', 'N/A')}\n"
        f"Rating: {doc.metadata.get('rating', 'N/A')}"
        for doc in filtered_docs
    ])
    
    # **THIS IS THE KEY FIX**: Return a dictionary that matches the prompt's input variables
    return {"reviews": formatted_context, "question": question}

# --- NEW: Correct and Final LCEL RAG Chain ---
rag_chain = (
    # Step 1: Create a dictionary with the original question and the retrieved documents
    {"docs": retriever, "question": RunnablePassthrough()}
    # Step 2: Pass this dictionary to our processing function
    | RunnableLambda(process_retrieved_documents)
    # Step 3: The output of our function is now a correctly formatted dictionary for the prompt
    | prompt
    # Step 4: Pass the populated prompt to the model
    | model
)

def run_query(question: str) -> str:
    """Invokes the RAG chain with the user's question."""
    logger.info(f"Processing question: '{question[:50]}{'...' if len(question) > 50 else ''}'")
    try:
        response = rag_chain.invoke(question)
        logger.info("✓ Question processed successfully")
        return response
    except Exception as e:
        logger.error(f"❌ Error invoking RAG chain: {e}")
        raise

# --- main() function remains the same ---
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

if __name__ == "__main__":
    main()