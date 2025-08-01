# main.py
import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from vector import retriever
from logger import AgentLogger

# Initialize logger for this module
logger = AgentLogger(__name__)

# ─── Load environment variables ───────────────────────────────────────────────
logger.info("Loading environment variables...")
load_dotenv()

LLM_HOST = os.getenv("LLM_HOST")
# if LLM_STUDIO then LLM_HOST_LLM_STUDIO ELSE LLM_HOST_LLAMA_CPP
if LLM_HOST == "LLM_STUDIO":
    LLM_HOST = os.getenv("LLM_HOST_LLM_STUDIO")
    LLM_PATH = os.getenv("LLM_PATH_LLM_STUDIO")
    MODEL_NAME = os.getenv("OLLAMA_MODEL_LLM_STUDIO")
    logger.info("Using LLM Studio configuration")
else:
    LLM_HOST = os.getenv("LLM_HOST_LLAMA_CPP")
    LLM_PATH = os.getenv("LLM_PATH_LLAMA_CPP")
    MODEL_NAME = os.getenv("OLLAMA_MODEL_LLAMA_CPP")
    logger.info("Using Llama.cpp configuration")

SYSTEM_CONTENT      = os.getenv("ROLE_SYSTEM_CONTENT", "You are a helpful assistant.")
PROMPT_FILE         = os.getenv("PROMPT_FILE", "prompt.txt")

logger.info(f"LLM Host: {LLM_HOST}")
logger.info(f"Model: {MODEL_NAME}")
logger.debug(f"Prompt file: {PROMPT_FILE}")

# Read and prepare prompt template
logger.info(f"Loading prompt template from {PROMPT_FILE}")
try:
    with open(PROMPT_FILE, encoding="utf-8") as f:
        prompt_template = f.read().strip()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    logger.info("✓ Prompt template loaded successfully")
except FileNotFoundError:
    logger.error(f"❌ Prompt file not found: {PROMPT_FILE}")
    raise
except Exception as e:
    logger.error(f"❌ Error loading prompt template: {e}")
    raise

# ─── Local LLM via HTTP ─────────────────────────────────────────────────────
class LocalLLM(Runnable):
    def invoke(self, input):
        raw_prompt  = input["prompt"]
        # unwrap ChatPromptValue to plain text before JSON-serializing
        prompt_text = raw_prompt.content if hasattr(raw_prompt, "content") else str(raw_prompt)
        url = f"{LLM_HOST}/{LLM_PATH}"
        
        logger.debug(f"Sending request to LLM: {url}")
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user",   "content": prompt_text}
            ]
        }
        
        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            response_content = resp.json()["choices"][0]["message"]["content"]
            logger.debug("✓ LLM response received successfully")
            return response_content
        except requests.RequestException as e:
            logger.error(f"❌ Failed to get LLM response: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"❌ Unexpected LLM response format: {e}")
            raise
model = LocalLLM()

def run_chain(question: str) -> str:
    logger.info(f"Processing question: {question[:50]}{'...' if len(question) > 50 else ''}")
    
    # Retrieve relevant documents
    logger.debug("Retrieving relevant documents...")
    reviews = retriever.invoke(question)
    logger.info(f"✓ Retrieved {len(reviews)} relevant documents")
    
    # Generate prompt
    logger.debug("Generating full prompt...")
    full_prompt = prompt.invoke({"reviews": reviews, "question": question})
    
    # Get LLM response
    logger.debug("Getting LLM response...")
    response = model.invoke({"prompt": full_prompt})
    logger.info("✓ Question processed successfully")
    
    return response


def main():
    logger.info("🚀 Starting agentRAG interactive session")
    logger.info("Type 'q' to quit")
    
    try:
        while True:
            print("\n\n-------------------------------")
            q = input("Ask your question (q to quit): ")
            if q.strip().lower() == "q":
                logger.info("👋 User requested exit")
                break
            
            if not q.strip():
                logger.warning("Empty question received, please try again")
                continue
                
            try:
                response = run_chain(q)
                print("\n" + response + "\n")
            except Exception as e:
                logger.error(f"❌ Error processing question: {e}")
                print(f"\n❌ Sorry, there was an error processing your question: {e}\n")
                
    except KeyboardInterrupt:
        logger.info("👋 Session interrupted by user")
    except Exception as e:
        logger.critical(f"🚨 Critical error in main loop: {e}")
        raise
    finally:
        logger.info("📝 Session ended")


# ─── CLI Loop ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
