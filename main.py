# main.py

import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from vector import retriever

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()

LLM_HOST            = os.getenv("LLM_HOST", "http://localhost:12434")
LLM_ENGINE          = os.getenv("LLM_ENGINE", "llama.cpp")
MODEL_NAME          = os.getenv("OLLAMA_MODEL", "ai/llama3.2:latest")
SYSTEM_CONTENT      = os.getenv("ROLE_SYSTEM_CONTENT", "You are a helpful assistant.")

# ─── Load prompt template ────────────────────────────────────────────────────
def load_prompt_template():
    with open("prompt.txt", encoding="utf-8") as f:
        return f.read()

# ─── Prompt template ────────────────────────────────────────────────────────
PROMPT_TEMPLATE_STR = load_prompt_template()

# ─── Local LLM via HTTP ─────────────────────────────────────────────────────
class LocalLLM(Runnable):
    def invoke(self, input):
        raw_prompt  = input["prompt"]
        # unwrap ChatPromptValue to plain text before JSON-serializing
        prompt_text = raw_prompt.content if hasattr(raw_prompt, "content") else str(raw_prompt)
        url = f"{LLM_HOST}/engines/{LLM_ENGINE}/v1/chat/completions"
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user",   "content": prompt_text}
            ]
        }
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

model = LocalLLM()

# ─── Prompt & Chain ─────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_STR)

def run_chain(question: str) -> str:
    reviews     = retriever.invoke(question)
    full_prompt = prompt.invoke({"reviews": reviews, "question": question})
    return model.invoke({"prompt": full_prompt})

# ─── CLI Loop ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        print("\n\n-------------------------------")
        q = input("Ask your question (q to quit): ")
        if q.strip().lower() == "q":
            break
        print("\n" + run_chain(q) + "\n")
