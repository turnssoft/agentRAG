#!/bin/bash 
source ~/git/agentRAG/.env

OLLAMA_MODEL="$MODEL"

echo "OLLAMA_MODEL: $OLLAMA_MODEL"

curl $DOCKER_HOST/engines/llama.cpp/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$OLLAMA_MODEL"'",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    }' 