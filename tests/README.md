# Tests Directory

This directory contains tests for the agentRAG project, specifically focused on testing the RAG (Retrieval-Augmented Generation) system's ability to answer questions about pizza customer feedback.

## Test Files

### `meat_lovers_test.py`

A comprehensive test that asks the question: **"What are customers saying about the Meat Lovers pizza?"**

This test demonstrates:

- Integration with the main RAG system (`run_chain` function)
- Proper logging using the project's `AgentLogger`
- Response validation and quality assessment
- Detailed output for manual review

### `run_meat_lovers_test.py`

A simple runner script that provides an easy way to execute the Meat Lovers pizza test without using unittest directly.

## How to Run the Tests

### Option 1: Using the Runner Script (Recommended)

```bash
cd ~/git/agentRAG
source ~/git/agentRAG/.env
source ~/git/agentRAG/venv/bin/activate
python tests/run_meat_lovers_test.py
```

### Option 2: Using unittest directly

```bash
cd ~/git/agentRAG
source ~/git/agentRAG/.env
source ~/git/agentRAG/venv/bin/activate
python -m unittest tests.meat_lovers_test -v
```

### Option 3: Running the test file directly

```bash
cd ~/git/agentRAG
source ~/git/agentRAG/.env
source ~/git/agentRAG/venv/bin/activate
python tests/meat_lovers_test.py
```

## Test Features

### Response Validation

The test validates that:

- The response is a non-empty string
- The response contains relevant keywords (meat, lovers, pizza, customer, feedback, etc.)
- The response provides meaningful insights

### Quality Assessment

The test includes a quality score based on:

- Response length (should be substantial)
- Not starting with generic "I don't know" responses
- Mentioning the specific pizza type
- Providing actual insights

### Logging

All test activities are logged using the project's centralized `AgentLogger`, making it easy to debug issues and track test execution.

### Output

The test provides detailed output including:

- The original question
- The full response from the RAG system
- Test results and quality scores
- Any errors or issues encountered

## Expected Behavior

When running the test, you should see:

1. Setup and initialization logs
2. The question being processed
3. The RAG system retrieving relevant documents
4. The LLM generating a response
5. The response being validated
6. A formatted output showing the question and answer
7. Test results indicating success or failure

## Troubleshooting

If the test fails, check:

1. Environment variables are properly set (`.env` file)
2. Virtual environment is activated
3. All dependencies are installed
4. The vector database is built and accessible
5. The LLM service is running and accessible

The test includes comprehensive error handling and logging to help identify issues.
