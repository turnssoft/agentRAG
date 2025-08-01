#!/usr/bin/env python3
"""
Simple runner script for the Meat Lovers pizza test.
This script provides an easy way to run the test without using unittest directly.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meat_lovers_test import run_meat_lovers_test
from logger import AgentLogger

def main():
    """Main function to run the Meat Lovers pizza test."""
    logger = AgentLogger(__name__)
    
    print("🍕 Meat Lovers Pizza Customer Feedback Test")
    print("=" * 50)
    
    try:
        # Run the test
        success = run_meat_lovers_test()
        
        if success:
            print("\n✅ Test completed successfully!")
            print("The RAG system successfully analyzed customer feedback for Meat Lovers pizza.")
        else:
            print("\n❌ Test failed!")
            print("Please check the logs for more details.")
            
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Unexpected error running test: {e}")
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 