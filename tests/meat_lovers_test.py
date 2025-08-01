#!/usr/bin/env python3
"""
Test for Meat Lovers pizza customer feedback analysis.
This test demonstrates the RAG system's ability to answer questions about specific pizza types.
"""

import sys
import os
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import run_chain
from logger import AgentLogger

class TestMeatLoversPizza(unittest.TestCase):
    """Test class for Meat Lovers pizza customer feedback analysis."""
    
    def setUp(self):
        """Set up test environment and logger."""
        self.logger = AgentLogger(__name__)
        self.logger.info("Setting up Meat Lovers pizza test")
        
        # Test question about Meat Lovers pizza
        self.question = "What are customers saying about the Meat Lovers pizza?"
        
    def test_meat_lovers_customer_feedback(self):
        """Test the RAG system's ability to analyze customer feedback for Meat Lovers pizza."""
        self.logger.info(f"Starting test: {self.question}")
        
        try:
            # Run the RAG chain to get response
            self.logger.info("Calling run_chain with Meat Lovers question")
            response = run_chain(self.question)
            
            # Log the response for debugging
            self.logger.info(f"Received response (first 200 chars): {response[:200]}...")
            
            # Basic assertions to validate the response
            self.assertIsInstance(response, str, "Response should be a string")
            self.assertGreater(len(response), 0, "Response should not be empty")
            
            # Check for expected content in the response
            # The response should contain information about customer feedback
            self.assertTrue(
                any(keyword in response.lower() for keyword in [
                    "meat", "lovers", "pizza", "customer", "feedback", "review", "opinion"
                ]),
                "Response should contain relevant keywords about Meat Lovers pizza"
            )
            
            # Log success
            self.logger.info("✓ Meat Lovers pizza test completed successfully")
            
            # Print the response for manual review
            print(f"\n{'='*60}")
            print("MEAT LOVERS PIZZA CUSTOMER FEEDBACK ANALYSIS")
            print(f"{'='*60}")
            print(f"Question: {self.question}")
            print(f"{'='*60}")
            print(response)
            print(f"{'='*60}\n")
            
        except Exception as e:
            self.logger.error(f"❌ Test failed with error: {e}")
            self.fail(f"Test failed with exception: {e}")
    
    def test_response_quality(self):
        """Test that the response provides meaningful insights about Meat Lovers pizza."""
        self.logger.info("Testing response quality for Meat Lovers pizza")
        
        try:
            response = run_chain(self.question)
            
            # Check response quality indicators
            quality_indicators = [
                len(response) > 50,  # Response should be substantial
                not response.lower().startswith("i don't know"),  # Should not be a generic "don't know"
                not response.lower().startswith("i cannot"),  # Should not be a generic "cannot"
                "meat" in response.lower() or "lovers" in response.lower(),  # Should mention the pizza type
            ]
            
            quality_score = sum(quality_indicators)
            self.assertGreaterEqual(
                quality_score, 
                3, 
                f"Response quality score {quality_score}/4 is too low. Response: {response[:200]}..."
            )
            
            self.logger.info(f"✓ Response quality test passed (score: {quality_score}/4)")
            
        except Exception as e:
            self.logger.error(f"❌ Response quality test failed: {e}")
            self.fail(f"Response quality test failed: {e}")


def run_meat_lovers_test():
    """Convenience function to run the Meat Lovers pizza test."""
    logger = AgentLogger(__name__)
    logger.info("Running Meat Lovers pizza customer feedback test")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMeatLoversPizza)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log results
    if result.wasSuccessful():
        logger.info("✓ All Meat Lovers pizza tests passed")
    else:
        logger.error(f"❌ Some Meat Lovers pizza tests failed: {result.failures}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the test directly
    success = run_meat_lovers_test()
    sys.exit(0 if success else 1)