# Getting Started With Testing in Python
## A script for testing. Source: https://realpython.com/python-testing/

#   DEFINITIONS

##  Exploratory testing: Is a form of testing that is donde without a plan.
##  In an exploratory test, you're just exploring the app.

##  Integration testing: Testing multiple components.

##  Unit test: It is a smaller test, one that checks that a single component operates in the right way.

""" # Example of a unit test for a function.
def test_sum():
    assert sum([1,2,3]) == 6, "Shoud be 6"

# Test with a tuple.
def test_sum_tuple():
    assert sum((1,2,2)) == 6, "Shoud be 6"

if __name__ == "__main__":
    test_sum()
    test_sum_tuple()
    print("Everything passed.") """

import unittest

class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(
            sum([1,2,3]), 6, "Shoud be 6"
        )
    
    def test_sum_tupple(self):
        self.assertEqual(
            sum((1,2,2)), 6, "Shoud be 6"
        )

if __name__ == '__main__':
    unittest.main()