# Testing

For testing, I'm going to follow this [guide:](https://realpython.com/python-testing/#testing-your-code)

From Dosepy proyect folder, open a terminal and run $python -W ignore:ResourceWarning -m unittest discover -s tests -v

1.- List of all the features your application has, the different types of input it can accept, and the expected results. Now, every time you make a change to your code, you need to go through every single item on that list and check it.

Unittest requieres that:

You put tests into classes as methods.

You use a series of special assertion methods in the unittest. TestCase class instead of the built-in assert statment.