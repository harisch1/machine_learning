import src
import sys
import unittest


def main(*args, **kwargs):

    test = kwargs.get("--run-test", True)
    if test:
        suite = unittest.TestLoader().discover(src.tests.__name__, pattern="test*.py")
        unittest.TextTestRunner(verbosity=2).run(suite)

args = sys.argv[1:]
kwargs = {arg: True for arg in args}

if __name__ == "__main__":
    main(*args, **kwargs)