import tracking_analysis

import numpy

import unittest

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.fitter = tracking_analysis.PolynomialFit()

class TestTrackingAnalysis(unittest.TestCase):
    def setUp(self):
        self.fitter = tracking_analysis.PolynomialFit()

if __name__ == "__main__":
    unittest.main()