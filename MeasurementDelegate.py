#!/usr/bin/env python
""" Provides the delegate for the Measurement Delegate

The measurement delegate preforms a measurement, and saves to the specified location.
"""

import logging
from BaseDelegate import BaseDelegate

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class AnalysisDelegate(BaseDelegate):

    # Default Constructor
    def __init__(self):
        BaseDelegate.__init__(self)
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating AnalysisDelegate")

    def startMeasurement(self, measurement_file):
        self.logger.debug("Starting new measurement")
