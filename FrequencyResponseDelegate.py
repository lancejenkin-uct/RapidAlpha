#!/usr/bin/env python
""" Provides methods for the Frequency Response Delegate Object.

The frequency response delegate is used for the measurement of frequency of
the loudspeaker. It loads the default settings for measuring the frequency
response.  It then allows the user to adjust the settings and preform the
measurement.  It then displays the measured impulse and frequency response of
the loudspeaker.

"""

import logging
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from BaseDelegate import BaseDelegate
from FrequencyResponseController import FrequencyResponseController

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class FrequencyResponseDelegate(BaseDelegate, QThread):

    def __init__(self):
        """ Default constructor, loading the Frequency Response view
        """

        BaseDelegate.__init__(self)
        QThread.__init__(self)

        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating FrequencyResponseDelegate")

        self.measurement_settings = self.config_db.getSettings("frequency")
        self.window = FrequencyResponseController(self.measurement_settings,
            self.audio_devices)

        self._setupSignals()

    def _setupSignals(self):
        """ Connects the signals that are emitted to the correct slots.
        """
        self.logger.debug("Entering _setupSignals")

        self.window.startMeasurement.connect(self._newMeasurement)
        self.window.saveMeasurement.connect(self._saveMeasurement)
        self.window.loadMeasurement.connect(self._loadMeasurement)

    def _newMeasurement(self):
        """ Helper method to start a new measurement.

        The base newMeasurement method requires measurement_settings, this
        helper method retrieves the measurement settings from the view, and
        passes it to the newMeasurement method.
        """
        self.logger.debug("Entering _newMeasurement")

        measurement_window = self.sender()

        measurement_settings = self.window.measurement_settings

        measurement = self.newMeasurement(measurement_settings)

        measurement_window.measurement = measurement

        self.window._showSaveDialog("measurement")

    def _loadMeasurement(self, measurement_filename):
        """ Loads measurement from the specified filename.

        :param measurement_filename:
            The filename containing the measurement to load.
        :type measurement_filename:
            str
        """
        self.logger.debug("Entering _loadMeasurement")

        measurement_filename = str(measurement_filename)

        freq_response = self.loadFrequencyResponse(measurement_filename)

        self.window.freq_response = freq_response


        self.window.updateGraphs()

    def _saveMeasurement(self, measurement_filename):
        """ Saves a measurement that has been preformed to the specified
            filename.

        :param measurement_filename:
            The filename to save the measurement to.
        :type measurement_filename:
            str
        """
        self.logger.debug("Entering _saveMeasurement")

        measurement_window = self.sender()

        measurement = measurement_window.measurement

        self.saveMeasurement(measurement, measurement_filename)

        self._loadMeasurement(measurement_filename)

if __name__ == "__main__":
    logger = logging.getLogger("Alpha")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s - "
                                  "%(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    app = QApplication([])
    freq_response = FrequencyResponseDelegate()

    app.exec_()

