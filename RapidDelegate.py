#!/usr/bin/env python
""" Provides methods for the Rapid Delegate Object.

The rapid delegate is used for the rapid measurement of a material.  It loads
the default settings to do the measurement, as well as to analyze the response.
"""

import logging
from multiprocessing import Pool

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from BaseDelegate import BaseDelegate
from RapidController import RapidController
from PreferenceDelegate import PreferenceDelegate

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class RapidDelegate(BaseDelegate, QThread):

    def __init__(self):
        """ Constructor for RapidDelegate """
        BaseDelegate.__init__(self)
        QThread.__init__(self)
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating RapidDelegate")

        self.window = RapidController(self.measurement_settings,
                                    self.audio_devices)

        self._setupSignals()

    def _setupSignals(self):
        """ Connect the required signals to the correct slots. """
        self.logger.debug("Entering _setupSignals")

        self.window.startMeasurement.connect(self._newMeasurement)
        self.window.loadMeasurement.connect(self._loadMeasurement)
        self.window.saveMeasurement.connect(self._saveMeasurement)

        self.window.showPreferences.connect(self._showPreferences)

    def _showPreferences(self):
        """ Show the preference dialog """
        self.logger.debug("Entering _showPreferences")

        if self.window.alpha is  None:
            self.preferences = PreferenceDelegate(self.measurement_settings)
        else:
            self.preferences = PreferenceDelegate(self.window.alpha.measurement_settings)
        self.preferences.finished.connect(self._setSettings)

    def _setSettings(self):
        """ Set the settings from the preference dialog """
        self.logger.debug("Entering _setSettings")

        self.measurement_settings = self.preferences.measurement_settings
        self.window.measurement_settings = self.measurement_settings
        self.window.updateWidgets()

        if self.window.alpha is not None:
            self.window.alpha.measurement_settings = self.preferences.measurement_settings
            self.window.alpha.determineAlpha()
            self.window.update()


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

    def _loadMeasurement(self, measurement_filename):
        """ Loads measurement from the specified filename.

        :param measurement_filename:
            The filename containing the measurement to load.
        :type measurement_filename:
            str
        """
        self.logger.debug("Entering _loadMeasurement")

        measurement_filename = str(measurement_filename)

        alpha = self.loadAbsorptionCoefficient(measurement_filename)

        self.window.alpha = alpha
        self.window.grapher.measurement_settings = alpha.measurement_settings
        self.window.setWindowTitle("Rapid Alpha - %s" % (measurement_filename))
        self.window.update()

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
    alpha = RapidDelegate()

    app.exec_()

