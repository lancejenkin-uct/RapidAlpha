#!/usr/bin/env python
""" Provides methods for the Preference View

The preference window allows the user to change the settings used
for frequency response measurements as well as absorption measurements.
"""

import logging
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from PreferenceController import PreferenceController

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class PreferenceDelegate(QObject):
    # pyqtSignals
    finished = pyqtSignal()

    def __init__(self, measurement_settings):
        """ Default constructor

        @param measurement_settings: The measurement settings used in the
            measurement
        """
        QObject.__init__(self)
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Entering PreferenceDelegate")

        self.measurement_settings = measurement_settings

        self.window = PreferenceController(self.measurement_settings)
        self.window.show()
        self._setupSignals()

    def _setupSignals(self):
        """ Setup the signals connecting signals to slots """
        self.logger.debug("Entering _setupSignals")

        self.window.saveSettings.connect(self._saveSettings)

    def _saveSettings(self):
        """ Set the measurement_settings to the newly set settings """
        self.logger.debug("Entering _saveSettings")

        self.window._updateSettings()
        self.measurement_settings = self.window.settings
        self.finished.emit()

