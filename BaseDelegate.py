#!/usr/bin/env python
""" Provides methods for the Base Delegate Object.

The base delegate object provides methods to organize the measurement of the 
absorption coefficient.  The different environments the measurement is preformed
in will inherent the BaseDelegate object and provide the synchronization of the
measurement.
"""

import logging

# Alpha Classes
from AbsorptionCoefficient import AbsorptionCoefficient
from AudioIO import AudioIO
from ConfigDb import ConfigDb
from FrequencyResponse import FrequencyResponse
from Measurement import Measurement
from MlsDb import MlsDb
from MeasurementDb import MeasurementDb
from SignalGenerator import SignalGenerator

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"

class BaseDelegate(object):

    def __init__(self):
        """ Constructor for BaseDelegate """
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating BaseDelegate")

        # Load Singular Objects
        self.config_db = ConfigDb()

        # Load the Config
        self._loadConfig()

    def _loadConfig(self):
        """ Loads the default configuration from the ConfigDb """
        self.logger.debug("Entering _loadConfig")

        self.config_settings = self.config_db.getSettings("config")
        self.signal_settings = self.config_db.getSettings("signal")
        self.analysis_settings = self.config_db.getSettings("analysis")
        # Merge config settings and signal settings to create measurement
        # settings
        self.measurement_settings = dict(self.config_settings.items() +
                                        self.signal_settings.items() +
                                        self.analysis_settings.items())

        self._getAudioDevices()

    def _saveConfig(self):
        """ Saves the settings as the new default settings """
        self.logger.debug("Entering _saveConfig")

        self.config_db.saveSettings("config", self.config_settings)
        self.config_db.saveSettings("signal", self.signal_settings)
        self.config_db.saveSettings("analysis", self.analysis_settings)

    def _resetDefaults(self):
        """ Resets to the default settings """
        self.logger.debug("Entering _resetDefaults")

        self.config_db._setDefaults()

    def newMeasurement(self, measurement_settings=None):
        """ Starts a new measurement with the specified settings.

        Note, could take a non-negligible amount of time, should run in separate
        thread if used in a GUI environment.

        :param measurement_settings:
            A dictionary containing the measurements settings to use for a new
            measurement. If not given, then use default settings
        :type measurement_settings:
            dict

        :returns:
            Measurement - The measurement object created, with the captured
            measurement.
        """
        self.logger.debug("Entering newMeasurement")

        if measurement_settings is None:
            measurement_settings = self.measurement_settings

        measurement = Measurement(measurement_settings)

        measurement.startMeasurement()

        return measurement

    def saveMeasurement(self, measurement, measurement_filename):
        """ Save the measurement to the specified filename.

        Note, could take a non-negligible amount of time, should run in separate
        thread if used in a GUI environment.

        :param measurement:
            A measurement that has be completed, returned from the
            newMeasurement function.
        :type measurement:
            Measurement
        :param measurement_filename:
            The filename to save the measurement to.
        :type measurement_filename:
            str
        """
        self.logger.debug("Entering saveMeasurement")

        measurement_filename = str(measurement_filename)

        measurement_db = MeasurementDb(measurement_filename)

        measurement_attr = measurement.measurement_settings
        microphone_signals = measurement.microphone_signals
        generator_signals = measurement.generator_signals

        measurement_db.saveMeasurementAttributes(measurement_attr)

        assert(len(microphone_signals) == len(generator_signals))

        for signal_index in range(len(microphone_signals)):
            measurement_db.saveSignal(microphone_signals[signal_index],
                                 generator_signals[signal_index])

    def loadFrequencyResponse(self, measurement_filename):
        """ Loads the frequency response measurement at the specified measurement_filename.

            :param measurement_filename:
                The filename of the saved measurement.
            :type measurement_filename:
                str

            :returns:
                FrequencyResponse - An object that has analyzed the measurement
                and determined the frequency response of the device under investigation
        """
        self.logger.debug("Entering loadFrequencyResponse")

        measurement_db = MeasurementDb(measurement_filename)

        signals = measurement_db.getSignals()

        microphone_signals = signals["microphone"]
        generator_signals = signals["generator"]

        measurment_settings = measurement_db.getMeasurementSettings()

        frequency_response = FrequencyResponse(microphone_signals, generator_signals, measurment_settings)

        return frequency_response
    def loadAbsorptionCoefficient(self, measurement_filename):
        """ Loads a measurement located by the specified measurement_filename.

        Note, could take a non-negligible amount of time, should run in separate
        thread if used in a GUI environment.

        :param measurement_filename:
            The filename of the saved measurement.
        :type measurement_filename:
            str

        :returns:
            AbsorptionCoefficient - An object that has analyzed the measurement
            and determined the absorption coefficient of the material under
            measurement.
        """
        self.logger.debug("Entering loadAbsorptionCoefficient")

        measurement_db = MeasurementDb(measurement_filename)

        # Get parameters from the measurement database
        analysis_settings = measurement_db.getAnalysisSettings()

        measurement_settings = measurement_db.getMeasurementSettings()

        signals = measurement_db.getSignals()

        microphone_signals = signals["microphone"]
        generator_signals = signals["generator"]

        alpha = AbsorptionCoefficient(microphone_signals, generator_signals,
                                      measurement_settings, analysis_settings)

        return alpha

    def _getAudioDevices(self):
        """ Gets the available audio devices on the system """
        self.logger.debug("Entering _getAudioDevices")

        audio = AudioIO()
        self.audio_devices = audio.getAudioDevices()

if __name__ == "__main__":
    """ A simple example showing the use of the BaseDelegate """
    import pylab as py
    logger = logging.getLogger("Alpha")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    logger.info("Creating BaseDelegate Object")
    delegate = BaseDelegate()
    measurement_file = "/Users/lance/Programming/Python/Masters/test data/120215_asphalt.db"
    #self.logger.info("Saving measurement")
    #delegate.saveMeasurement(measurement, measurement_file)
    alpha = delegate.loadMeasurement(measurement_file)
    mic_signal = alpha.microphone_signals[0]
    t = py.arange(0, len(mic_signal) / 44100.0, 1 / 44100.0)
    py.subplot(211)

    py.plot(t, alpha.microphone_signals[0])
    py.subplot(212)
    py.plot(alpha.generator_response)
    py.show()


