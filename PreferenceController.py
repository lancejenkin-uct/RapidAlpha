#!/usr/bin/env python
""" Provides the controller for the PreferenceView

The PrefernceView provides a means for setting the configuration
for the signals.  Used for both the Frequency Response and the
Absorption Coefficient Delegate.

"""

import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from math import log10

from PreferenceView.Preferences import Ui_Preference
from AudioIO import AudioIO

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class PreferenceController(QDialog, Ui_Preference):
    # pyqtSignals
    saveSettings = pyqtSignal()

    def __init__(self, settings):
        """ Constructor for Preference Controller

        """
        QDialog.__init__(self)
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating PreferenceController")

        self.settings = settings
        self.audio_io = AudioIO()

        self.setupUi(self)

        self._setupSignals()
        self._setupWidgets()

    def _setupWidgets(self):
        """ Using the settings passed to the constructor, set the
        values of the widgets.

        """
        self.logger.debug("Entering _setupWidgets")

        # Audio Devices Tab
        ## Populate the combo boxes
        self._updateAudioDevices()

        ## Set the current input and output devices
        if "input device" in self.settings:
            input_device = int(self.settings["input device"])
            input_index = self.inputDeviceBox.findData(input_device)

            self.inputDeviceBox.setCurrentIndex(input_index)

        if "output device" in self.settings:
            output_device = int(self.settings["output device"])
            output_index = self.outputDeviceBox.findData(output_device)

            self.outputDeviceBox.setCurrentIndex(output_index)

        ## Set the gain
        self._setValue("gain", self.gainSlider, lambda x: 20 * log10(float(x)))

        ## Set the buffer size
        self._setValue("buffer size", self.bufferSizeBox, lambda x: int(x))

        # Measurement Settings Tab
        ## Excitation Signal Settings
        if "signal type" in self.settings:
            signal_type = self.settings["signal type"]
            signal_index = self.signalBox.findText(signal_type,
                ~Qt.MatchCaseSensitive | Qt.MatchExactly)

            self.signalBox.setCurrentIndex(signal_index)

        self._setValue("signal reps", self.signalRepetitionsBox,
            lambda x: int(x))

        ### MLS Signal Settings
        self._setValue("mls taps", self.mlsTapsBox, lambda x: int(x))
        self._setValue("mls reps", self.mlsRepetitionsBox, lambda x: int(x))

        ### Swept Sine Settings
        self._setValue("lower frequency", self.lowerFrequencyBox,
            lambda x: int(x))
        self._setValue("upper frequency", self.upperFrequencyBox,
            lambda x: int(x))
        self._setValue("signal length", self.signalLengthBox,
            lambda x: float(x) * 1000.0)

        ## Filter Settings
        self._setValue("hpf cutoff", self.highPassCutOffBox, lambda x: int(x))
        self._setValue("hpf order", self.highPassOrderBox, lambda x: int(x))
        self._setChecked("hpf enabled", self.highPassEnabledBox)

        self._setValue("lpf cutoff", self.lowPassCutOffBox, lambda x: int(x))
        self._setValue("lpf order", self.lowPassOrderBox, lambda x: int(x))
        self._setChecked("lpf enabled", self.lowPassEnabledBox)

        ## Extraction Settings
        self._setValue("window start", self.windowStartBox, lambda x: float(x))
        self._setValue("window end", self.windowEndBox, lambda x: float(x))
        self._setValue("taper length", self.taperLengthBox, lambda x: float(x))
        self._setValue("decimation factor", self.decimationBox,
            lambda x: int(x))

    def _setChecked(self, key, widget):
        """ Set the check status of the widget.

        @param key: The key in the settings dictionary to determined the check
            status of the widget.
        @param widget: The checkable widget to set the value.

        """
        self.logger.debug("Entering _setChecked")

        if key in self.settings:
            value = bool(self.settings[key])

            widget.setChecked(value)

    def _setValue(self, key, widget, modifier=None):
        """ Set the value for the specified widget.

        @param key: The key in the settings dictionary to set the widget.
        @param widget: The widget to set the value
        @param modifier: Optional modifier function to preform on the value

        """
        self.logger.debug("Entering _setBoxValue")

        if key in self.settings:
            value = self.settings[key]
            if modifier is not None:
                value = modifier(value)
            widget.setValue(value)

    def _updateAudioDevices(self):
        """ Get the available audio devices, and populate the combo boxes """
        self.logger.debug("Entering _updateAudioDevices")

        audio_devices = self.audio_io.getAudioDevices()

        for device in audio_devices:
            if device.input_channels > 0:
                self.inputDeviceBox.addItem(device.name, device.index)
            if device.output_channels > 0:
                self.outputDeviceBox.addItem(device.name, device.index)

    def _updateSettings(self):
        """ Update the settings with the widget values

        """
        self.logger.debug("Entering _updateSettings")

        # Audio Settings
        input_index = self.inputDeviceBox.currentIndex()
        input_device = int(self.inputDeviceBox.itemData(input_index).toInt()[0])
        self.settings["input device"] = input_device

        output_index = self.outputDeviceBox.currentIndex()
        output_device = int(self.outputDeviceBox.itemData(output_index).toInt()[0])
        self.settings["output device"] = output_device

        gain = 10 ** ((self.gainSlider.value() / 100.0) / 20.0)
        self.settings["gain"] = gain

        buffer_size = self.bufferSizeBox.value()
        self.settings["buffer size"] = buffer_size

        # Measurement Settings
        ## Excitation Signal
        signal_type = str(self.signalBox.currentText())
        self.settings["signal type"] = signal_type

        signal_reps = self.signalRepetitionsBox.value()
        self.settings["signal reps"] = signal_reps

        ### Swept Sine Settings
        lower_frequency = self.lowerFrequencyBox.value()
        self.settings["lower frequency"] = lower_frequency

        upper_frequency = self.upperFrequencyBox.value()
        self.settings["upper frequency"] = upper_frequency

        signal_length = self.signalLengthBox.value()
        signal_length /= 1000.0  # convert to seconds
        self.settings["signal length"] = signal_length

        ### MLS Signal Settings
        mls_taps = self.mlsTapsBox.value()
        self.settings["mls taps"] = mls_taps

        mls_reps = self.mlsRepetitionsBox.value()
        self.settings["mls reps"] = mls_reps

        ## Filter Settings
        hpf_cutoff = self.highPassCutOffBox.value()
        self.settings["hpf cutoff"] = hpf_cutoff

        hpf_order = self.highPassOrderBox.value()
        self.settings["hpf order"] = hpf_order

        hpf_enabled = self.highPassEnabledBox.isChecked()
        self.settings["hpf enabled"] = int(hpf_enabled)

        lpf_cutoff = self.lowPassCutOffBox.value()
        self.settings["lpf cutoff"] = lpf_cutoff

        lpf_order = self.lowPassOrderBox.value()
        self.settings["lpf order"] = lpf_order

        lpf_enabled = self.lowPassEnabledBox.isChecked()
        self.settings["lpf enabled"] = int(lpf_enabled)

        ## Extraction Settings
        window_start = self.windowStartBox.value()
        self.settings["window start"] = window_start

        window_end = self.windowEndBox.value()
        self.settings["window end"] = window_end

        window_type = str(self.windowTypeBox.currentText())
        self.settings["window type"] = window_type.lower()

        taper_length = self.taperLengthBox.value()
        self.settings["taper length"] = taper_length

        decimation_factor = self.decimationBox.value()
        self.settings["decimation factor"] = decimation_factor

    def _synchronizeWidgets(self):
        """ Synchronize widgets so that the correct widgets are
        displayed.

        """
        self.logger.debug("Entering _synchronizeWidgets")



        # Synchronize measurementStacked
        index = self.measurementSettingsList.currentRow()
        self.measurementStacked.setCurrentIndex(index)

        # Synchronize signalStacked
        index = self.signalBox.currentIndex()
        if (index == 0 or index == 1):
            # Swept Sine based signal
            self.signalStacked.setCurrentIndex(0)
        else:
            # MLS based signal
            self.signalStacked.setCurrentIndex(1)

        # Synchronise gain
        self.gainLabel.setText("%.2f dB" % (self.gainSlider.value() / 100.0))

        # Synchronize MLS length label
        if "sample rate" in self.settings:
            mls_taps = self.mlsTapsBox.value()
            sample_rate = float(self.settings["sample rate"])

            mls_length = 1000 * (2 ** mls_taps) / sample_rate
            self.signalLengthLabel.setText("Signal Length: %f ms" % (mls_length))

    def _setupSignals(self):
        """ Sets up the signals of the widgets, connecting them
        to the required slots.

        """
        self.logger.debug("Entering _setupSignals")

        self.buttonBox.accepted.connect(self.saveSettings)

        # Synchronize Widgets
        self.gainSlider.valueChanged.connect(self._synchronizeWidgets)
        self.preferenceTab.currentChanged.connect(self._synchronizeWidgets)
        self.measurementSettingsList.itemClicked.connect(
            self._synchronizeWidgets)
        self.signalBox.currentIndexChanged.connect(self._synchronizeWidgets)
