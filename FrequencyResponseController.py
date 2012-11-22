#!/usr/bin/env python
""" Provides the controller for the FrequencyResponseView

The FrequencyResponseView provides controls for the user to select
the excitation signal to use, the type of filter to use, options
to do with the signal and the number of repeats.

It also shows the impulse response measured, along with the
corresponding frequency response.
"""

import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from FrequencyResponseView.FrequencyResponse import Ui_FrequencyResponse
from Grapher import Grapher

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class FrequencyResponseController(QMainWindow, Ui_FrequencyResponse):
    # pyqtSignals
    startMeasurement = pyqtSignal()
    updateExctration = pyqtSignal()
    saveGraph = pyqtSignal("QString")
    exportData = pyqtSignal("QString")
    loadMeasurement = pyqtSignal("QString")
    saveMeasurement = pyqtSignal("QString")

    def __init__(self, measurement_settings, audio_devices):
        """ Constructor to for the Frequency Response view.

            :param measurement_settings:
                A dictionary containing the settings to used for the
                measurement.
            :type measurement_settings:
                dict
            :param audio_devices:
                A list of all the input / output devices available in the
                system.
            :type:
                array of AudioDevice
        """

        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating FrequencyResponseController")

        QMainWindow.__init__(self)

        self.setupUi(self)

        self.measurement_settings = measurement_settings
        self.audio_devices = audio_devices

        self.grapher = Grapher(self.measurement_settings)

        self._populateWidgets()

        self._loadSettings()
        self._setupSignals()
        self._updateWidgets()

        self.showMaximized()

    def updateGraphs(self):
        """ Function called to update the graphs, called when the measurement
        is first loaded, also called when extraction options have been updated.

        """
        self.logger.debug("Entering updateGraphs")

        self.grapher.graphImpulseResponse(self.freq_response.impulse_response,
            self.impulsePlot)
        self.grapher.graphFrequencyResponse(self.freq_response.frequency_response,
            self.frequencyPlot)

    def _setupSignals(self):
        """ Connects various signals that will be emitted to the required
            slots.

        """
        self.logger.debug("Setting up signals")

        # Update measurement settings whenever a setting has updated

        # Combo Boxes
        self.inputDevices.currentIndexChanged["int"].connect(self._updateSettings)
        self.outputDevices.currentIndexChanged["int"].connect(self._updateSettings)
        self.signalType.currentIndexChanged["int"].connect(self._updateSettings)
        self.filterType.currentIndexChanged["int"].connect(self._updateSettings)

        self.signalType.currentIndexChanged["int"].connect(self._updateWidgets)
        self.filterType.currentIndexChanged["int"].connect(self._updateWidgets)
        # Signal Settings
        self.numTaps.valueChanged["int"].connect(self._updateSettings)
        self.numBursts.valueChanged["int"].connect(self._updateSettings)
        self.upperFreq.valueChanged["int"].connect(self._updateSettings)
        self.signalLength.valueChanged["int"].connect(self._updateSettings)
        self.numRepititions.valueChanged["int"].connect(self._updateSettings)

        # Filter Settings
        self.freqLPF.valueChanged["int"].connect(self._updateSettings)
        self.orderLPF.valueChanged["int"].connect(self._updateSettings)
        self.freqHPF.valueChanged["int"].connect(self._updateSettings)
        self.orderHPF.valueChanged["int"].connect(self._updateSettings)
        self.freqLow.valueChanged["int"].connect(self._updateSettings)
        self.orderLow.valueChanged["int"].connect(self._updateSettings)
        self.freqHigh.valueChanged["int"].connect(self._updateSettings)
        self.orderHigh.valueChanged["int"].connect(self._updateSettings)

        # Window Settings
        self.winLength.valueChanged["int"].connect(self._updateSettings)
        self.taperLength.valueChanged["int"].connect(self._updateSettings)

        # Emit signal when new measurement button is pressed
        self.startButton.clicked.connect(self.startMeasurement)

        load_func = self._showOpenDialog
        self.actionOpen.triggered.connect(lambda: load_func("measurement"))

        # Emit a signal when the window settings have changed
        self.winLength.valueChanged["int"].connect(self.updateExctration)
        self.taperLength.valueChanged["int"].connect(self.updateExctration)

    def _updateWidgets(self):
        """ Certain widgets have side-effects on other widgets.  This method
            will keep widgets synchronized.
        """
        self.logger.debug("Entering _updateWidgets")

        # The signal options stacked widget has two pages, one for MLS-type
        # signals, and one for swept sine signals.
        signal_type = str(self.signalType.currentText())
        if (signal_type == "Inverse Repeat Sequence" or
            signal_type == "Maximum Length Sequence"):

            self.signalOptions.setCurrentIndex(0)
        else:
            self.signalOptions.setCurrentIndex(1)

        # The filter options stacked widget has 4 pages, one for when no filters
        # are enabled, one for low pass, one for high pass and one for band pass
        filter_type = str(self.filterType.currentText())
        if filter_type == "Disabled":
            self.filterOptions.setCurrentIndex(0)
        elif filter_type == "Low Pass Filter":
            self.filterOptions.setCurrentIndex(1)
        elif filter_type == "High Pass Filter":
            self.filterOptions.setCurrentIndex(2)
        elif filter_type == "Bandpass Filter":
            self.filterOptions.setCurrentIndex(3)

    def _populateWidgets(self):
        """ Populates some widgets with default values.

            Adds excitation signals to the drop-down box
            Adds the various filters.
            Populates the input / output device.
        """
        self.logger.debug("Entering populateWidgets")

        # Populate Audio device drop-down boxes
        self.inputDevices.clear()
        self.outputDevices.clear()
        for audio_device in self.audio_devices:
            if audio_device.input_channels > 0:
                self.inputDevices.addItem(audio_device.name, audio_device.index)
            if audio_device.output_channels > 0:
                self.outputDevices.addItem(audio_device.name, audio_device.index)

        # Populate Excitation Signals
        self.signalType.clear()
        self.signalType.addItem("Inverse Repeat Sequence")
        self.signalType.addItem("Maximum Length Sequence")
        self.signalType.addItem("Low Pass Swept Sine")
        self.signalType.addItem("Swept Sine")

        # Populate Filters
        self.filterType.clear()
        self.filterType.addItem("Disabled")
        self.filterType.addItem("Low Pass Filter")
        self.filterType.addItem("High Pass Filter")
        self.filterType.addItem("Bandpass Filter")

    def _updateSettings(self):
        """ When settings change, updated the measurement settings dictionary
            associated with the dialog.
        """
        self.logger.debug("Updating measurement settings")

        # Update Audio Devices
        selected_index = self.inputDevices.currentIndex()
        input_device = self.inputDevices.itemData(selected_index).toInt()
        self.measurement_settings["input device"] = int(input_device[0])

        selected_index = self.outputDevices.currentIndex()
        output_device = self.outputDevices.itemData(selected_index).toInt()
        self.measurement_settings["output device"] = int(output_device[0])

        # Update excitation signal
        signal_type = str(self.signalType.currentText())
        self.measurement_settings["signal type"] = signal_type

        upper_frequency = self.upperFreq.value()
        self.measurement_settings["upper frequency"] = int(upper_frequency)

        signal_length = self.signalLength.value()
        # signal_length is in ms, convert to seconds
        signal_length /= 1000
        self.measurement_settings["signal length"] = signal_length

        num_taps = self.numTaps.value()
        self.measurement_settings["mls taps"] = int(num_taps)

        num_bursts = self.numBursts.value()
        self.measurement_settings["mls reps"] = int(num_bursts)

        signal_reps = self.numRepititions.value()
        self.measurement_settings["signal reps"] = int(signal_reps)

        # Update filter settings
        filter_type = self.filterType.currentText()

        if filter_type == "Bandpass Filter":
            lpf_cutoff = self.freqLow.value()
            lpf_order = self.orderLow.value()
            hpf_cutoff = self.freqHigh.value()
            hpf_order = self.orderHigh.value()

            lpf_enabled = 1
            hpf_enabled = 1

            self.measurement_settings["lpf cutoff"] = int(lpf_cutoff)
            self.measurement_settings["lpf order"] = int(lpf_order)
            self.measurement_settings["hpf cutoff"] = int(hpf_cutoff)
            self.measurement_settings["hpf order"] = int(hpf_order)

            self.measurement_settings["lpf enabled"] = lpf_enabled
            self.measurement_settings["hpf enabled"] = hpf_enabled
        elif filter_type == "Low Pass Filter":
            lpf_cutoff = self.freqLPF.value()
            lpf_order = self.orderLPF.value()

            lpf_enabled = 1
            hpf_enabled = 0

            self.measurement_settings["lpf cutoff"] = int(lpf_cutoff)
            self.measurement_settings["lpf order"] = int(lpf_order)

            self.measurement_settings["lpf enabled"] = lpf_enabled
            self.measurement_settings["hpf enabled"] = hpf_enabled
        elif filter_type == "High Pass Filter":
            hpf_cutoff = self.freqHPF.value()
            hpf_order = self.orderHPF.value()

            lpf_enabled = 0
            hpf_enabled = 1

            self.measurement_settings["hpf cutoff"] = int(hpf_cutoff)
            self.measurement_settings["hpf order"] = int(hpf_order)

            self.measurement_settings["lpf enabled"] = lpf_enabled
            self.measurement_settings["hpf enabled"] = hpf_enabled
        elif filter_type == "Disabled":
            lpf_enabled = 0
            hpf_enabled = 0

            self.measurement_settings["lpf enabled"] = lpf_enabled
            self.measurement_settings["hpf enabled"] = hpf_enabled

        # Update Window options
        window_length = self.winLength.value()
        self.measurement_settings["window length"] = int(window_length)

        taper_length = self.taperLength.value()
        self.measurement_settings["taper length"] = int(taper_length)

    def _loadSettings(self):
        """ Loads the default settings for the frequency response measurements.

        """
        self.logger.debug("Loading default settings")

        # Set the audio devices
        default_input_device = int(self.measurement_settings["input device"])
        input_index = self.inputDevices.findData(default_input_device)
        self.inputDevices.setCurrentIndex(input_index)

        default_output_device = int(self.measurement_settings["output device"])
        output_index = self.outputDevices.findData(default_output_device)
        self.outputDevices.setCurrentIndex(output_index)

        # Set Excitation Signal
        default_signal = str(self.measurement_settings["signal type"])
        signal_index = self.signalType.findText(default_signal)
        self.signalType.setCurrentIndex(signal_index)

        # Set swept sine signal settings
        upper_frequency = int(self.measurement_settings["upper frequency"])
        self.upperFreq.setValue(upper_frequency)

        # Signal length is in seconds, convert to ms
        signal_length = float(self.measurement_settings["signal length"])
        signal_length *= 1000
        self.signalLength.setValue(int(signal_length))

        # Set MLS / IRS settings
        num_taps = int(self.measurement_settings["mls taps"])
        print self.measurement_settings
        self.numTaps.setValue(num_taps)

        num_bursts = int(self.measurement_settings["mls reps"])
        self.numBursts.setValue(num_bursts)

        signal_reps = int(self.measurement_settings["signal reps"])
        self.numRepititions.setValue(signal_reps)

        # Set filter options
        lpf_cutoff = int(self.measurement_settings["lpf cutoff"])
        lpf_order = int(self.measurement_settings["lpf order"])
        hpf_cutoff = int(self.measurement_settings["hpf cutoff"])
        hpf_order = int(self.measurement_settings["hpf order"])

        # Band Pass Filter
        self.freqLow.setValue(lpf_cutoff)
        self.orderLow.setValue(lpf_order)
        self.freqHigh.setValue(hpf_cutoff)
        self.orderHigh.setValue(hpf_order)

        # Low Pass Filter
        self.freqLPF.setValue(lpf_cutoff)
        self.orderLPF.setValue(lpf_order)

        # High Pass Filter
        self.freqHPF.setValue(hpf_cutoff)
        self.orderHPF.setValue(hpf_order)

        # Set the correct drop-down options
        lpf_enabled = int(self.measurement_settings["lpf enabled"])
        hpf_enabled = int(self.measurement_settings["hpf enabled"])

        if lpf_enabled == 1 and hpf_enabled == 1:
            # Band pass filter
            bpf_index = self.filterType.findText("Bandpass Filter")
            self.filterType.setCurrentIndex(bpf_index)
        elif lpf_enabled == 1:
            # Low pass filter
            lpf_index = self.filterType.findText("Low Pass Filter")
            self.filterType.setCurrentIndex(lpf_index)
        elif hpf_enabled == 1:
            # High pass filter
            hpf_index = self.filterType.findText("High Pass Filter")
            self.filterType.setCurrentIndex(hpf_index)
        else:
            # Disabled
            index = self.filterType.findText("Disabled")
            self.filterType.setCurrentIndex(index)

        # Set window options
        window_length = int(self.measurement_settings["window length"])
        taper_length = int(self.measurement_settings["taper length"])

        self.winLength.setValue(window_length)
        self.taperLength.setValue(taper_length)

    def _showSaveDialog(self, file_type):
        """ Shows the save dialog to get the filename to save the required
            data to.

        :param file_type:
            The type of data to be saved, could be one of "graph", "csv",
            "measurement"
        :type file_type:
            str
        """
        self.logger.debug("Entering _showSaveDialog")

        if file_type == "graph":
            caption = "Select file to save the graph"
            filter = "PNG (*.png)"
            signal = self.saveGraph
        elif file_type == "csv":
            caption = "Select file to export data to"
            filter = "CSV (*.csv)"
            signal = self.exportData
        elif file_type == "measurement":
            caption = "Select file to save the measurement to"
            filter = "FrequencyResponse (*.fdb)"
            signal = self.saveMeasurement
        else:
            self.logger.debug("Invalid file_type passed: %s" % (file_type))
            return

        dir = "./"

        filename = QFileDialog.getSaveFileName(self, caption, dir, filter)

        if filename != "":
            signal.emit(filename)

    def _showOpenDialog(self, file_type):
        """ Shows the open dialog to get the filename to load the required data.

        :param file_type:
            The type of data to be saved, could be one of "graph", "csv",
            "measurement"
        :type file_type:
            str
        """
        self.logger.debug("Entering _showOpenDialog")

        if file_type == "measurement":
            caption = "Select Measurement File to Load"
            filter = "FrequencyResponse (*.fdb)"
            signal = self.loadMeasurement
        else:
            self.logger.debug("Invalid file_type passed: %s" % (file_type))
            return

        dir = "./"

        filename = QFileDialog.getOpenFileName(self, caption, dir, filter)
        # filename is a tuple (filename, selected filter) when file is selected
        # else a blank string if dialog closed
        print filename
        if filename != "":
            signal.emit(filename)
