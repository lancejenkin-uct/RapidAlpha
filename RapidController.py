#!/usr/bin/env python
""" Provides the controller for the RapidView

The RapidAlpha view provides a means to preform a quick absorption measurement
of a material.  The only settings that can be changed in the rapid view, is the
selecting of input and output devices.
"""

import logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from math import log10
from RapidView.RapidAlphaWindow import Ui_RapidAlphaWindow
from Grapher import Grapher

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class RapidController(QMainWindow, Ui_RapidAlphaWindow):
    # pyqtSignals
    startMeasurement = pyqtSignal()
    saveGraph = pyqtSignal("QString")
    exportData = pyqtSignal("QString")
    loadMeasurement = pyqtSignal("QString")
    saveMeasurement = pyqtSignal("QString")
    showPreferences = pyqtSignal()
    savePreferences = pyqtSignal("QString")
    loadPreferences = pyqtSignal("QString")
    exit = pyqtSignal()

    def __init__(self, measurement_settings, audio_devices):
        """ Constructor for RapidController, sets up the view, signals and shows
            the window.

            :param measurement_settings:
                A dictionary containing the settings to used for the measurement.
            :type measurement_settings:
                dict
            :param audio_devices:
                A list of all the input / output devices available in the
                system.
            :type:
                array of AudioDevice
        """
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating RapidController")

        QMainWindow.__init__(self)

        self.measurement_settings = measurement_settings
        self.audio_devices = audio_devices
        self.grapher = Grapher(self.measurement_settings)
        self.alpha = None



        self.setupUi(self)
        self._setupWidgets()
        self._setupSignals()

        self.showMaximized()

    def update(self):
        """ Updates the graph showing the absorption coefficient of the material
        measured.
        """
        self.logger.debug("Entering update")

        self.grapher.graphAbsorption(self.alpha.alpha, self.AlphaPlot)
        self.grapher.graphCepstrum(self.alpha.microphone_cepstrum,
            self.alpha.generator_cepstrum, self.alpha.power_cepstrum, self.alpha.impulse_response,
            self.alpha.window, float(self.alpha.measurement_settings["window start"]), self.CepstrumPlot)

    def _setupWidgets(self):
        """ Setup the widgets to show the user.

            The graph is formatted with no data.
        """
        self.logger.debug("Entering _setupWidgets")

        self.grapher.graphAbsorption([], self.AlphaPlot)
        self.grapher.graphCepstrum([], [], [], [], [], 0, self.CepstrumPlot)

        # Add Volume slider to toolbar
        self.gainSlider = QSlider(Qt.Horizontal)
        self.gainSlider.setMaximumWidth(100)
        self.gainSlider.setMaximum(0)
        self.gainSlider.setMinimum(-1000)

        self.gainSpin = QDoubleSpinBox()
        self.gainSpin.setMaximum(0)
        self.gainSpin.setMinimum(-10)
        self.gainSpin.setSingleStep(0.01)

        self.toolBar.addSeparator()
        self.toolBar.addWidget(QSpacerItem(0,0).widget())
        self.toolBar.addWidget(QLabel("Gain: "))
        self.toolBar.addWidget(self.gainSlider)
        self.toolBar.addWidget(self.gainSpin)
        self.toolBar.addWidget(QLabel(" dB"))

        self.updateWidgets()

    def updateWidgets(self):
        """ Set the values for widgets on the screen.

        """

        gain = 20 * log10(float(self.measurement_settings["gain"]) )
        self.gainSlider.setValue(gain)
        self.gainSpin.setValue(gain)

    def _updateMeasurementSettings(self):
        """ Update the Measurement Settings dictionary.

        For the Rapid View, the only settings that change are the input and
        output devices.
        """
        self.logger.debug("Entering _updateMeasurementSettings")

        selected_index = self.InputDeviceList.currentIndex()
        input_device = self.InputDeviceList.itemData(selected_index).toInt()
        self.measurement_settings["input device"] = input_device[0]

        selected_index = self.OutputDeviceList.currentIndex()
        output_device = self.OutputDeviceList.itemData(selected_index).toInt()
        self.measurement_settings["output device"] = output_device[0]

    def _setupSignals(self):
        """ Connects the various button signals to the class signals. """
        self.logger.debug("Entering _setupSignals")

        save_func = self._showSaveDialog
        self.actionSave.triggered.connect(lambda: save_func("measurement"))
        self.actionExport_Data.triggered.connect(lambda: save_func("csv"))
        self.actionExport_Graph.triggered.connect(lambda: save_func("graph"))
        self.actionSave_Preferences.triggered.connect(lambda: save_func("preferences"))

        load_func = self._showOpenDialog
        self.actionSave_Preferences.triggered.connect(lambda: load_func("preferences"))
        self.actionLoad_Measurement.triggered.connect(lambda: load_func("measurement"))
        self.actionExit.triggered.connect(self.exit)

        self.actionStart_Measurement.triggered.connect(self.startMeasurement)

        self.actionPreferences.triggered.connect(self.showPreferences)

        self.gainSlider.valueChanged.connect(self._updateWidgets)
        self.gainSpin.valueChanged.connect(self._updateWidgets)

    def _updateWidgets(self):
        """ Keeps widgets synchronized with the measurement settings, and each other
        """
        self.logger.debug("Entering _updateWidgets")

        # Keep the gain in sync
        gain = float(self.measurement_settings["gain"])
        gain_db = 20 * log10(gain)

        self.logger.debug("sender: %s" %(self.sender()))
        if self.sender() == self.gainSlider:
            self.logger.debug("Slider: %s" % (self.gainSlider.value()))
            self.gainSpin.setValue((self.gainSlider.value() / 100.0))
        elif self.sender() == self.gainSpin:
            self.gainSlider.setValue(self.gainSpin.value() * 100.0)

        gain = 10 ** (self.gainSpin.value() / 20.0)

        self.measurement_settings["gain"] = gain

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
            filter = "AlphaDb (*.db)"
            signal = self.loadMeasurement
        elif file_type == "preferences":
            caption = "Select Preferences File to Load"
            filter = "Preferences (*.db)"
            signal = self.loadPreferences
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
            supported_file_types = self.AlphaPlot.figure.canvas.get_supported_filetypes_grouped()
            # Get available output formats
            filter = []
            for key, value in supported_file_types.items():
                filter.append("%s (*.%s)" % (key, " *.".join(value)))
            filter = ";;".join(filter)
            signal = self.saveGraph
        elif file_type == "csv":
            caption = "Select file to export data to"
            filter = "CSV (*.csv)"
            signal = self.exportData
        elif file_type == "measurement":
            caption = "Select file to save the measurement to"
            filter = "AlphaDb (*.db)"
            signal = self.saveMeasurement
        elif file_type == "preferences":
            caption = "Select Filename to save Preferences"
            filter = "Preferences (*.db)"
            signal = self.savePreferences
        else:
            self.logger.debug("Invalid file_type passed: %s" % (file_type))
            return

        dir = "./"

        filename = QFileDialog.getSaveFileName(self, caption, dir, filter)

        if filename != "":
            signal.emit(filename)
