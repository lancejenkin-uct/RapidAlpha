#!/usr/bin/env python
""" Provides an interface to create various graphs. """

import logging

from pylab import *
from matplotlib.ticker import MultipleLocator

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class Grapher(object):

    def __init__(self, measurement_settings):
        """ Constructor for the Grapher Object. 

        :param measurement_settings:
            The settings used for the measurement.
        :type measurement_settings:
            dict
        """
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating Grapher Object")

        self.measurement_settings = measurement_settings

    def graphImpulseResponse(self, impulse, plot_handler):
        """ Graph the impulse response of a system.  Creates a generic time
        domain plot, with the time given in milliseconds.

        :param impulse:
            The impulse response of the system.
        :type impulse:
            array
        :param plot_handler:
            The plot handler used to plot the data.
        :type plot_handler:
            matplotlib object
        """
        self.logger.debug("Entering graphImpulseResponse")

        # get parameters
        sample_rate = float(self.measurement_settings["sample rate"])

        t = arange(0, len(impulse) / sample_rate, 1 / sample_rate)

        # plot time in milliseconds
        normalized_impulse = impulse / max(impulse)
        plot_handler.axes.plot(t * 1000, normalized_impulse)
        y_minorLocator = MultipleLocator(0.1)
        x_minorLocator = MultipleLocator(1)
        plot_handler.axes.yaxis.set_minor_locator(y_minorLocator)
        plot_handler.axes.xaxis.set_minor_locator(x_minorLocator)
        plot_handler.axes.yaxis.grid(True, which='minor', color="grey", linestyle="--")
        plot_handler.axes.xaxis.grid(True, which='minor', color="grey", linestyle="--")

        plot_handler.axes.yaxis.grid(True, which='major', color="grey", linestyle="-")
        plot_handler.axes.xaxis.grid(True, which='major', color="grey", linestyle="-")
        plot_handler.axes.set_ylim(top=1.1)
        plot_handler.axes.set_xlabel("Time (ms)")
        plot_handler.axes.set_ylabel("Amplitude")
        plot_handler.draw()


    def graphFrequencyResponse(self, frequency, plot_handler):
        """ Graph the frequency response of the system,
        plots the frequency response, normalized to 0 dB, between frequencies
        20 to 10000 Hz.

        :param frequency:
            The frequency response of the system.
        :type frequency:
            array
        :param plot_handler:
            The plot handler used to plot the data.
        :type plot_handler:
            matplotlib object
        """
        self.logger.debug("Entering graphFrequencyResponse")

        # get parameters
        sample_rate = float(self.measurement_settings["sample rate"])
        fft_size = int(self.measurement_settings["fft size"])

        # Plot frequency response
        freq = fftfreq(fft_size, 1 / sample_rate)

        if len(frequency) > 0:
            normalized_frequency = 20 * log10(frequency) - max(20 * log10(frequency))
            # make the level at 1000 Hz = 0
            #bin1000 = int(1000 * fft_size / (sample_rate))
            #normalized_frequency = 20 * log10(frequency) - 20 * log10(frequency[bin1000])
            plot_handler.axes.semilogx(freq, normalized_frequency)
        else:
            data = zeros(fft_size)
            plot_handler.axes.semilogx(freq, data, lw=0)

        plot_handler.axes.set_xticks([16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125,
            160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
            4000, 5000, 6300, 8000, 1000, 12500])
        plot_handler.axes.set_xticklabels([16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125,
            160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
            4000, 5000, 6300, 8000, 1000, 12500], rotation=30)
        plot_handler.axes.set_yticks(arange(-50, 1, 5))
        plot_handler.axes.set_yticklabels(arange(-50, 1, 5))
        plot_handler.axes.set_xlim([20, 5000])
        plot_handler.axes.set_ylim([-50, 0])
        plot_handler.axes.grid(color="grey", linestyle="--")
        plot_handler.axes.set_xlabel("Frequency (Hz)")
        plot_handler.axes.set_ylabel("$L_P (dB)$")
        plot_handler.draw()


    def graphAbsorption(self, alpha, plot_handler):
        """ Graph the Absorption Coefficient of the material under measurement.

        :param alpha:
            The absorption coefficient for the material tested.
        :type alpha:
            array
        :param plot_handler:
            The plot handler to plot the data on.
        :type plot_handler:
            matplotlib object
        """
        self.logger.debug("Entering graphAbsorption")

        # Get Parameters
        fft_size = int(self.measurement_settings["fft size"])
        sample_rate = float(self.measurement_settings["sample rate"])
        decimation_factor = float(self.measurement_settings["decimation factor"])

        effective_sample_rate = sample_rate / decimation_factor



        if len(alpha) > 0:
            # Plot real data
            freq = fftfreq(len(alpha), 1 / effective_sample_rate)
            plot_handler.axes.semilogx(freq, alpha)
            plot_handler.axes.hold(True)
            data = [0.041118711,
                    0.042323201,
                    0.044548802,
                    0.062729163,
                    0.049690929,
                    0.02259814,
                    0.028213799,
                    0.122448519,
                    0.143197981,
                    0.056113836,
                    0.161168295,
                    0.190914479,
                    0.203262684,
                    0.277842701,
                    0.392483137,
                    0.474093027,
                    0.607150545,
                    0.711739335,
                    0.896910867,
                    0.818357459,
                    0.785472591,
                    0.743618322,
                    0.630230089,
                    0.539540816,
                    0.433922902,
                    0.323950617
            ]
            freq = [100,
                    112,
                    125,
                    140,
                    150,
                    200,
                    224,
                    250,
                    280,
                    315,
                    355,
                    400,
                    450,
                    500,
                    560,
                    630,
                    710,
                    800,
                    900,
                    1000,
                    1120,
                    1250,
                    1400,
                    1600,
                    1800,
                    2000
            ]
            plot_handler.axes.semilogx(freq, data, "x")
            plot_handler.axes.hold(False)
        else:
            # Set up the axis
            freq = fftfreq(fft_size, 1 / effective_sample_rate)
            data = zeros(fft_size)
            plot_handler.axes.semilogx(freq, data, lw=0)

        plot_handler.axes.set_xlim([100, 2100])
        plot_handler.axes.set_xticks([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000])
        plot_handler.axes.set_xticklabels([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000])
        plot_handler.axes.xaxis.set_minor_locator(NullLocator())
        plot_handler.axes.set_ylim([0, 1])
        plot_handler.axes.yaxis.set_major_locator(MultipleLocator(0.1))
        plot_handler.axes.grid(color="grey", linestyle="--")
        plot_handler.axes.set_xlabel("Frequency (Hz)")
        plot_handler.axes.set_ylabel(r"Absorption Coefficient")
        plot_handler.figure.subplots_adjust(bottom=0.1, top=0.98, right=0.98, left=0.05)
        plot_handler.draw()

    def graphCepstrum(self, microphone_cepstrum, generator_cepstrum, power_cepstrum, impulse_response,
                      window, window_start, plot_handler):
        """
        Graph the microphone cepstrum, generator cepstrum as well as the microphone cepstrum with the generator
        cepstrum subtracted - known as the power cepstrum in this program.  It also graphs the impulse response
        extracted from the power cepstrum.

        :param microphone_cepstrum:
            The cepstrum of the signal captured by the microphone.
        :type microphone_cepstrum:
            array of floats
        :param generator_cepstrum:
            The cepstrum of the signal from the output of the signal generator directly to the signal acquisition
            device.
        :type generator_cepstrum:
            array of floats
        :param power_cepstrum:
            The result of the microphone cepstrum with the generator's cepstrum subtracted.
        :type power_cepstrum:
            array of floats
        :param impulse_response:
            The impulse response which has been liftered from the power cepstrum.
        :type impulse_response:
            array of floats
        :param window:
            The window used to window out the impulse response of the cepstrum.
        :type window:
            array of floats
        :param window_start:
            The start of the window in seconds
        :type window_start:
            float
        :param plot_handler:
            The matplotlib plot handler
        :type plot_handler:
            matplotlib object
        """
        self.logger.debug("Entering graphCepstrum")

        # Get paramaters
        sample_rate = float(self.measurement_settings["sample rate"])
        decimation_factor = float(self.measurement_settings["decimation factor"])
        window_type = self.measurement_settings["window type"]
        effective_sample_rate = sample_rate / decimation_factor

        # Clear figure
        plot_handler.figure.clf()

        plot_handler.cepstrum_axes = plot_handler.figure.add_subplot(3,1,1)
        t = arange(0, len(microphone_cepstrum) / effective_sample_rate, 1 / effective_sample_rate)
        plot_handler.cepstrum_axes.plot(1000 * t, microphone_cepstrum, ls="-", label="Microphone Cepstrum")

        t = arange(0, len(generator_cepstrum) / effective_sample_rate, 1 / effective_sample_rate)
        plot_handler.cepstrum_axes.plot(1000 * t, generator_cepstrum, color="black", ls="--", label="Generator Cepstrum")
        plot_handler.cepstrum_axes.legend()
        plot_handler.cepstrum_axes.set_xlim(0, 15)
        plot_handler.cepstrum_axes.set_ylim(-1, 1)
        plot_handler.cepstrum_axes.grid(True)
        plot_handler.cepstrum_axes.xaxis.set_major_locator(MultipleLocator(1))
        plot_handler.cepstrum_axes.xaxis.set_minor_locator(MultipleLocator(0.1))

        plot_handler.cepstrum_axes.set_ylabel("c$[n]$")
        plot_handler.cepstrum_axes.set_title("Microphone and Generator Cepstra")

        plot_handler.power_cepstrum_axes = plot_handler.figure.add_subplot(3,1,2)

        t = arange(0, len(power_cepstrum) / effective_sample_rate, 1 / effective_sample_rate)
        plot_handler.power_cepstrum_axes.plot(1000 * t, power_cepstrum, ls="-", label="Generator Cepstrum")
        plot_handler.power_cepstrum_axes.set_xlim(0, 15)
        plot_handler.power_cepstrum_axes.set_ylim(-1, 1)
        plot_handler.power_cepstrum_axes.grid(True)

        plot_handler.power_cepstrum_axes.set_ylabel("c$[n]$")
        plot_handler.power_cepstrum_axes.set_title("Power Cepstrum")
        plot_handler.power_cepstrum_axes.xaxis.set_major_locator(MultipleLocator(1))
        plot_handler.power_cepstrum_axes.xaxis.set_minor_locator(MultipleLocator(0.1))
        # Plot the window

        t = arange(window_start,window_start + len(window) / effective_sample_rate, 1 / effective_sample_rate)

        # Fix off by one errors
        if len(window) == len(t) - 1:
            t = t[:-1]
        elif len(window) == len(t) + 1:
            t = r_[t, t[-1] + (1 / effective_sample_rate)]

        if impulse_response != []:
            plot_handler.power_cepstrum_axes.plot(1000 * t, max(impulse_response) * window, color="grey", ls="--")

        plot_handler.impulse_axes = plot_handler.figure.add_subplot(3,1,3)
        plot_handler.impulse_axes.set_xlabel("Time (ms)")
        plot_handler.impulse_axes.set_ylabel("h$[n]$")
        plot_handler.impulse_axes.set_title("Impulse Response")
        plot_handler.impulse_axes.xaxis.set_major_locator(MultipleLocator(0.5))
        plot_handler.impulse_axes.xaxis.set_minor_locator(MultipleLocator(0.1))
        t = arange(0, len(impulse_response) / effective_sample_rate, 1 / effective_sample_rate)

        # Fix off by one errors
        if len(impulse_response) == len(t) - 1:
            t = t[:-1]
        elif len(impulse_response) == len(t) + 1:
            t = r_[t, t[-1] + (1 / effective_sample_rate)]
        if len(impulse_response) > 0:
            plot_handler.impulse_axes.stem(1000 * t, impulse_response)
        else:
            plot_handler.impulse_axes.plot(1000 * t, impulse_response)
        plot_handler.impulse_axes.grid(True)
        plot_handler.figure.subplots_adjust(bottom=0.1, top=0.95, right=0.98, left=0.05, hspace=0.3)
        plot_handler.draw()

