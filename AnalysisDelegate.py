#!/usr/bin/env python
""" Provides the controller for the Analysis Delegate

The Analysis Delegate is used for creating statistical analysis of using the
cepstral technique to measure the absorption coefficient of materials.

All figures used in the write up are generated in the Analysis Delegate
"""

import logging
from pylab import *
from scipy.signal import butter, lfilter
from scipy.fftpack import rfftfreq
from SignalGenerator import SignalGenerator
import tempfile


from AudioIO import AudioIO
from BaseDelegate import BaseDelegate
from Grapher import Grapher
from MlsDb import MlsDb

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class AnalysisDelegate(BaseDelegate):

    def __init__(self):
        BaseDelegate.__init__(self)
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating AnalysisDelegate")

    def responseAnalysis(self):
        """ Method to analyse responses """
        self.logger.debug("Entering responseAnalysis")

        measurement_file = "../test data/120301_asphalt.db"

        alpha = self.loadAbsorptionCoefficient(measurement_file)

        gen_signal = alpha.generator_signals[0]
        mic_signal = alpha.microphone_signals[0]

        plot(alpha.generator_cepstrum)

        show()

    def loudspeakerFrequencyResponse(self):
        """ Creates graphs for the frequency response curves
        for the Phillips AD3714 loudspeaker driver.
        """
        self.logger.debug("Entering loudspeakerFrequencyResponse")

        

        def plot_frequency_response(frequency_response, name):
                # Plot frequency response

                fig = figure(figsize=(2*3.375, 3.375))
                handler = Object()
                handler.axes = fig.add_subplot(111)
                handler.draw = draw
                grapher.graphFrequencyResponse(frequency_response, handler)
                handler.axes.set_xticklabels(["", "", "", 31.5, "", "", 63, "", "", 125,
                    "", "", 250, "", "", 500, "", "", "1K", "", "", "2K", "", "",
                    "4K", "", "", "8K", "", ""])
                fig.subplots_adjust(bottom=0.15, top=0.98, right=0.98, left=0.1)
                savefig("Analysis/Images/%s_frequency_response.eps" % (name,))

        def plot_impulse_response(impulse_response, name):
                # Plot the impulse response
                fig = figure(figsize=(2 * 3.375, 3.375))
                handler = Object()
                handler.axes = fig.add_subplot(111)
                handler.draw = draw
                grapher.graphImpulseResponse(impulse_response, handler)
                fig.subplots_adjust(bottom=0.15, top=0.98, right=0.98, left=0.1)
                savefig("Analysis/Images/%s_impulse_response.eps" % (name))
        name = "120602_loudspeaker_3"

        measurement_filename = "../testdata/%s.fdb" % (name)
        freq = self.loadFrequencyResponse(measurement_filename)
        grapher = Grapher(freq.measurement_settings)
        plot_frequency_response(freq.frequency_response, name)
        plot_impulse_response(freq.impulse_response, name)

        return

        # Base case, no modifications
        for i in range(1, 13):
                name = "120514_loudspeaker_%s" % (i)
                measurement_filename = "../testdata/%s.fdb" % (name)
                freq = self.loadFrequencyResponse(measurement_filename)
                grapher = Grapher(freq.measurement_settings)
                plot_frequency_response(freq.frequency_response, name)
                plot_impulse_response(freq.impulse_response, name)


    def analysisImpulseResponse(self):
        """ Record the impulse response from the loudspeaker. """
        self.measurement_settings["output device"] = 2
        self.measurement_settings["input device"] = 3

        measurement_filename = "../test data/120309_impulse_test.db"

        measurement = self.newMeasurement(self.measurement_settings)
        self.saveMeasurement(measurement, measurement_filename)
        alpha = self.loadAbsorptionCoefficient(measurement_filename)

        plot(left)
        show()

    def filterTest(self):
        """ Test various filter cut off and orders and the effects on the absorption
        coefficient. """
        self.logger.debug("Entering filterTest")

        lpf_order_tests = [1, 3, 4, 5]
        lpf_cutoff_tests = [3000, 3500, 4000, 4500]

        self.measurement_settings["signal reps"] = 3
        self.measurement_settings["output device"] = 2
        self.measurement_settings["input device"] = 3
        for lpf_order in lpf_order_tests:
                for lpf_cutoff in lpf_cutoff_tests:
                        print "Testing LPF Order %s with cut off of %s" % (lpf_order, lpf_cutoff)
                        self.measurement_settings["lpf order"] = lpf_order
                        self.measurement_settings["lpf cutoff"] = lpf_cutoff

                        measurement = self.newMeasurement(self.measurement_settings)
                        measurement_filename = "../test data/120305_lpf_%s_%s.db" % (lpf_cutoff, lpf_order)
                        self.saveMeasurement(measurement, measurement_filename)
    
    def hpFilterTest(self):
        """ Test various high pass filter cut off and orders and the effects on the absorption
        coefficient. """
        self.logger.debug("Entering hpFilterTest")

        hpf_order_tests = [1, 3, 4, 5]
        hpf_cutoff_tests = [100, 150, 200, 500, 1000]

        self.measurement_settings["signal reps"] = 3
        self.measurement_settings["output device"] = 2
        self.measurement_settings["input device"] = 3
        for hpf_order in hpf_order_tests:
                for hpf_cutoff in hpf_cutoff_tests:
                        print "Testing HPF Order %s with cut off of %s" % (hpf_order, hpf_cutoff)
                        self.measurement_settings["hpf order"] = hpf_order
                        self.measurement_settings["hpf cutoff"] = hpf_cutoff
                        self.measurement_settings["hpf enable"] = 1

                        measurement = self.newMeasurement(self.measurement_settings)
                        measurement_filename = "../test data/120323_hpf_%s_%s.db" % (hpf_order, hpf_cutoff)
                        self.saveMeasurement(measurement, measurement_filename)
    def showFilterTests(self):
        """" plot the absorption coefficient of the various tests done with
        different filter orders """
        self.logger.debug("Entering showFilterTests")

        hpf_order_tests = [1, 3, 4, 5]
        hpf_cutoff_tests = [100, 150, 200, 500, 1000]

        fig = figure()
        for order_index, hpf_order in enumerate(hpf_order_tests):
                for cutoff_index, hpf_cutoff in enumerate(hpf_cutoff_tests):
                        measurement_filename = "../test data/120323_hpf_%s_%s.db" % (hpf_order, hpf_cutoff)
                        alpha = self.loadAbsorptionCoefficient(measurement_filename)
                        grapher = Grapher(alpha.measurement_settings)
                        
                        handler = Object()
                        handler.axes = fig.add_subplot(len(hpf_order_tests), len(hpf_cutoff_tests), order_index * len(hpf_cutoff_tests) + cutoff_index)
                        handler.draw = draw
                        grapher.graphAbsorption(alpha.alpha, handler)
                        plot(alpha.alpha)
                        title("%s Hz order %s" % (hpf_cutoff, hpf_order))

        show()


    def misidentificationAnalysis(self):
        """ Preform analysis on misidentifying the synchronization
        impulse """
        self.logger.debug("Entering misidentificationAnalysis")

        measurement_file = "/Users/lance/Programming/Python/Masters/testdata/2012/08/120806_reflective_63.db"

        alpha = self.loadAbsorptionCoefficient(measurement_file)
        print alpha.measurement_settings
        grapher = Grapher(alpha.measurement_settings)

        fig = figure(figsize=(7, 5))

        mic_impulse_loc = int(alpha.measurement_settings["microphone impulse location"])
        gen_impulse_loc = int(alpha.measurement_settings["generator impulse location"])

        (gen_start, gen_end) = (gen_impulse_loc - 20, gen_impulse_loc + 20)
        (mic_start, mic_end) = (mic_impulse_loc - 20, mic_impulse_loc + 100)
        gen_signal = alpha.generator_signals[0][gen_start:gen_end]
        mic_signal = alpha.microphone_signals[0][mic_start:mic_end]
        ax = fig.add_subplot(211)
        ax.plot(gen_signal)
        ax.axvline(x=gen_impulse_loc - gen_start, color="black", linestyle="--", lw=1)
        ax = fig.add_subplot(212)
        ax.plot(mic_signal)
        ax.axvline(x=mic_impulse_loc - mic_start, color="black", linestyle="--", lw=1)
        show()

        fig = figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        resp = abs(fft(alpha.microphone_signals[0])) ** 2
        t = arange(0, len(alpha.generator_cepstrum) / 44100.0, 1 / 44100.0)
        ax.plot(t, alpha.generator_cepstrum)
        ax.plot(t, alpha.microphone_cepstrum)
        ax.plot(t, alpha.power_cepstrum)
        show()
        fig = figure(figsize=(7, 5))
        handler = Object()
        handler.figure = fig
        handler.axes = fig.add_subplot(111)
        handler.draw = draw
        grapher.graphAbsorption(alpha.alpha, handler)
        show()

        alpha.measurement_settings["microphone impulse location"] = mic_impulse_loc
        alpha.measurement_settings["generator impulse location"] = gen_impulse_loc + 3

        alpha.determineAlpha()
        fig = figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.plot(alpha.generator_cepstrum)
        ax.plot(alpha.microphone_cepstrum)
        show()
        fig = figure(figsize=(7, 5))
        handler = Object()
        handler.figure = fig
        handler.axes = fig.add_subplot(111)
        handler.draw = draw
        grapher.graphAbsorption(alpha.alpha, handler)
        show()
        

    def synchronizeAnalysis(self):
        """ Method to analyse the synchronization of recieved responses. """
        self.logger.debug("Entering synchronizeAnalysis")

        measurement_file = "../test data/120215_asphalt.db"

        alpha = self.loadAbsorptionCoefficient(measurement_file)

        sample_rate = 44100.0

        gen_signal = alpha.generator_signals[0]
        mic_signal = alpha.microphone_signals[0]

        # Show Generator Impulse
        generator_impulse = gen_signal[19375:19450]

        fig = figure()
        ax = fig.add_subplot(111)
        ax.axhline(y=0, linestyle="-", color="black", linewidth=1)
        ax.plot(generator_impulse)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.text(26, 0.22, "pre-ringing", ha="center", va="bottom", size=10)
        ax.annotate("", xy=(19, 0), xycoords="data", xytext=(26, 0.2),
                    arrowprops=dict(arrowstyle="->"))
        ax.annotate("", xy=(33, 0.08), xycoords="data", xytext=(26, 0.2),
                    arrowprops=dict(arrowstyle="->"))

        peak_y = max(generator_impulse)
        peak_x = where(generator_impulse == peak_y)[0][0]

        ax.annotate("(%d, %.2f)" % (peak_x, peak_y), xy=(peak_x, peak_y),
                    xycoords="data", xytext=(peak_x + 2, peak_y + 0.1),
                    arrowprops=dict(arrowstyle="->"))
        line = Line2D([19, 19], [0, 0.2], color="black", linestyle="--", lw=1)
        ax.add_line(line)
        line = Line2D([33, 33], [0, 0.2], color="black", linestyle="--", lw=1)
        ax.add_line(line)

        ax.set_xlim([0, 70])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        savefig("Analysis/Images/generator_impulse_with_preringing.eps")
        # Show Generator Impulse with a Phase Shift
        cla()
        generator_impulse = hilbert(generator_impulse)
        ax = fig.add_subplot(111)
        ax.axhline(y=0, linestyle="-", color="black", linewidth=1)
        ax.plot(generator_impulse)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")

        peak_y = max(generator_impulse)
        peak_x = where(generator_impulse == peak_y)[0][0]

        ax.annotate("(%d, %.2f)" % (peak_x, peak_y), xy=(peak_x, peak_y),
                    xycoords="data", xytext=(peak_x + 2, peak_y + 0.1),
                    arrowprops=dict(arrowstyle="->"))
        ax.set_xlim([0, 70])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        savefig("Analysis/Images/generator_impulse_phase_shifted.eps")

        # Show the Microphone Impulse Response
        mic_impulse = mic_signal[19470:20427]

        cla()
        ax.axhline(y=0, linestyle="-", color="black", linewidth=1)
        ax = fig.add_subplot(111)
        ax.plot(mic_impulse)

        ax.text(50, 0.01, "onset", ha="center", size=10)
        ax.annotate("", xy=(73, 0),
                    xycoords="data", xytext=(50, 0.01),
                    arrowprops=dict(arrowstyle="->"))
        ax.text(70, -0.019, "19 samples", ha="right", va="bottom", size=10)
        ax.annotate("", xy=(92, -0.02), xycoords="data", xytext=(115, -0.02),
                    arrowprops=dict(arrowstyle="->"))
        ax.annotate("", xy=(73, -0.02), xycoords="data", xytext=(50, -0.02),
                    arrowprops=dict(arrowstyle="->"))

        line = Line2D([73, 73], [0, -0.035], color="black", linestyle="--",
                        lw=1)
        ax.add_line(line)
        line = Line2D([92, 92], [0, -0.035], color="black", linestyle="--",
                        lw=1)
        ax.add_line(line)

        ax.set_xlim([0, 300])
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.yaxis.set_ticks(arange(-0.04, 0.04, 0.02))
        savefig("Analysis/Images/microphone_impulse.eps")

        # Plot the Difference Microphone Response
        mic_impulse = abs((mic_impulse[1:] - mic_impulse[:-1]))
        d_mic = mic_signal[1:] - mic_signal[:-1]
        mic_noise = d_mic[abs(d_mic) > 0][:1000]
        max_noise = max(abs(mic_noise))
        std_noise = std(abs(mic_noise))

        mic_threshold = max_noise + 2.5 * std_noise
        onset = where(mic_impulse > mic_threshold)[0][0] - 1
        print onset
        cla()
        ax.axhline(y=0, linestyle="-", color="black", linewidth=1)
        ax.axhline(y=mic_threshold, linestyle="--", color="black", lw=1)
        ax.axvline(x=onset, linestyle="-.", color="grey", lw=1)
        ax = fig.add_subplot(111)
        ax.plot(mic_impulse)
        ax.set_xlim([0, 300])
        ax.text(30, 0.001, "onset at 73", ha="center", size=10)
        ax.annotate("", xy=(73, mic_impulse[onset]),
                    xycoords="data", xytext=(30, 0.001),
                    arrowprops=dict(arrowstyle="->"))
        ax.text(30, mic_threshold + 0.0001, "threshold", ha="center", size=10)
        xlabel("Samples")
        ylabel("Amplitude")
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.yaxis.set_ticks(arange(0, 0.008, 0.002))
        savefig("Analysis/Images/onset_detection.eps")

        # Extreme Value Distribution
        from scipy.stats import norm
        from scipy.special import erf, erfc, erfcinv
        cla()
        icdf = lambda x: sqrt(2) * erf(2 * x - 1)

        n = 10
        alpha = icdf(1 - 1 / (n * exp(1)))
        beta = icdf(1 - 1 / n)

        x = arange(-10, 30, 0.1)
        evd = (1 / beta) * exp(-(x - alpha) / beta) * exp(-exp(-(x - alpha) / beta))
        plot(x, evd)
        xlabel("Maximum Value")
        ylabel("Probability")
        savefig("Analysis/Images/extreme_value_distribution.eps")
        # Mean Extreme Value
        cla()

        gamma = 0.57721566490153286060651209008240243104215933593992
        M = lambda n: sqrt(2) * ((-1 + gamma) * (erfcinv(2 - 2 / float(n))) - gamma * erfcinv(2 - 2 / (n * exp(1))))
        n = range(2, 1000)
        mean_max = [M(_) for _ in n]
        plot(n, mean_max)
        xlabel("Samples")
        ylabel("Expected Maximum Value")
        savefig("Analysis/Images/expected_maximum_value.eps")

        cla()
        eps = finfo(float).eps
        N = 1000
        multiplier = arange(0, 5, 0.1)
        samples = ((1 - norm.cdf(M(N) + multiplier)) ** -1) / 44100.0

        semilogy(multiplier, samples)

        samples_1 = ((1 - norm.cdf(M(N) + 1)) ** -1) / 44100.0
        samples_25 = ((1 - norm.cdf(M(N) + 2.5)) ** -1) / 44100.0

        line = Line2D([1, 1], [eps, samples_1], color="black", linestyle="-.", lw=1)
        ax.add_line(line)
        line = Line2D([0, 1], [samples_1, samples_1], color="black", linestyle="-.",
                        lw=1)
        ax.add_line(line)
        ax.text(0.5, samples_1 + 1, "39 seconds", ha="center", va="bottom", size=10)

        line = Line2D([2.5, 2.5], [eps, samples_25], color="black", linestyle="-.", lw=1)
        ax.add_line(line)
        line = Line2D([0, 2.5], [samples_25, samples_25], color="black", linestyle="-.",
                        lw=1)
        ax.add_line(line)
        ax.text(1.25, samples_25 + 5000, "62 hours", ha="center", va="bottom", size=10)

        xlabel("Multiplier")
        ylabel("Seconds")
        savefig("Analysis/Images/maximum_value_probability_multiplier.eps")

    def lowpassSweptSineGeneration(self):
        """ Function to illustrate the steps taken to generate a low pass swept sine.

        Instead of using the Signal Generator, preform the inverse filtering manually
        so that the steps may be illustrated.
        """
        from scipy.signal import butter, lfilter, filtfilt

        self.logger.debug("Entering lowpassSweptSineGeneration")

        T = 64 * 10e-3  # 125 ms
        sample_rate = 44100.0
        f_1 = sample_rate / 2.0
        fft_size = 2 ** 18

        # Generate time vector
        t = arange(0, T, 1 / sample_rate)

        # Generate the signal from 0 to Nyquist frequency
        a = pi * f_1 / T

        s = sin(a * t ** 2)

        plot(1000 * t, s)
        xlabel("Time (ms)")
        ylabel(r"$s(t)$")
        xlim(0, 125)
        ylim(-1.1, 1.1)
        subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.10)
        savefig("Analysis/Images/swept_sine.eps")
        cla()

        # Determine the spectrum
        S = fft(s, fft_size)
        # Inverse of the magnitude spectrum
        iaS = abs(S) ** -1
        liaS = log(iaS)
        liaS -= min(liaS)

        plot(fftfreq(fft_size, 1 / sample_rate)[:fft_size / 2], liaS[:fft_size / 2])
        xlabel("Frequency (Hz)")
        ylabel(r"ln$| S(\omega) | ^ {-1}$")
        xlim(0, sample_rate / 2)
        subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.10)
        savefig("Analysis/Images/inverse_log_spectrum.eps")
        cla()
        # c, similiar to the cepstrum, is the inverse of the logarithmic inverse
        # magnitude spectrum
        c = ifft(log(iaS))

        # Window c to produce m
        m = r_[c[0], 2 * c[1:len(S) / 2 - 1], c[len(S) / 2], zeros(len(S) / 2)]
        plot(m)
        xlabel("samples")
        ylabel(r"$m\left[n\right]$")
        gca().get_yaxis().set_ticks([])
        subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.10)
        ylim(-0.05, 0.05)
        xlim(0, 1000)
        savefig("Analysis/Images/minimum_phase.eps")
        cla()
        # Determine the spectrum of the windowed 'cepstrum'
        M = fft(m, fft_size)

        # Determine the minimum phase inverse filter
        iSmp = exp(M)
        plot(fftfreq(fft_size, 1 / sample_rate)[:fft_size / 2], iSmp[:fft_size / 2])
        xlim(0, sample_rate / 2.0)
        ylim(0.010, 0.030)
        ylabel(r"$X_{mp}^{-1}\left[k\right]$")
        xlabel("Frequency (Hz)")
        subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.10)
        gca().get_yaxis().set_ticks([])
        xlim(0, sample_rate / 2.0)
        savefig("Analysis/Images/inverse_minimum_phase.eps")
        cla()

        # Determine the minimum phase spectrum
        Smp = S * iSmp

        # Determine the minimum phase signal
        smp = ifft(Smp)

        # smp will have fft_size samples, which could be very long
        # reduce to length of the signal specified
        smp = smp[:len(t)]

         # Low pass filter the signal to the upper frequency
        [b, a] = butter(8, 0.9, btype="low")
        smp = lfilter(b, a, smp)

        SMP = abs(rfft(smp, 2 ** 14))
        SMP -= max(SMP)
        S = abs(rfft(s, 2 ** 14))
        S -= max(S)
        S += 13

        plot(rfftfreq(len(SMP), 1 / sample_rate), SMP)
        plot(rfftfreq(len(S), 1 / sample_rate), S)

        title("SMP")
        show()
        # Normalize so that the maximum value is 1
        smp /= max(abs(smp))

        plot(smp)
        show()
        signal = smp

    def windowAnalysis(self):
        """ Function to preform some analysis on various windows used
        to lift the impulse response from the cepstrum.

        """
        self.logger.debug("Entering windowAnalysis")
        N = 2 ** 10
        L = 2 ** 6

        rect = ones(L)
        hann = hanning(L)
        hamm = hamming(L)

        tmp_hann = hanning(20)
        mine = r_[tmp_hann[:10], ones(L - 20), tmp_hann[-10:]]
        RECT = fft(rect, N)
        HANN = fft(hann, N)
        HAMM = fft(hamm, N)
        MINE = fft(mine, N)

        plot(fftfreq(N), 20 * log10(abs(RECT)), label="rect")
        plot(fftfreq(N), 20 * log10(abs(HANN)), label="hanning")
        plot(fftfreq(N), 20 * log10(abs(HAMM)), label="hamming")
        plot(fftfreq(N), 20 * log10(abs(MINE)), label="flat-top")

        legend()
        show()

    def sweptSineTesting(self):
        """ Test some techniques to use with swept sine signals """
        self.logger.debug("Entering sweptSineTesting")

    def resonatorExample(self):
        """ Method to draw the example of the resonator's absorption coefficient
         and the significance of the paramaters.
         """
        self.logger.debug("Entering resonatorExample")

        B = 500.0
        f_0 = 1000.0
        a = 0.8
        H_gen = lambda f: sqrt(a / (1 + ((f - f_0) ** 2) / (B / 2) ** 2))
        f = arange(1, 10000)

        semilogx(f, 1 - (H_gen(f)) ** 2)
        ylim([0, 1])
        show()
    def misc_flatspectrum_plots(self):
        """ Miscilanous plots for the flat spectrum paper """
        self.logger.debug("Entering misc_flatspectrum_plots")

        # Plot piston resistance and reactance functions
        from scipy.special import j1, struve

        x = arange(0, 15, 0.001)

        r1 = lambda x: 1 - 2 * j1(x) / x
        x1 = lambda x: 2 * struve(1, x) / x

        plot(x, r1(x), label="$R_{1}\left(x)\\right)$")
        plot(x, x1(x), linestyle="--", color="black", label="$X_{1}\left(x)\\right)$")

        xlim([0, 13])
        annotate("$R_{1}(x)$", xy=(10, r1(10)), xycoords="data", xytext=(10.5, r1(10.5) - 0.2),
                    arrowprops=dict(arrowstyle="->"))
        annotate("$X_{1}(x)$", xy=(5, x1(5)), xycoords="data", xytext=(6, x1(6) + 0.2),
                    arrowprops=dict(arrowstyle="->"))
        grid(True)
        savefig("Analysis/Images/piston_impedance_functions.eps")

        # Plot the piston resisance for a 100mm piston
        cla()
        a = 0.055
        c = 340
        p0 = 1.2250
        f = arange(100, 10000, 0.01)
        k = 2 * pi * f / c
        Rr = p0 * c * pi * a ** 2 * r1(2 * k * a)

        normalized_Rr = 10 * log10(Rr) - 10 * log10(Rr[-1])
        semilogx(f, normalized_Rr)
        ax = gca()
        ax.set_xticks([16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125,
            160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
            4000, 5000, 6300, 8000, 1000, 12500])
        ax.set_xticklabels([16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125,
            160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
            4000, 5000, 6300, 8000, 1000, 12500])
        xlim(100, 5000)

        for label in ax.get_xticklabels():
            label.set_rotation('vertical')

        grid(True)
        subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.15)
        xlabel("Frequency Hz")
        ylabel("dB re $R_{1}(2ka) = 1$")
        savefig("Analysis/Images/piston_resistance_100mm.eps")

    def wave_guide_propagation(self):
        """ Plots some graphs that are related to the wave propagation
        in wave-guides 
        """
        self.logger.debug("Entering wave_guide_propagation")

        f_n = 1000
        f = arange(0, 5000, 0.1)
        c = 340
        fig = figure()

        ax = fig.gca()
        ax.plot(f, c * sqrt(1 - (f_n / f) ** 2))
        ax.set_xlim(0, 5000)
        ax.axhline(y=c, linestyle="--")

        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels([0, "$f_{n}$", "$2f_{n}$", "$3f_{n}$", "$4f_{n}$", "$5f_{n}$"])

        ax.set_yticks([0, 340])
        ax.set_yticklabels([0, "c"])

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Velocity")

        savefig("Analysis/Images/wave_velocity.eps")

        c = 340.0
        f = c / 2
        p = 1
        kn = 2 * pi * f / c
        fs = 100000.0
        t = arange(0, 0.1, 1 / fs)

        fig = figure()
        ax = fig.add_subplot(311)

        sig = cos(kn * sqrt(c ** 2 * t ** 2 - p ** 2)) / (kn * sqrt(c ** 2 * t ** 2 - p ** 2))
        sig[isnan(sig)] = 0

        ax.plot(t, sig)
        ax.set_xticks([0, p / c, 2 * p / c, 3 * p / c, 4 * p / c])
        ax.set_xticklabels([0, "$p/c$", "$2p/c$", "$3p/c$", "$4p/c$"])
        ax.set_xlim(0, 4 * p / c)

        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)

        ax.axhline(y=0)
        ax.axvline(x=2 * p / c, linestyle="--")

        plt.text(0.01, 0.2, "$\lambda_n=2p$", ha="center", family="sans-serif", size=14)

        ax = fig.add_subplot(312)

        f = c
        kn = 2 * pi * f / c

        sig = cos(kn * sqrt(c ** 2 * t ** 2 - p ** 2)) / (kn * sqrt(c ** 2 * t ** 2 - p ** 2))
        sig[isnan(sig)] = 0

        ax.plot(t, sig)
        ax.set_xticks([0, p / c, 2 * p / c, 3 * p / c, 4 * p / c])
        ax.set_xticklabels([0, "$p/c$", "$2p/c$", "$3p/c$", "$4p/c$"])
        ax.set_xlim(0, 4 * p / c)

        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)

        ax.axhline(y=0)
        ax.axvline(x=2 * p / c, linestyle="--")

        plt.text(0.01, 0.2, "$\lambda_n=p$", ha="center", family="sans-serif", size=14)
        ax.set_ylabel("Signal Amplitude")

        ax = fig.add_subplot(313)

        f = 2 * c
        kn = 2 * pi * f / c

        sig = cos(kn * sqrt(c ** 2 * t ** 2 - p ** 2)) / (kn * sqrt(c ** 2 * t ** 2 - p ** 2))
        sig[isnan(sig)] = 0

        ax.plot(t, sig)
        ax.set_xticks([0, p / c, 2 * p / c, 3 * p / c, 4 * p / c])
        ax.set_xticklabels([0, "$p/c$", "$2p/c$", "$3p/c$", "$4p/c$"])
        ax.set_xlim(0, 4 * p / c)

        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)

        ax.axhline(y=0)
        ax.axvline(x=2 * p / c, linestyle="--")
        plt.text(0.01, 0.2, "$\lambda_n=0.5p$", ha="center", family="sans-serif", size=14)
        ax.set_xlabel("Time")
        
        savefig("Analysis/Images/one_mode_impulse_response.eps")

    def compare_results(self):
        """ Function to compare the absorption coefficient using the cepstral
        technique comparing to the impedance tube.
        """
        self.logger.debug("Entering compare_results")

        measurement_filename = "../testdata/120519_asphalt_13.db"

        alpha = self.loadAbsorptionCoefficient(measurement_filename)
        grapher = Grapher(alpha.measurement_settings)
        fig = figure()

        handler = Object()
        handler.axes = fig.add_subplot(111)
        handler.draw = draw
        grapher.graphAbsorption(alpha.alpha, handler)
        
        a = [0.032116172, 0.034017778, 0.032430265, 0.02675464, 0.192021209, 0.415370952,
                0.372468791, 0.691662969, 0.54285943, 0.338953418, 0.284023669, 0.355485023,
                0.475263874, 0.282777409, 0.595041322]
        f = [100, 125, 160, 200, 250, 300, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000]
        handler.axes.plot(f, a, "x")
        show()

    def windowEffects(self):
        """ Function used to generate figures for the window function section. """
        self.logger.debug("Entering windowEffects")
        
        # Declare system variables
        fs = 44100.0
        
        c = 344.0
        l = 2.0
        mic_l = l / 2.0
        N = 2 ** 18

        [bl, al] = butter(1, 10000 / fs / 2.0, 'low')

        T = 0.1
        t = arange(0, T, 1 / fs)
        s = r_[sin(2 * pi * t * (fs / 2 / (2 * T))), zeros(fs * (1 - T) + 1)]  # Impulse
        s = r_[1, rand(fs)]

        h = r_[1, zeros(fs)]  # perfect reflector
        h = lfilter(bl, al, h)

        y = r_[s, zeros(fs * (l + mic_l) / c)]
        y += r_[zeros(fs * (l + mic_l) / c), ifft(fft(s) * fft(h))]

        cepstrum = ifft(log(abs(fft(y, 2 ** 18)) ** 2))

        t = arange(0, len(cepstrum) / fs, 1 / fs)
        t *= 1000

        plot(t, cepstrum)

        win = r_[hanning(100)[:50], ones(400), hanning(100)[50:]]
        win *= max(abs(cepstrum[100:500]))
        t = arange(0, len(win) / fs, 1 / fs)
        t += 0.0058
        t *= 1000
        plot(t, win, c="black", ls="--")
        xlim(0, 25)
        ylim(-0.1, 0.3)
        grid(True)

        xlabel("quenfrency (ms)")

        savefig("Analysis/Images/lifting_the_impulse_response.eps")
        cla()

        t = arange(0, 1, 1 / fs)
        x = cos(2 * pi * 1000 * t)
        N = 2 ** 18
        n = 1000

        plot(rfftfreq(N / 2 + 1, 1 / fs), rfft(x[:n], N))
        xlim(0, 2000)

        ax = gca()

        ax.set_xticks([1000])
        ax.set_xticklabels(["$f_{0}$"])
        ax.set_yticks([0])

        grid(True)
        axhline(y=0)
        xlabel("Frequency (Hz)")
        ylabel(r"$X\left(f\right)$")
        savefig("Analysis/Images/windowed_sinusoidal_signal.eps")

        cla()

        n = 10

        W = abs(fft(ones(n), N)) ** 2

        plot(fftfreq(N), W)

        enbw = 100
        enbw_rect = r_[0, max(W) * ones(enbw), 0]

        plot(linspace(-0.05, 0.05, enbw + 2), enbw_rect, c="black", ls="--")
        ax = gca()

        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_yticklabels("")

        xlabel("Frequency (Hz)")
        ylabel(r"$\left|W\left(f\right)\right|^{2}$")

        axvline(x=0)

        ylim(0, 110)

        annotate("peak power gain\n" + r"at $\left|W\left(0\right)\right|^{2}$", xy=(0.05, max(W)),
                xytext=(0.15, max(W) - 2), arrowprops=dict(fc="black", width=1, headwidth=5),
                verticalalignment='top', fontsize=10)
        annotate("Equivalent\nNoise Bandwidth", xy=(-0.05, max(W) / 2), xytext=(-0.15, max(W) / 2),
                arrowprops=dict(fc="black", width=1, headwidth=5), horizontalalignment="right",
                verticalalignment="center", fontsize=10)
        annotate("", xy=(0.05, max(W) / 2), xytext=(0.15, max(W) / 2),
                arrowprops=dict(fc="black", width=1, headwidth=5), horizontalalignment="right",
                fontsize=10)

        savefig("Analysis/Images/equivalent_noise_bandwidth.eps")

        cla()

        tukey = lambda N, a: r_[hanning(a * N)[:a * N / 2.0], ones(N * (1 - a)), hanning(a * N)[a * N / 2.0:]]

        N = 220

        plot(ones(N)[N / 2:], c="black", ls="-", label="Rectangle")
        #plot([0, 0], [0, 1], c="black", ls="-")
        plot([N / 2, N / 2], [0, 1], c="black", ls="-")

        plot(tukey(N, 0.25)[N / 2:], c="black", ls="--", label=r"Tukey $\alpha = 0.25$")
        plot(tukey(N, 0.55)[N / 2:], c="black", ls="-.", label=r"Tukey $\alpha = 0.50$")
        plot(tukey(N, 0.75)[N / 2:], c="black", ls=":", label=r"Tukey $\alpha = 0.75$")
        plot(hanning(N)[N / 2:], c="gray", label="Hanning")

        legend()
        leg = gca().get_legend()
        setp(leg.get_texts(), fontsize="small")

        xlim(0, N + 10)
        ylim(0, 1.05)

        ax = gca()
        ax.set_xticks([N / 2])
        ax.set_xticklabels([r"N/2"])

        savefig("Analysis/Images/window_shapes.eps")

        cla()

        # n = 100
        # N = 2 ** 18

        # R = 20 * log10(rfft(ones(n), N))
        # R -= max(R)

        # T25 = 20 * log10(rfft(tukey(n, 0.25), N))
        # T25 -= max(T25)

        # T50 = 20 * log10(rfft(tukey(n, 0.50), N))
        # T50 -= max(T50)

        # T75 = 20 * log10(rfft(tukey(n, 0.75), N))
        # T75 -= max(T75)

        # H = 20 * log10(rfft(hanning(n), N))
        # H -= max(H)

        # semilogx(rfftfreq(N / 2 + 1) * n, R, label="Rectangle", c="black", ls="-")
        # semilogx(rfftfreq(N / 2 + 1) * n, T25, label=r"Tukey $\alpha = 0.25$", c="black", ls="--")
        # semilogx(rfftfreq(N / 2 + 1) * n, T50, label=r"Tukey $\alpha = 0.50$", c="black", ls="-.")
        # semilogx(rfftfreq(N / 2 + 1) * n, T75, label=r"Tukey $\alpha = 0.75$", c="black", ls=":")
        # semilogx(rfftfreq(N / 2 + 1) * n, H, label="Hanning", c="gray", ls="-")

        # xlim(0.1, 15)
        # ylim(-60, 0)

        # legend(loc="lower left")
        # leg = gca().get_legend()
        # setp(leg.get_texts(), fontsize="small")
        # show()

        cla()

        n = 220
        N = 2 ** 18

        fig, (ax1, ax2, ax3, ax4) = subplots(4, 1, sharex=True, sharey=True)
        ax1.semilogx(rfftfreq(N / 2 + 1, 1 / fs), rfft(ones(n), N), label="rect")
        ax1.text(800, 50, "Rectangle Window", horizontalalignment="center", verticalalignment="center")
        ax1.grid(True)

        ax2.semilogx(rfftfreq(N / 2 + 1, 1 / fs), rfft(tukey(n, 0.25), N), label="a = 0.25")
        ax2.text(800, 50, r"Tukey Window, $\alpha = 0.25$", horizontalalignment="center", verticalalignment="center")
        ax2.grid(True)

        ax3.semilogx(rfftfreq(N / 2 + 1, 1 / fs), rfft(tukey(n, 0.50), N), label="a = 0.50")
        ax3.text(800, 50, r"Tukey Window, $\alpha = 0.50$", horizontalalignment="center", verticalalignment="center")
        ax3.grid(True)

        ax4.semilogx(rfftfreq(N / 2 + 1, 1 / fs), rfft(tukey(n, 0.75), N), label="a = 0.75")
        ax4.text(800, 50, r"Tukey Window, $\alpha = 0.75$", horizontalalignment="center", verticalalignment="center")
        ax4.grid(True)

        axes = gca()

        axes.set_xticks([100, 125, 160, 200, 250, 315, 400,
                        500, 630, 800, 1000, 1250, 1600, 2000,
                        2500, 3150, 4000, 5000])
        axes.set_xticklabels([100, 125, 160, 200, 250, 315, 400,
                        500, 630, 800, 1000, 1250, 1600, 2000,
                        2500, 3150, 4000, 5000], rotation="vertical")
        axes.set_yticks([0])
        axes.set_yticklabels([0])

        ylim(-100, 100)
        xlim(100, 5000)

        fig.subplots_adjust(bottom=0.15, top=0.98, right=0.98, left=0.05)
        xlabel("Frequency (Hz)")

        savefig("Analysis/Images/window_frequency_response.eps")

    def mlsExciationSignal(self):
        """ function to create graphs to illustrate generating mls signals """
        self.logger.debug("entering mlsExciationSignal")

        # get the mls signal
        mls_db = MlsDb()

        mls = mls_db.getMls(5)

        # convert into -1's and 1s
        mls = -2 * mls + 1

        # hold values to produce plots
        mls_plot = reduce(lambda x, y: x + y, zip(mls, mls))
        bins_plot = reduce(lambda x, y: x + y, zip(arange(len(mls)), arange(1, len(mls) + 1)))

        plot(bins_plot, mls_plot)
        ylim(-1.1, 1.1)
        yticks([-1, 0, 1])
        xlabel("Bins")
        ylabel("Amplitude")
        axhline(y=-1, ls="--", color="gray")
        axhline(y=0, ls="--", color="gray")
        axhline(y=1, ls="--", color="gray")
        subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.10)
        xlim(0, 31)
        
        savefig("Analysis/Images/5_tap_mls_signal.eps")

    def burstsVersusRepetitions(self):
        """ Illustration of number of bursts of an MLS signal versus the number of repetitions.
        """
        self.logger.debug("Entering burstsVersusRepetitions")
        self.measurement_settings["signal reps"] = 2
        signal_gen = SignalGenerator(self.measurement_settings)

        plot(signal_gen.signal)
        xlim(0, 500e3)
        show()

class Object(object):
    pass

if __name__ == "__main__":
    logger = logging.getLogger("Alpha")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    analysis = AnalysisDelegate()
    analysis.misidentificationAnalysis()
    #analysis.synchronizeAnalysis()
