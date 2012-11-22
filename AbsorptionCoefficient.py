#!/usr/bin/env python
""" Analyzes the recorded response and determines the absorption coefficient.

    Using Cepstral techniques to determine the absorption by taking the power
    cepstrum of the microphone and generator signals.  The generator cepstrum
    can then be subtracted from the microphone cepstrum.  The impulse response
    of the material can then be directly 'lifted' of the power cepstrum.
"""

import logging
from scipy.signal import butter, lfilter, deconvolve, cheby1, filtfilt, decimate
from scipy.signal import *
from numpy import *
from pylab import *
from MlsDb import MlsDb

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class AbsorptionCoefficient(object):

    def __init__(self, microphone_signals, generator_signals,
        measurement_settings, analysis_settings):

        """ Constructor for AbsorptionCoefficient object.

        :param microphone_signals:
            An array of signals recorded from the microphone.
        :type microphone_signals:
            array of float
        :param generator_signals:
            An array of signals recorded directly from the generator.
        :type generator_signals:
            array of float
        :param measurement_settings:
            A dictionary containing the settings used to measure the signals.
            As well as the location of the impulse in the microphone signal and
            the generator signal.
        :type measurement_settings:
            dict
        :param analysis_settings:
            A dictionary containing the settings used to analyze the signals.
        :type analysis_settings:
            dict
        """
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating AbsorptionCoefficient Object")

        self.microphone_signals = microphone_signals
        self.generator_signals = generator_signals
        self.measurement_settings = measurement_settings
        self.analysis_settings = analysis_settings

        self.mls_db = MlsDb()

        self.determineAlpha()

    def determineAlpha(self):
        """ Determine the absorption coefficient of the material under test.

        The absorption coefficient is determined directly by lifting the
        impulse response from the power cepstrum.  The power cepstrum is the
        generator's power cepstrum subtracted from microphone's power cepstrum.
        The microphone and generator power cepstra are determined from the
        averaged recorded microphone and generator response.  These averaged
        signals are filtered with an anti-aliasing filter, and decimated.
        """
        self.logger.debug("Entering determineAlpha")

        # Extract the signals from the recorded responses, and average
        self._extractSignals()

        # Determine the system response of the signals
        self._determineResponse()

        # LPF the signals and decimate
        self._downsampleSignals()

        # Determine the Cepstrum
        self._determineCepstrum()

        # Lift the impulse response
        self._liftImpulseResponse()

        # Determine the absorption coefficient
        self._determineAbsorptionCoefficient()

    def _determineAbsorptionCoefficient(self):
        """ By taking the Fourier transform of the impulse response, the absorption coefficient can be determined.

        The absorption coefficient is simply,
         a = 1 - |F{h}|^2
         with:
            a - the absorption coefficient
            h - the impulse response
            F{} - The Fourier transform operator
        """
        self.logger.debug("Entering _determineAbsorptionCoefficient")

        fft_size = int(self.measurement_settings["fft size"])
        self.alpha = 1 - abs(fft(self.impulse_response, fft_size)) ** 2

    def _determineResponse(self):
        """ For Psuedorandom signal, determine the system response using the autocorrellation property of the signal.

        """
        self.logger.debug("Entering _determineResponse")

        # If MLS signal, then utilize the circular convolution property
        signal_type = self.measurement_settings["signal type"]
        sample_rate = float(self.measurement_settings["sample rate"])

        if signal_type.lower() == "maximum length sequence":
            number_taps = int(self.measurement_settings["mls taps"])

            self.microphone_response = self.mls_db.getSystemResponse(self.average_microphone_response, number_taps)

            self.generator_response = self.mls_db.getSystemResponse(self.average_generator_response, number_taps)

            self.system_response = ifft(fft(self.microphone_response) / fft(self.generator_response))

        elif signal_type.lower() == "inverse repeat sequence":
            number_taps = int(self.measurement_settings["mls taps"])
            # TODO: Re-factor this out of absorption coefficient
            mls = self.mls_db.getMls(number_taps)
            mls = -2 * mls + 1
            irs = array([])
            for index, sample in enumerate(r_[mls, mls]):
                if index % 2 == 0:
                    irs = r_[irs, sample]
                else:
                    irs = r_[irs, -1 * sample]


            self.microphone_response = irfft(rfft(self.average_microphone_response) * rfft(irs[-1::-1]))
            self.generator_response = irfft(rfft(self.average_generator_response) * rfft(irs[-1::-1]))

            # The output of the auto-correlation of an irs signal is a + impulse at 0 and a
            # - impulse at N / 2.  One is only interested in the positive impulse, so extract
            # up to N / 2 samples, or the MLS length of 2 ^ (number of taps) - 1.  BUT, since
            # the DFT is circular, samples at the end wrap to be connected to sample's at the
            # start - if the start of the IRS signal is missed, then part of the - impulse
            # response will corrupt the positive impulse response.

            impulse_length = len(mls)
            window = hanning(0.1 * impulse_length)

            self.microphone_response = self.microphone_response[:0.9 * impulse_length]
            self.microphone_response[-len(window) / 2 :] *= window[-len(window) / 2 :]
            tmp_response = self.microphone_response[-0.1 * impulse_length:]
            tmp_response[:len(window) / 2] *= window[:len(window) / 2]
            append(self.microphone_response, tmp_response)

            self.generator_response = self.generator_response[:0.9 * impulse_length]
            self.generator_response[-len(window) / 2 :] *= window[-len(window) / 2 :]
            tmp_response = self.generator_response[-0.1 * impulse_length:]
            tmp_response[:len(window) / 2] *= window[:len(window) / 2]
            append(self.generator_response, tmp_response)

        else:
            self.microphone_response = self.average_microphone_response
            self.generator_response = self.average_generator_response
    def _extractSignals(self):
        """ Extract the microphone and generator signals from the raw signals.

        The microphone and generator signal are preceded by an impulse.  The
        location of the impulse is given in the signal settings.  The delay
        from the impulse to the start of the signal is also specified in the
        signal settings.  The microphone and generator can therefore be extracted
        from the start of the signal.
        """
        self.logger.debug("Entering _extractSignals")

        sample_rate = float(self.measurement_settings["sample rate"])
        signal_type = str(self.measurement_settings["signal type"])
        impulse_location = int(self.measurement_settings["microphone impulse location"])
        impulse_signal_delay = float(self.measurement_settings["impulse delay"])

        impulse_signal_samples = impulse_signal_delay * sample_rate

        signal_start = impulse_location + impulse_signal_samples

        self.microphone_responses = []
        for signal_index, signal in enumerate(self.microphone_signals):
            # Ignore first signal, unless there is only one signal
            if signal_index > 0 or len(self.microphone_signals) == 1:
                self.microphone_responses.append(signal[signal_start:])
        self.average_microphone_response = average(self.microphone_responses, axis=0)

        impulse_location = int(self.measurement_settings["generator impulse location"])
        signal_start = impulse_location + impulse_signal_samples

        self.generator_responses = []
        for signal_index, signal in enumerate(self.generator_signals):
            # Ignore first signal, unless there is only one signal
            if signal_index > 0 or len(self.generator_signals) == 1:
                self.generator_responses.append(array(signal[signal_start:]))

        self.average_generator_response = average(self.generator_responses,axis=0)

        if signal_type.lower() == "maximum length sequence":
            mls_reps = int(self.measurement_settings["mls reps"])
            mls_taps = int(self.measurement_settings["mls taps"])
            assert(mls_reps > 0)

            mls_length = 2 ** mls_taps - 1
            mls_sig = self.average_microphone_response[mls_length:(mls_length * (mls_reps))]
            mls_array = reshape(mls_sig, (mls_reps - 1, -1))
            self.average_microphone_response = average(mls_array, axis=0)

            mls_sig = self.average_generator_response[mls_length:(mls_length * (mls_reps))]
            mls_array = reshape(mls_sig, (mls_reps - 1, -1))
            self.average_generator_response = average(mls_array, axis=0)

        elif signal_type.lower() == "inverse repeat sequence":
            mls_reps = int(self.measurement_settings["mls reps"])
            mls_taps = int(self.measurement_settings["mls taps"])
            assert(mls_reps > 1)

            mls_length = 2 ** mls_taps - 1
            irs_length = 2 * mls_length

            irs_sig = self.average_microphone_response[irs_length:(irs_length * (mls_reps))]
            irs_array = reshape(irs_sig, (mls_reps - 1, -1))
            self.average_microphone_response = average(irs_array, axis=0)

            irs_sig = self.average_generator_response[irs_length:(irs_length * (mls_reps))]
            irs_array = reshape(irs_sig, (mls_reps - 1, -1))
            self.average_generator_response = average(irs_array, axis=0)

    def _downsampleSignals(self):
        """ Low pass filter microphone response and generator response, then down
            sample by an Integer factor.

            The response signals is either the raw signal, or if the MLS signal
            was used to excite the system, the system response determined by
            the convolution property of MLS signals.
        """
        self.logger.debug("Entering _downsampleSignals")

        # Get required variables
        sample_rate = float(self.measurement_settings["sample rate"])
        decimation_factor = int(self.analysis_settings["decimation factor"])
        filter_order = int(self.analysis_settings["antialiasing filter order"])

        # Down sample the responses
        if decimation_factor > 1:
            self.microphone_response_ds = decimate(self.microphone_response, decimation_factor, ftype="fir")
            self.generator_response_ds = decimate(self.generator_response, decimation_factor, ftype="fir")
        else:
            self.microphone_response_ds = self.microphone_response
            self.generator_response_ds = self.generator_response
        #self.system_response_ds = decimate(self.system_response, decimation_factor)
        #self.microphone_response_ds = lfilter(b, a, self.microphone_response[::decimation_factor])
        #self.generator_response_ds = lfilter(b, a, self.generator_response[::decimation_factor])

    def _liftImpulseResponse(self):
        """ Lift the impulse response off the power cepstrum.

        Using the specified window settings, create a window to lift the
        impulse response off the power cepstrum.
        """
        self.logger.debug("Entering _liftImpulseResponse")

        # Get required variables
        window_type = str(self.analysis_settings["window type"])
        window_start = float(self.analysis_settings["window start"])
        window_end = float(self.analysis_settings["window end"])
        taper_length = float(self.analysis_settings["taper length"])
        sample_rate = float(self.measurement_settings["sample rate"])
        decimation_factor = float(self.analysis_settings["decimation factor"])

        effective_sample_rate = sample_rate / decimation_factor

        window_length = window_end - window_start

        window_samples = window_length * effective_sample_rate
        taper_samples = taper_length * effective_sample_rate

        # Create the window
        tapers = hanning(2 * taper_samples)
        if window_type == "one sided":
            self.window = r_[ones(window_samples - taper_samples), tapers[taper_samples:]]
        elif window_type == "two sided":
            self.window = r_[tapers[:taper_samples],
                        ones(window_samples - (2 * taper_samples)),
                        tapers[taper_samples:]]
        self.window[-1] = 0
        # Lift impulse response
        start = window_start * effective_sample_rate
        end = start + len(self.window)

        self.impulse_response = self.power_cepstrum[start:end].copy()
        self.impulse_response *= self.window

    def _determineCepstrum(self):
        """ Determines the power cepstra of the averaged microphone and
            generator signals.

            The power cepstra is defined as:
                c(t) = ifft(log(abs(fft(x(t)) ** 2))

            "The inverse Fourier Transform of the logarithmic squared magnitude
            of the Fourier Transform of a signal"

            The scaled generator cepstrum is subtracted from the microphone
            cepstrum to determine the overall system cepstrum.

            The algorithm to determine the scaling factor, beta, is as follows:
            1. Determine where the squared modulus of the generator cepstrum reaches
                1% of it's peak.
            2. Window out the generator cepstrum from that sample to the sample before
                the start of the impulse response lifter.
            3. Window out the same section from the microphone cepstrum.
            4. Beta is then equal to the ratio of the windowed generator cepstrum to
                that of the windowed microphone cepstrum.

        """
        self.logger.debug("Entering _determineCepstrum")

        # required variables
        fft_size = int(self.measurement_settings["fft size"])
        window_start = float(self.analysis_settings["window start"])
        sample_rate = float(self.measurement_settings["sample rate"])
        decimation_factor = float(self.analysis_settings["decimation factor"])

        effective_sample_rate = sample_rate / decimation_factor

        cepstrum = lambda x: irfft(log(abs(rfft(x, fft_size)) ** 2))

        # Determine Cepstra
        self.microphone_cepstrum = cepstrum(self.microphone_response_ds)
        self.generator_cepstrum = cepstrum(self.generator_response_ds)
        #self.system_cepstrum = cepstrum(self.system_response_ds)

        #self.power_cepstrum = ifft(log(abs(fft(self.microphone_response_ds, fft_size) /  fft(self.generator_response_ds, fft_size)) ** 2))

        self.power_cepstrum = self.microphone_cepstrum - self.generator_cepstrum





