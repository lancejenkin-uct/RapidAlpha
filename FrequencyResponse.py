#!/usr/bin/env python
""" Used to determine the frequency response of a loudspeaker.
    
    The system response is determined by a method specific to the type of
    excitation signal used. If it is a swept sine, then the system response is
    the inverse Fourier Transform of the magnitude  of the Fourier Transform of
    the response, ie :
        h[n] ~= ifft[ abs[ fft[ x[n] ] ] ]  
    If the excitation is either MLS or IRS, then the circular deconvolution 
    property of MLS-type signals are used. The generators response is 
    deconvolved from the microphone response so that the frequency response of 
    the loudspeaker may be determined.

"""

import logging
from pylab import *
from scipy.signal import deconvolve

from MlsDb import MlsDb

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class FrequencyResponse(object):

    def __init__(self, microphone_signals, generator_signals,
        measurement_settings):
        """ Default constructor to create the FrequencyResponse object.

            :param microphone_signals:
                An array of signals recorded from the microphone.
            :type microphone_signals:
                array of float
            :param generator_signals:
                An array of signals recorded directly from the generator.
            :type generator_signals:
                array of float
            :param measurement_settings:
                A dictionary containing the settings used to measure the
                signals. As well as the location of the impulse in the
                microphone signal and the generator signal.
            :type measurement_settings:
                dict
        """
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating Frequency Response Object")

        # Set up object variables
        self.microphone_signals = microphone_signals
        self.generator_signals = generator_signals
        self.measurement_settings = measurement_settings

        # Create new object to access MLS database
        self.mls_db = MlsDb()

        self.determineFreqResp()

    def determineFreqResp(self):
        """ Determine the frequency response of device under test, usually a
        loudspeaker.

        First, the system response of the microphone response and the generator
        response are determined.  From this, the generator response is
        deconvolved from the microphone response.  The resulting response is
        then transformed into the frequency domain, giving the frequency
        response of the device.
        """
        self.logger.debug("Entering determineFreqResp")

        # First the signals need be extracted and averaged together
        self._extractSignals()

        # Then determine the system response
        self._determineSystemResponse()

        # deconvolve the generator response from the microphone response
        self._deconvoleSignals()

        # Extract the impulse response
        self._extractImpulseResponse()

    def _extractImpulseResponse(self):
        """ Extract the impulse response of the device under investigation, by
        windowing it out directly from the system response.
        """
        self.logger.debug("Entering _extractImpulseResponse")

        window_length = int(self.measurement_settings["window length"])
        taper_length = int(self.measurement_settings["taper length"])
        fft_size = int(self.measurement_settings["fft size"])

        taper = hanning(2 * taper_length)

        self.impulse_response = self.microphone_response[:window_length]
        self.impulse_response[-taper_length:] *= taper[-taper_length:]

        self.frequency_response = fft(self.impulse_response, fft_size)

    def _deconvoleSignals(self):
        """ Deconvolves the generator signal from the microphone signal,
            removing the effects of the generator on the signal.
        """
        self.logger.debug("Entering _deconvoleSignals")

        self.system_response = deconvolve(self.microphone_response,self.generator_response)
        # deconvolve function returns a tuple (remainder, deconvolved signal)
        self.system_response = ifft(fft(self.microphone_response) / fft(self.generator_response))


    def _determineSystemResponse(self):
        """ Determines the system response of both the microphone and generator
            recorded signals.
        """
        self.logger.debug("Entering _determineSystemResponse")

        # If MLS / IRS signal is used, use the circular convolution property
        # to determine the system response.
        signal_type = self.measurement_settings["signal type"]

        if signal_type.lower() == "maximum length sequence":
            number_taps = int(self.measurement_settings["mls taps"])

            self.microphone_response = self.mls_db.getSystemResponse(self.average_microphone_response, number_taps)
            self.generator_response = self.mls_db.getSystemResponse(self.average_generator_response, number_taps)
        elif signal_type.lower() == "inverse repeat sequence":
            number_taps = int(self.measurement_settings["mls taps"])

            # Preform the circular convolution manually!
            mls = self.mls_db.getMls(number_taps)
            mls = -2 * mls + 1
            irs = array([])
            for index, sample in enumerate(r_[mls, mls]):
                if index % 2 == 0:
                    irs = r_[irs, sample]
                else:
                    irs = r_[irs, -1 * sample]

            assert (len(self.average_microphone_response) == len(irs))
            assert (len(self.average_generator_response) == len(irs))

            self.microphone_response = ifft(fft(self.average_microphone_response) * fft(irs[-1::-1]))
            self.microphone_response = self.microphone_response[:2 ** number_taps - 1]
            self.generator_response = ifft(fft(self.average_generator_response) * fft(irs[-1::-1]))
            self.generator_response = self.generator_response[:2 ** number_taps - 1]
        else:
            self.microphone_response = self.average_microphone_response
            self.generator_response = self.average_microphone_response

    def _extractSignals(self):
        """ Extract the microphone and generator signals from the raw signals.

        The microphone and generator signal are preceded by an impulse.  The
        location of the impulse is given in the signal settings.  The delay
        from the impulse to the start of the signal is also specified in the
        signal settings.  The microphone and generator can therefore be
        extracted from the start of the signal.
        """
        self.logger.debug("Entering _extractSignals")

        sample_rate = float(self.measurement_settings["sample rate"])
        signal_type = str(self.measurement_settings["signal type"])
        impulse_location = int(self.measurement_settings["microphone impulse location"])
        impulse_signal_delay = float(self.measurement_settings["impulse delay"])

        impulse_signal_samples = impulse_signal_delay * sample_rate
        # Since the impulse is also a sample, we need to add one before the actual
        # start of the signal.

        signal_start = impulse_location + impulse_signal_samples

        self.microphone_responses = []
        for signal in self.microphone_signals:
            self.microphone_responses.append(signal[signal_start:])
        self.average_microphone_response = average(self.microphone_responses[1:], axis=0)

        impulse_location = int(self.measurement_settings["generator impulse location"])
        signal_start = impulse_location + impulse_signal_samples

        self.generator_responses = []
        for signal in self.generator_signals:
            self.generator_responses.append(array(signal[signal_start:]))
        self.average_generator_response = average(self.generator_responses[1:], axis=0)

        if signal_type.lower() == "maximum length sequence":
            mls_reps = int(self.measurement_settings["mls reps"])
            mls_taps = int(self.measurement_settings["mls taps"])
            assert(mls_reps > 0)

            mls_length = 2 ** mls_taps - 1
            mls_sig = self.average_microphone_response[mls_length:(mls_length * (mls_reps + 1))]
            mls_array = reshape(mls_sig, (mls_reps, -1))
            self.average_microphone_response = average(mls_array, axis=0)
            mls_sig = self.average_generator_response[mls_length:(mls_length * (mls_reps + 1))]
            mls_array = reshape(mls_sig, (mls_reps, -1))
            self.average_generator_response = average(mls_array, axis=0)
        elif signal_type.lower() == "inverse repeat sequence":
            mls_reps = int(self.measurement_settings["mls reps"])
            mls_taps = int(self.measurement_settings["mls taps"])
            assert(mls_reps > 0)

            mls_length = 2 ** mls_taps - 1
            irs_length = 2 * mls_length

            irs_sig = self.average_microphone_response[irs_length:(irs_length * (mls_reps + 1))]
            irs_array = reshape(irs_sig, (mls_reps, -1))
            self.average_microphone_response = average(irs_array, axis=0)
            irs_sig = self.average_generator_response[irs_length:(irs_length * (mls_reps + 1))]
            irs_array = reshape(irs_sig, (mls_reps, -1))
            self.average_generator_response = average(irs_array, axis=0)

