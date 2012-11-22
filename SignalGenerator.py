#!/usr/bin/env python
""" Provides a class to generate signals to use with Cepstral Anaylsis

Cepstral analysis requires signals that have flat spectra.  This signal
generate is used to generate two different signals containing flat spectra,
namely Swept Sine and Maximum Length Sequence (MLS).  There is also an option
to modified the Swept Sine signal to remove the ripples in the spectrum.

MLS signals are precalculated, and retrieved from a database using a seperate
interface.
"""
from pylab import *
from scipy.signal import butter, lfilter, filtfilt, iirdesign, firwin
from scipy.fftpack import rfft, rfftfreq
import logging

from MlsDb import MlsDb

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


class SignalGenerator(object):

    def __init__(self, parameters):
        """Constructor to create signal generator

        :param parameters:
            The parameters to use to generate the signal.  It is a dictionary
            containing information, such as sample rate, signal type and other
            depending on the signal.
        :type parameters:
            dict
        """
        self.logger = logging.getLogger("Alpha")

        self.parameters = parameters

        self.mls_db = MlsDb()

        self.generateSignal()

    def setParameters(self, parameters):
        """" Update the signal parameters, and update the signal.

        :param parameters:
            The signal parameters to use to generate the signal.
        :type parameters:
            dict
        """
        self.logger.debug("Entering setParameters")

        self.parameters = parameters
        self.generateSignal()

    def generateSignal(self):
        """Generate the signal specified by the parameters.

        Using the parameters specified, a new signal will be generated.  The
        signal will then be accessible by the signal property of the
        SignalGenerator object.
        """
        self.logger.debug("Entering generateSignal")

        # Get parameters
        signal_type = str(self.parameters["signal type"])

        lpf_enabled = int(self.parameters["lpf enabled"])
        lpf_cutoff = int(self.parameters["lpf cutoff"])
        lpf_order = int(self.parameters["lpf order"])
        hpf_enabled = int(self.parameters["hpf enabled"])
        hpf_cutoff = int(self.parameters["hpf cutoff"])
        hpf_order = int(self.parameters["hpf order"])

        pad_signal = int(self.parameters["pad signal"])
        signal_reps = int(self.parameters["signal reps"])

        if "gain" in self.parameters:
            gain = float(self.parameters["gain"])
            if gain <= 0:
                # Gain is in dB; convert to decimal
                gain = 10 ** (gain / 20.0)
        else:
            gain = 0.562341325190349 # -6 dB
        # Generate the signal
        if signal_type.lower() == "swept sine":
            self.generateSweptSine()
        if signal_type.lower() == "low pass swept sine":
            self.generateLowPassSweptSine()
        if signal_type.lower() == "maximum length sequence":
            self.generateMls()
        if signal_type.lower() == "inverse repeat sequence":
            self.generateIRS()

        # Filter the signal
        if lpf_enabled == 1:
            self.filterSignal(lpf_cutoff, lpf_order, "low")
        if hpf_enabled == 1:
            self.filterSignal(hpf_cutoff, hpf_order, "high")


        #self.inverseFilter()
        # Adjust gain
        # TODO: Get gain from database
        self.signal /= max(abs(self.signal))
        self.signal *= gain

        # Pad the filter with impulse, and delay at the end
        if pad_signal == 1:
            self.padSignal()

        # Repeat the signal to improve SNR
        tmp_signal = self.signal
        for i in range(signal_reps):
            self.signal = r_[self.signal, tmp_signal]

    def inverseFilter(self):
        """
            Loads the frequency response of the loud speaker,
            fits a cosine series to the logarithm of squared magnitude of the frequency response.
            determines the magnitude of the spectrum by
                |Sf| = |St| / |Sl|
            with:
                Sf the frequency repsonse of the consentation filter
                St is the target spectrum
                Sl is the loudspeaker response

            determines the minimum phase of the compensation filter:
        """
        self.logger.debug("Entering inverseFilter")
        import BaseDelegate
        # Create new base delegate
        bd = BaseDelegate.BaseDelegate()

        # Load the frequency response
        measurement_file = "../testdata/120802_frequency_response_20.fdb"

        freq_response = bd.loadFrequencyResponse(measurement_file)
        sample_rate = float(freq_response.measurement_settings["sample rate"])

        N = len(freq_response.frequency_response)
        # find the bin of 4000 Hz
        bin = float(floor(4410* N / sample_rate))
        freq = freq_response.frequency_response

        # We are solving Ax = 2 * log10(abs(y))
        # Determine A
        M = 20
        k = arange(bin)

        a = array([])
        for m in range(M):
            a = r_[a, cos(2 * pi * k * m / bin)]
        A = matrix(reshape(a, (M, bin)))

        # Determine the weights
        W = pinv(A).transpose()*asmatrix(2 * log10(abs(freq[:bin]))).transpose()

        # Create 2 * log10(abs(y))
        s = zeros(bin)
        for m, w in enumerate(W):
            s += w[0,0] * cos(2 * pi * k * m / bin)

        # target spectrum is now
        mix_samples = ceil(bin * 0.1)
        # create first half of s
        transistion = linspace(1, 0, mix_samples) * s[-mix_samples:] + linspace(0, 1, mix_samples) * 2 * log10(freq_response.frequency_response[bin - mix_samples: bin])
        s = r_[s[:bin - mix_samples], transistion, 2 * log10(freq_response.frequency_response[bin:N / 2])]

        # mirror it
        s = r_[s, s[::-1]]

        plot(s)
        plot(2*log10(freq_response.frequency_response))
        show()

        S = 10 ** (s / 2.0)
        #plot(S,  "--")
        #plot(freq_response.frequency_response)
        #show()
        # compensation filter
        X = fft(self.signal, N)
        Sc = abs(freq_response.frequency_response) / abs(X)

        #Sc = abs(S) / abs(freq_response.frequency_response)

        # To ensure that the filter is causal, and the impulse response is as short as possible in the time domain
        # determine the minimum phase to use with the filter
        c = ifft(log(abs(Sc) ** -1), N)
        m = r_[c[0], 2 * c[1:N / 2.0 - 1], c[N/2] ]
        m = r_[m, zeros(N - len(m))]

        Scmp = exp(fft(m, N))

        Y = Scmp * X
        x = ifft(Y)

        x = x[:len(self.signal)]

        self.signal = x / max(abs(x))



    def generateSweptSine(self):
        """Generate a linear swept sine wave from lower frequency to an upper
           frequency in a specific length of time.

           A swept sine signal sweeps in frequency from a lower frequency, f_0,
           to a upper frequency, f_1, in a given time, T.

           a = pi * ( f_1 - f_0 ) / T
           b = 2 * pi * f_0

           s = sin( (at + b) * t)

           The parameters for the swept sine are stored in the parameters
           dictionary.
        """
        self.logger.debug("Entering generateSweptSine")

        # Get signal parameters
        if "lower frequency" in self.parameters:
            f_0 = int(self.parameters["lower frequency"])
        else:
            f_0 = 0
        f_1 = int(self.parameters["upper frequency"])
        T = float(self.parameters["signal length"])
        sample_rate = float(self.parameters["sample rate"])
        signal_length = float(self.parameters["signal length"])

        # Generate time
        t = arange(0, signal_length, 1 / sample_rate)

        # Generate the signal
        a = pi * (f_1 - f_0) / T
        b = 2 * pi * f_0

        self.signal = sin((a * t + b) * t)

    def generateLowPassSweptSine(self):
        """Generates a swept sine signal, the inverse filters the signal to
            reduce the ripples in the spectrum.

        Creates a minimum phases inverse filter to create a flat spectrum for
        the swept sine.  It should be noted that this algorithm creates a flat
        spectrum up to half of the Nyquist ( Fs / 2 ); where as for cepstral
        techniques a spectrum up to a specified frequency (in this case, the
        upper frequency of the swept sine) is ideal.

        This function therefore assumes a sampling rate of 2 * upper frequency
        of the swept sine, inverse filters the signal, and re-samples the signal
        up to the real sampling rate.

        The algorithm is given as:
          First a swept sine is generated using
              s = sin((a * t + b) * t)
          where:
              a = pi * ((sample_rate / 2) - f_0) / T
              b = 2 * pi * freq1
              T is the signal length
              f_1, f_0 are the upper and lower frequencies

          Then zero pad to 4096 points, and get
              S = FFT(s)

          Take the inverse of S.

          The minimum phase is generated by the following algorithm
              cp = IFFT(log|S| ^ -1)

          This function is then windowed as follows:
                       / cp[n]         n = 0, N/2
              m[n] =  | 2 * cp[n]      1 <= n < N/2
                       \ 0             N/2 < n <= N-1
          Then
              Re[FFT(m)] is equal to log|S| ^ -1
              Im[FFT(m)] is the minimum phase function ie <S_mp

          The minimum phase inverse spectrum is then given by
              S_mp ^ -1 = e ^ { IFFT(m) }
        """
        self.logger.debug("Entering generateModifiedSweptSine")

        # Get signal parameters
        f_0 = int(self.parameters["lower frequency"])
        f_1 = int(self.parameters["upper frequency"])
        T = float(self.parameters["signal length"])
        sample_rate = float(self.parameters["sample rate"])
        fft_size = int(self.parameters["fft size"])
        signal_length = float(self.parameters["signal length"])

        # Generate time vector
        t = arange(0, signal_length, 1 / sample_rate)

        # Generate the signal from 0 to Nyquist frequency
        s = sin(2 * pi * (((sample_rate / 2)   - 0) / (2 * T) * t + 0) * t)


        # Determine the spectrum
        S = fft(s, fft_size)

        # Inverse of the magnitude spectrum
        iaS = abs(S) ** -1

        # c, similar to the cepstrum, is the inverse of the logarithmic inverse
        # magnitude spectrum
        c = ifft(log(iaS))

        # Window c to produce m
        m = r_[c[0], 2 * c[1:len(S) / 2 - 1], c[len(S) / 2], zeros(len(S) / 2)]

        # Determine the spectrum of the windowed 'cepstrum'
        M = fft(m, fft_size)

        # Determine the minimum phase inverse filter
        iSmp = exp(M)

        # Determine the minimum phase spectrum
        Smp = S * iSmp

        # Determin the minimum phase signal
        smp = ifft(Smp)

        # smp will have fft_size samples, which could be very long
        # reduce to length of the signal specified
        smp = smp[:len(t)]

         # Low pass filter the signal to the upper frequency
        [b, a] = butter(8, 0.8 * f_1 / (sample_rate / 2), btype="low")
        #smp = lfilter(b, a, smp)

        # Normalize so that the maximum value is 1
        smp /= max(abs(smp))

        self.signal = smp

    def generateMls(self):
        """Fetches the MLS signal from the database with the desired number of
            taps, and maps the values from {0,1} -> {+1, -1}

        """
        self.logger.debug("Entering generateMls")

        # Get signal parameters
        taps = int(self.parameters["mls taps"])
        reps = int(self.parameters["mls reps"])

        mls = self.mls_db.getMls(taps)

        mls = -2 * mls + 1

        repeated_mls = mls

        for i in range(reps + 1):
            repeated_mls = r_[repeated_mls, mls]

        self.signal = repeated_mls

    def generateIRS(self):
        """ Creates Inverse Repeat Sequence by fetching the required number of
            taps from the MLS db, then forming the IRS signal as follows:
                x[n] = s[n]     n is even
                x[n] = -s[n]    n is odd
        """
        self.logger.debug("Entering generateIRS")

        # Get signal parameters
        taps = int(self.parameters["mls taps"])
        reps = int(self.parameters["mls reps"])

        mls = self.mls_db.getMls(taps)

        mls = -2 * mls + 1

        irs = array([])
        for index, sample in enumerate(r_[mls, mls]):
            if index % 2 == 0:
                irs = r_[irs, sample]
            else:
                irs = r_[irs, -1 * sample]

        repeated_irs = irs
        for i in range(reps + 1):
            repeated_irs = r_[repeated_irs, irs]

        self.signal = repeated_irs

    def filterSignal(self, cutoff, order, type):
        """ Filters the current signal with a specified filter.

        :param cutoff:
            The cut off frequency of the filter.
        :type cutoff:
            float
        :param order:
            The order of the filter to use.
        :type order:
            int
        :param type:
            The type of filter to use, either "low" or "high", for low pass
            filter, or high pass filter.
        :type type:
            str
        """
        self.logger.debug("Entering filterSignal (%s, %s, %s)" % (cutoff, order, type))

        # Get signal parameters
        sample_rate = float(self.parameters["sample rate"])

        if type == "high":
            [b, a] = butter(order, cutoff / (sample_rate / 2), btype=type)
            self.signal = lfilter(b, a, self.signal)
        #    b = firwin(251, cutoff=[50, 200], nyq=(sample_rate / 2))
        #    a = 1
        #    self.signal = lfilter(b, a, self.signal)

        #    [b, a] = iirdesign(wp=cutoff / (sample_rate / 2), ws = 50 / (sample_rate / 2), gpass=1, gstop=12, ftype="butter")
        else:

            [b, a] = butter(order, cutoff / (sample_rate / 2), btype=type)

            self.signal = lfilter(b, a, self.signal)

    def padSignal(self):
        """ Pad signal with an impulse at the front of the signal, followed by
            a delay, also adds a delay to the end of the signal.
        """
        self.logger.debug("Entering padSignal")

        # Get signal parameters
        impulse_delay = float(self.parameters["impulse delay"])
        signal_padding = float(self.parameters["signal padding"])
        sample_rate = float(self.parameters["sample rate"])

        impulse_delay_samples = zeros(int(impulse_delay * sample_rate))
        signal_padding_samples = zeros(int(signal_padding * sample_rate))

        self.signal = r_[0, 1, impulse_delay_samples, self.signal,
                         signal_padding_samples]

if __name__ == "__main__":
    """ A simple example showing the use of the Signal Generator """
    import pylab as py

    logger = logging.getLogger("Alpha")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    parameters = {
        "fft size": 2 ** 18,
        "hpf cutoff": 500,
        "hpf enabled": 0,
        "hpf order": 1,
        "impulse delay": 0.02,
        "lower frequency": 0,
        "lpf cutoff": 3500,
        "lpf enabled": 0,
        "lpf order": 4,
        "mls reps": 1,
        "mls taps": 14,
        "pad signal": 0,
        "sample rate": 44100,
        "signal length": 100 * 10 ** -3,
        "signal padding": 200 * 10 ** -3,
        "signal reps": 0,
        "signal type": "low pass swept sine",
        "upper frequency": 6400,
    }

    signal_gen = SignalGenerator(parameters)
    signal = signal_gen.signal
    print len(signal)
    py.plot(signal)
    py.show()

