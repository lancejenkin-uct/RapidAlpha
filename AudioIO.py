#!/usr/bin/env python
""" Provides audio interface to playback and capture audio.

AudioIO is a class that utilizes PortAudio, in order to make a platform
independent audio interface.  It provides means to playback and record audio
signals at the same time.  It also provides a class to retrieve audio device
information
"""

from numpy import *
from ctypes import *
from ctypes.util import *
import logging

__author__ = "Lance Jenkin"
__email__ = "lancejenkin@gmail.com"


# Structures used to communicate with PortAudio C Library
class _PaData(Structure):
    _fields_ = [("sample_index", c_ulong),
                ("left_signal_length", c_ulong),
                ("right_signal_length", c_ulong),
                ("left_channel_signal", c_void_p),
                ("right_channel_signal", c_void_p),
                ("num_samples_read", c_ulong),
                ("left_channel_buffer", c_void_p),
                ("right_channel_buffer", c_void_p),
                ("first_run", c_double)]


class _PaDeviceInfo(Structure):
    _fields_ = [("structVersion", c_int),
                ("name", c_char_p),
                ("hostApi", c_int),
                ("maxInputChannels", c_int),
                ("maxOutputChannels", c_int),
                ("defaultLowInputLatency", c_double),
                ("defaultLowOutputLatency", c_double),
                ("defaultHighInputLatency", c_double),
                ("defaultHighOutputLatency", c_double),
                ("defaultSampleRate", c_double)]


class _PaStreamParameters(Structure):
    _fields_ = [("device", c_int),
                ("channelCount", c_int),
                ("sampleFormat", c_ulong),
                ("suggestedLatency", c_double),
                ("hostApiSpecificStreamInfo", c_void_p)]


class _PaStreamCallbackTimeInfo(Structure):
    _fields_ = [("inputBufferAdcTime", c_double),
                ("currentTime", c_double),
                ("outputBufferDacTime", c_double)]


class AudioDevice(object):

    def __init__(self, name, index, input_channels, output_channels):
        """ Constructor for Audio Device class

        :param name:
            The name of the audio device.
        :type name:
            str
        :param index:
            The index of the audio device, as referenced by PortAudio
        :type index:
            int
        :param input_channels:
            he number of input channels provided by the device
        :type input_channels:
            int
        :param output_channels:
            The number of output channels provided by the device
        :type output_channels:
            int
        :param name:
            The name of the audio device.
        :type name:
            str
        """
        self.name = name
        self.index = index
        self.input_channels = input_channels
        self.output_channels = output_channels


class AudioIO(object):
    # Number of frames stored in the buffer
    _FRAMES_PER_BUFFER = 4096 / 4

    def __init__(self, sample_rate=44100):
        """ Default Constructor

        :param sample_rate:
            The sample rate to use for playback and capture. Defaults to 
            44.1 kHz.
        :type sample_rate:
            An int or float, gets converted to float.
        """
        self.logger = logging.getLogger("Alpha")
        self.logger.debug("Creating AudioIO Object")

        self.sample_rate = float(sample_rate)

        self._loadPortAudio()

    def __del__(self):
        """Deconstructor to terminate connection to PortAudio """
        self.logger.debug("Deleting AudioIO Object")

        self.port_audio.Pa_Terminate()

    def _loadPortAudio(self):
        """ Load the PortAudio library

        Tries to locate the PortAudio library, and load it.  If it can't be
        found, or initialiazed, an Exception will be raised.
        """
        self.logger.debug("Entering _loadPortAudio")

        port_audio = find_library("portaudio")

        if port_audio is None:
            port_audio = "./PortAudio.dll"

        try:
            self.port_audio = CDLL(port_audio)
        except Exception as inst:
            raise Exception("Port Audio not load: %s" % (inst))

        error = self.port_audio.Pa_Initialize()

        if error != 0:
            error_text = self.port_audio.Pa_GetErrorText(error)
            raise Exception("Error initializing Port Audio: %s" % (error_text))

    def getAudioDevices(self):
        """ Method to return list of audio devices present.

        Queries PortAudio to return a list of audio devices present in the
        system, along with the audio device name, it also returns the number
        of input and output devices, and its specific device index.

        :returns:
            array : An array of AudioDevice objects, containing the audio
            devices available on in the system.
        """
        self.logger.debug("Entering getAudioDevices")

        audio_devices = []

        device_count = self.port_audio.Pa_GetDeviceCount()
        for device_index in range(device_count):
            # Ask Port Audio for the device info with specified device_index
            self.port_audio.Pa_GetDeviceInfo.restype = POINTER(_PaDeviceInfo)
            pa_device_info = self.port_audio.Pa_GetDeviceInfo(device_index)

            name = pa_device_info.contents.name
            input_channels = pa_device_info.contents.maxInputChannels
            output_channels = pa_device_info.contents.maxOutputChannels

            audio_device = AudioDevice(name, device_index, input_channels,
                                       output_channels)

            audio_devices.append(audio_device)

        return audio_devices

    def setInputDevice(self, device_index):
        """ Sets the input device to capture the signals.

        :param device_index:
            The index of the device to use as an input device.
        :type device_index:
            int
        """
        self.logger.debug("Entering setInputDevice (%s)" % (device_index))

        self.input_device = device_index

    def setOutputDevice(self, device_index):
        """ Sets the output device to playback the signals.

        :param device_index:
            The index of the device to use as an output device.
        :type device_index:
            int
        """
        self.logger.debug("Entering setOutputDevice (%s)" % (device_index))

        self.output_device = device_index

    def playbackAndRecord(self, left_channel_signal, right_channel_signal):
        """ Playback the given signal and record the response.

        Plays back the specified left and right channel signals.  While playing
        the signal, record the response.  When the signal with the longest
        length has finished playing, return the recorded response.

        This is a blocking function, and only returns when the signals have
        completed playing.

        :param left_channel_signal:
            The signal to playback through the left channel.
        :type left_channel_signal:
            Array representing the signal, containing float values between -1
            and +1.
        :param right_channel_signal:
            The signal to playback through the right channel.
        :type right_channel_signal:
            Array representing the signal, containing float values between -1
            and +1.

        :returns:
            tuple : a two-tuple, with the first element an array containing the
            the left channel recorded response, and the second element
            containing the right channel response.
        """
        self.logger.debug("Entering playbackAndRecord")

        # Open the stream
        self._openStream(left_channel_signal, right_channel_signal)

        # Begin playback
        error = self.port_audio.Pa_StartStream(self.stream)
        if error < 0:
            error_text = self.port_audio.Pa_GetErrorText(error)
            raise Exception("Couldn't start stream: %s" % (error_text))

        # Wait until completed playback
        while self.port_audio.Pa_IsStreamActive(self.stream) == 1:
            self.port_audio.Pa_Sleep(500)

        # Ensure the stream has stopped
        self.port_audio.Pa_StopStream(self.stream)

        # Retrieve the recorded response
        left_channel_buffer = cast(self.data.left_channel_buffer, POINTER(c_float))

        right_channel_buffer = cast(self.data.right_channel_buffer, POINTER(c_float))

        # Cast to Python array
        left_channel_data = []
        right_channel_data = []

        samples_read = self.data.num_samples_read
        for signal_index in range(samples_read):
            left_channel_data.append(left_channel_buffer[signal_index])
            right_channel_data.append(right_channel_buffer[signal_index])

        return (left_channel_data, right_channel_data)

    def _openStream(self, left_channel_signal, right_channel_signal):
        """ Open a PortAudio stream to playback the specified signals, and
            record the response.

        Creates a new stream, and passes the stream information to PortAudio.

        Raises an Exception if the stream could not open.

        :param left_channel_signal:
            The signal to playback through the left channel.
        :type left_channel_signal:
            Array representing the signal, containing float values between -1
            and +1.
        :param right_channel_signal:
            The signal to playback through the right channel.
        """
        self.logger.debug("Entering _openStream")

        # Set the input / output paramaters
        input_paramaters = _PaStreamParameters()
        output_paramaters = _PaStreamParameters()

        # Set input device
        self.port_audio.Pa_GetDeviceInfo.restype = POINTER(_PaDeviceInfo)
        device_info = self.port_audio.Pa_GetDeviceInfo(self.input_device)
        input_paramaters.device = self.input_device
        input_paramaters.channelCount = 2
        input_paramaters.sampleFormat = c_ulong(1)
        latency = device_info.contents.defaultHighInputLatency
        input_paramaters.suggestedLatency = latency
        input_paramaters.hostApiSpecificStreamInfo = c_void_p()

        # Set output device
        self.port_audio.Pa_GetDeviceInfo.restype = POINTER(_PaDeviceInfo)
        device_info = self.port_audio.Pa_GetDeviceInfo(self.output_device)
        output_paramaters.device = self.output_device
        output_paramaters.channelCount = 2
        output_paramaters.sampleFormat = c_ulong(1)
        latency = device_info.contents.defaultHighOutputLatency
        output_paramaters.suggestedLatency = latency
        output_paramaters.hostApiSpecificStreamInfo = c_void_p()

        # Reserve space for the signal
        left_signal_length = len(left_channel_signal)
        right_signal_length = len(right_channel_signal)

        self.left_channel_signal_memory = (c_float * left_signal_length)()
        self.right_channel_signal_memory = (c_float * right_signal_length)()

        # Copy the signal data to the memory
        for signal_index in range(left_signal_length):
            self.left_channel_signal_memory[signal_index] = (left_channel_signal[signal_index])

        for signal_index in range(right_signal_length):
            self.right_channel_signal_memory[signal_index] = (right_channel_signal[signal_index])

        # Reserve memory to record the response
        max_signal_length = max(left_signal_length, right_signal_length)
        self.left_channel_buffer = (c_float * max_signal_length)()
        self.right_channel_buffer = (c_float * max_signal_length)()

        # Create the Data structure
        left_signal_address = addressof(self.left_channel_signal_memory)
        right_signal_address = addressof(self.right_channel_signal_memory)
        left_buffer_address = addressof(self.left_channel_buffer)
        right_buffer_address = addressof(self.right_channel_buffer)

        self.data = _PaData()
        self.data.first_run = True
        self.data.num_samples_read = 0
        self.data.sample_index = 0
        self.data.left_signal_length = left_signal_length
        self.data.right_signal_length = right_signal_length
        self.data.left_channel_signal = left_signal_address
        self.data.right_channel_signal = right_signal_address
        self.data.left_channel_buffer = left_buffer_address
        self.data.right_channel_buffer = right_buffer_address

        # Open the stream
        PACALLBACK = CFUNCTYPE(c_int, c_void_p, c_void_p, c_ulong,
                             POINTER(_PaStreamCallbackTimeInfo),
                             c_ulong, c_void_p)

        self.pa_callback_cfunc = PACALLBACK(self.pa_callback)

        self.stream = c_void_p()
        pa_openstream = self.port_audio.Pa_OpenStream
        pa_openstream.argtypes = [POINTER(c_void_p),
                                  POINTER(_PaStreamParameters),
                                  POINTER(_PaStreamParameters), c_double,
                                  c_long, c_long, c_void_p, POINTER(_PaData)]

        pa_openstream.restype = c_int
        error = pa_openstream(pointer(self.stream), pointer(input_paramaters),
                      pointer(output_paramaters), self.sample_rate,
                      self._FRAMES_PER_BUFFER, 1, self.pa_callback_cfunc,
                      pointer(self.data))

        if error < 0:
            error_text = self.port_audio.Pa_GetErrorText(error)
            raise Exception("Couldn't open stream: %s" % (error_text))

    @staticmethod
    def pa_callback(input_buffer, output_buffer, frames_per_buffer, time_info,
                  status_flags, user_data):
        """PortAudio callback function called to fill PortAudio's buffer.

        It provides PortAudio with signal to playback, as well as store data
        captured by PortAudio.

        :param input_buffer:
            A pointer to the input (microphone) buffer, which contains the
            frames read from the input channel.
        :type input_buffer:
            ctype pointer
        :param output_buffer:
            A pointer to the output buffer, which contains the signal to be
            played back.
        :type output_buffer:
            ctype pointer
        :param frames_per_buffer:
            The number of frames PortAudio expects to fill up the output
            buffer, as well as the number of frame it has provided in the
            input buffer.
        :type frames_per_buffer:
            int
        :param time_info:
            A structure containing the timing information.
        :type time_info:
            ctype structure
        :param status_flags:
            The current status of the PortAudio connection.
        :type status_flags:
            long
        :param user_data:
            A pointer to the user data structure, to be used to fill buffers,
            both input and output.
        :type user_data:
            ctype pointer

        :returns:
            Status indicating if Port Audio needs to continue with the stream
            0 means continue, and 1 means stop.
        """
        # Cast variables so that they can be used
        data = cast(user_data, POINTER(_PaData))
        out_ptr = cast(output_buffer, POINTER(c_float))
        in_ptr = cast(input_buffer, POINTER(c_float))

        # Determine the maximum signal length, as left channel's signal may be
        # different to the right channel.
        left_signal_length = data.contents.left_signal_length
        right_signal_length = data.contents.right_signal_length
        max_length = max([left_signal_length, right_signal_length])

        # Determine how many samples to read from the microphone, bearing in
        # mind that we only want as many samples as the longest signal
        # provided.
        samples_read = data.contents.num_samples_read
        read_length = min([max_length - samples_read, frames_per_buffer])
        
        # Cast the pointers into usable c_float pointers
        left_chan_buf = data.contents.left_channel_buffer
        right_chan_buf = data.contents.right_channel_buffer
        
        left_channel_buffer = cast(left_chan_buf, POINTER(c_float))
        right_channel_buffer = cast(right_chan_buf, POINTER(c_float))

        if data.contents.first_run:
            # Ignore the first run's buffer
            data.contents.first_run = False
        else:
            # Read the information from the microphone, keeping in mind that
            # the data is interleaved, with the first c_float the left
            # channel, the next c_float the right channel, and alternating
            # from there.
            for i in range(read_length):
                index = 2 * i
                samples_read = data.contents.num_samples_read

                buf_index = i + samples_read
                left_channel_buffer[buf_index] = c_float(in_ptr[index])
                right_channel_buffer[buf_index] = c_float(in_ptr[index + 1])

        data.contents.num_samples_read += read_length

        # Play Signal
        left_chan_ptr = data.contents.left_channel_signal
        right_chan_ptr = data.contents.right_channel_signal
        
        left_channel_signal = cast(left_chan_ptr, POINTER(c_float))
        right_channel_signal = cast(right_chan_ptr, POINTER(c_float))

        for i in range(frames_per_buffer):
            # The output pointer is interleaved
            index = 2 * i
            sample_index = data.contents.sample_index
            # Left Channel
            left_signal_length = data.contents.left_signal_length
            if (i + data.contents.sample_index < left_signal_length):
                out_ptr[index] = c_float(left_channel_signal[i + sample_index])
            else:
                out_ptr[index] = c_float(0)

            # Then Right
            right_signal_length = data.contents.right_signal_length
            if (i + data.contents.sample_index < right_signal_length):
                out_ptr[index + 1] = c_float(right_channel_signal[i + sample_index ])
            else:
                out_ptr[index + 1] = c_float(0)

        # Only update the sample index, if it is less than the maximum signal
        # length, else we don't care as there is no more signal left
        if data.contents.sample_index < max_length:
            data.contents.sample_index += frames_per_buffer

        # Return 0, to continue, if we still need to read more samples
        return int(data.contents.num_samples_read >= max_length)


if __name__ == "__main__":
    """ A simple example showing the use of the Audio Interface """
    logger = logging.getLogger("Alpha")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    sample_rate = 44100.0

    audio = AudioIO(sample_rate)

    audio_devices = audio.getAudioDevices()

    for audio_device in audio_devices:
        device_index = audio_device.index
        device_name = audio_device.name
        input_channels = audio_device.input_channels
        output_channels = audio_device.output_channels

        print "%s: %s" % (device_index, device_name)
        print "Input Channels: %s" % (input_channels)
        print "Output Channels: %s" % (output_channels)

        if input_channels > 0:
            audio.setInputDevice(device_index)
        if output_channels > 0:
            audio.setOutputDevice(device_index)

    t = arange(0, 1, 1 / sample_rate)
    s = sin(2 * pi * 440 * t)

    (left, right) = audio.playbackAndRecord(s, s)

    from pylab import *
    subplot(211)
    plot(left)
    title("Left Channel")
    subplot(212)
    plot(right)
    title("Right Channel")
    show()

    pass
