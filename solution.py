""" Downsamples an audio signal by a factor of two. 

Contains the following functions:

    * decimate_by_2 - removes every other sample from the input array
    * downsample_by_2 - applies a low-pass filter and decimation to an input array
    * convolve_arrays - implementation of the convolution operation
"""

KAISER_CONVOLUTION_FILTER = [-0.01452123, -0.0155227 , 0.01667252, 0.01800633, -0.01957209,
                             -0.0214361 , 0.02369253, 0.02647989, -0.03001054, -0.03462755,
                              0.04092347, 0.05001757, -0.06430831, -0.09003163, 0.15005272,
                              0.45015816, 0.45015816, 0.15005272, -0.09003163, -0.06430831,
                              0.05001757, 0.04092347, -0.03462755, -0.03001054, 0.02647989,
                              0.02369253, -0.0214361 , -0.01957209, 0.01800633, 0.01667252,
                             -0.0155227 , -0.01452123] 

def decimate_by_2(signal):
    """ Removes every other element from an array

    Parameters
    -----------
    signal: list
        Audio signal, time-domain

    Returns
    -----------
    list
        Decimated Signal
    """
    if not signal:
        raise ValueError('Audio signal not provided.')

    return signal[::2]

def downsample_by_2(signal):
    """ Returns a smoother, downsampled version of the input audio signal

    Parameters
    -----------
    signal: list
        Audio signal

    Returns
    -----------
    list
        Resampled audio signal passed through a low-pass filter and decimated by 2
    """
    if not signal:
        raise ValueError('Audio signal not provided.')

    filtered_signal = convolve_arrays(signal, KAISER_CONVOLUTION_FILTER)
    decimated_filtered_signal = decimate_by_2(filtered_signal)

    return decimated_filtered_signal

def convolve_arrays(signal, conv_filter):
    """ Performs the discrete, linear convolution operation on two arrays (Finite impulse response)

    Resources used: 
        * http://digitalsoundandmusic.com/7-3-1-convolution-and-time-domain-filtering/
        * https://en.wikipedia.org/wiki/Convolution

    Parameters
    -----------
    signal: list
        Audio signal, though the operation works for any array

    conv_filter: list
        Convolution kernel, though the operation works for any array

    Returns
    -----------
    list with length = max(length of both input arrays)
        Represents filtered audio samples
    """
    output_array_length = max(len(signal), len(conv_filter))
    # The full convolved output array could have length (len(signal) + len(conv_filter) - 1)
    # To have the output array have length N = max(len(signal), len(conv_filter)), we need to find
    # the middle N elements. The 'centering_factor' provides the needed shift
    centering_factor = (min(len(signal), len(conv_filter)) - 1) // 2

    output = [0] * output_array_length

    for n in range(output_array_length):

        running_sum = 0

        for k in range(len(conv_filter)):
            signal_idx = n - k + centering_factor

            if (signal_idx) < 0:
                break
            elif (signal_idx) > len(signal) - 1:
                continue

            running_sum += conv_filter[k] * signal[signal_idx]

        output[n] = running_sum

    return output
