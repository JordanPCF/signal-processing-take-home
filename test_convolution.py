""" Unit test suite to check my implementation of the convolution function

Designed for use with pytest. Contains the following tests:

    * Check answer against numpy's convolution function for input arrays with the same length
    * Check commutative property
    * Check answer against numpy's convolution function for input arrays with differing lengths
        - Case: Longer array has an odd number of elements. Shorter array has odd number elements.
        - Case: '     '    '        odd     '    '        '      '       '    even    '     '
        - Case: '     '    '        even    '    '        '      '       '    odd     '     '
        - Case: '     '    '        even    '    '        '      '       '    even    '     '
        These cases test that the boundary effects are the same as those from the numpy function

Also contains the function:
    * plot_original_and_downsampled - Compares noisy sinusoidal signal before and after downsampling
"""

import numpy as np
import matplotlib.pyplot as plt

from solution import downsample_by_2, convolve_arrays

class TestConvolutionFunction:
    def test_arrays_same_length(self):
        """
        Test that convolve_arrays has the same output as numpy's convolve() given two arrays of
        equal length
        """
        signal = [2, 1, -1, 3]
        conv_filter = [4, 3, 2, 1]

        my_convolved_output = convolve_arrays(signal, conv_filter)
        numpy_solution = np.convolve(signal, conv_filter, 'same').tolist()

        assert my_convolved_output == numpy_solution

    def test_commutative_property(self):
        """
        Test that the result is indifferent to the order the arrays are given.
        This also verifies the function works in both cases where the filter array is shorter or
        longer than the signal array.
        """
        longer = [2, 1, -1, 3, 10]
        shorter = [4, 3, 2]

        assert convolve_arrays(longer, shorter) == convolve_arrays(shorter, longer)

    def test_boundary_effects_longer_odd_shorter_odd(self):
        """
        Test that convolve_arrays has the same output as numpy's convolve() given two 
        arrays of differing length, where both input arrays have an odd number of elements.
        """
        signal = [2, 1, -1, 3, 3, 6, 7, -3, 1, 12, 9]
        conv_filter = [4, 3, 2, 1, 6]

        my_convolved_output = convolve_arrays(signal, conv_filter)
        numpy_solution = np.convolve(signal, conv_filter, 'same').tolist()

        assert my_convolved_output == numpy_solution

    def test_boundary_effects_longer_odd_shorter_even(self):
        """
        Test that convolve_arrays has the same output as numpy's convolve() given two 
        arrays of differing length, where the longer input array has an odd number of elements, 
        and the shorter array an even number of elements. 
        """
        signal = [2, 1, -1, 3, 3, 6, 7, -3, 1, 12, 9]
        conv_filter = [4, 3, 2, 1]

        my_convolved_output = convolve_arrays(signal, conv_filter)
        numpy_solution = np.convolve(signal, conv_filter, 'same').tolist()

        assert my_convolved_output == numpy_solution

    def test_boundary_effects_longer_even_shorter_odd(self):
        """
        Test that convolve_arrays has the same output as numpy's convolve() given two 
        arrays of differing length, where the longer input array has an even number of elements, 
        and the shorter array an odd number of elements. 
        """
        signal = [2, 1, -1, 3, 3, 6, 7, -3, 1, 12, 9, 41]
        conv_filter = [4, 3, 2, 1, 6]

        my_convolved_output = convolve_arrays(signal, conv_filter)
        numpy_solution = np.convolve(signal, conv_filter, 'same').tolist()

        assert my_convolved_output == numpy_solution

    def test_boundary_effects_longer_even_shorter_even(self):
        """
        Test that convolve_arrays has the same output as numpy's convolve() given two 
        arrays of differing length, where both input arrays have an even number of elements.
        """
        signal = [2, 1, -1, 3, 3, 6, 7, -3, 1, 12, 9, 41]
        conv_filter = [4, 3, 2, 1, 6, 10]

        my_convolved_output = convolve_arrays(signal, conv_filter)
        numpy_solution = np.convolve(signal, conv_filter, 'same').tolist()

        assert my_convolved_output == numpy_solution

def plot_original_and_downsampled():
    """
    A 'visual test' to see the efficacy of the function downsample_by_2

    A noisy sinusoid is plotted before and after the application of the low-pass filter and
    decimation.
    """

    t = np.linspace(1, 100, 1000)
    t_decimated = np.linspace(1, 100, 500)

    ideal_signal = np.sin(t/(2*np.pi))
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = ideal_signal + noise
    downsampled_signal = downsample_by_2(noisy_signal.tolist())

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True)

    ax1.plot(t, noisy_signal)
    ax1.set_title('Original Signal')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')

    ax2.plot(t_decimated, downsampled_signal)
    ax2.set_title('Downsampled Signal')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')

    plt.show()