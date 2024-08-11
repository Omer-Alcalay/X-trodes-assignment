import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import neo

class SignalProcessor:   #creat a class Signal procesor. with 4 function in it.
    def __init__(self, file_path, num_channels=8, sample_rate=4000, num_ADC_bits=15, voltage_resolution=4.12e-7):
        self.file_path = file_path
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.num_ADC_bits = num_ADC_bits
        self.voltage_resolution = voltage_resolution
        self.data = None
        self.df = None

    def load_data(self):
        # Load DT8 file
        reader = neo.RawBinarySignalIO(filename=self.file_path, dtype='int16',
                                       nb_channel=self.num_channels,
                                       sampling_rate=self.sample_rate)
        data = reader.read_segment().analogsignals[0]

        # Convert to numpy array and transpose to match our expected format
        self.data = data.magnitude.T
        # Convert to voltage values from the ADC [as defined in the task]
        self.data = np.multiply(self.voltage_resolution,
                                (data - np.float_power(2, self.num_ADC_bits - 1)))

    def pd_to_dataframe(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert to DataFrame [making charts with column name 'chanel_1' etc]
        self.df = pd.DataFrame(self.data, columns=[f'Channel_{i + 1}' for i in range(self.num_channels)])

        # Add time column to the first column
        time = np.arange(len(self.df)) / self.sample_rate # each sample signal associated with time of occurrence.
        self.df.insert(0, 'Time', time)

    def zero_phase_bandpass_filter(self, lowcut, highcut, order=6):
        if self.df is None:
            raise ValueError("DataFrame not created. Call to_dataframe() first.")

        nyq = 0.5 * self.sample_rate #nyquist frequency
        low = lowcut / nyq #normalize the cutoff frequency
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band') #creating the bandpass filter (with high and low cuts)

        for col in self.df.columns[1:]:  # Skip the Time column, and lop all the chanels.
            self.df[f'{col}_filtered'] = signal.filtfilt(b, a, self.df[col])

    def plot_data(self, channels=None, filtered=False):
        if self.df is None:
            raise ValueError("DataFrame not created. Call to_dataframe() first.")

        if channels is None:
            channels = range(1, self.num_channels + 1)

        fig, axs = plt.subplots(len(channels), 1, figsize=(12, 4 * len(channels)), sharex=True) #creating subplot for each chanel

        for i, channel in enumerate(channels):
            col = f'Channel_{channel}'
            axs[i].plot(self.df['Time'], self.df[col], label=f'Filtered Channel {channel}')
            axs[i].set_ylabel('Voltage (V)')
            axs[i].set_title(f'Filtered Channel {channel}')
            axs[i].legend()

        axs[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

# Usage example:
if __name__ == "__main__":
    processor = SignalProcessor(r'C:\Users\Owner\OneDrive\Desktop\Intervies Project X-trodes\NEUR0000 (2).dt8')
    processor.load_data()
    processor.pd_to_dataframe()
    processor.zero_phase_bandpass_filter(lowcut=100, highcut=300)  # Apply 10-500 Hz bandpass filter
    processor.plot_data(channels=[1,2], filtered=True)