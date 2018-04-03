'''
created by Jan Behrenbeck 02.02.2018
'''
import os
import csv
import numpy as np
from scipy.signal import firwin
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import number

class Encoder():

    def __init__(self,path_to_neucube_folder,number_of_input_channels,number_of_samples,signal_duration,signal_timestep,subject):

        self.path_to_neucube_folder         = path_to_neucube_folder                # 'C:\.....'
        self.number_of_input_channels       = number_of_input_channels              #
        self.number_of_samples              = number_of_samples                     #
        self.signal_duration                = signal_duration                       # in ms
        self.signal_timestep                = signal_timestep                       # in ms
        print("Stage 1:")
        print("Loading input data...")
        self.input_eeg_data                 = self.load_input_data(subject)                # list of eeg samples
        print("Input data loaded! Ready for encoding.")
        self.SSA_data                       = []                                    # list of SSAs
        self.reconstructed_data             = []                                    # list of reconstructed samples
        self.error_data                     = []                                    # list of error metrics

        # parameters for BSA encoding
        self.filter = firwin(7,0.1)
        self.threshold_BSA = 0.679
        #self.filter = firwin(20,0.1)        # Nuntalid 2011
        #self.threshold_BSA = 0.955
        # for 20-tap
        #self.filter = [0.0105401844532279  ,  0.0210803689064559 ,   0.0342555994729908  ,  0.0461133069828722  ,  0.0579710144927536  ,  0.0685111989459816  ,  0.0777338603425560  ,  0.0843214756258235 ,   0.0856389986824769,   0.0843214756258235 ,   0.0803689064558630   , 0.0750988142292490 ,   0.0685111989459816  ,  0.0592885375494071  ,  0.0487483530961792  ,  0.0382081686429513 ,   0.0276679841897233  ,  0.0171277997364954 ,   0.00922266139657444 ,   0.00527009222661397]
        #self.threshold_BSA = 0.955

        #parameters for TD encoding
        self.threshold_TD = 0.05

        #parameters for mod_TD encoding
        self.threshold_mod_TD = 0.05

    def load_input_data(self,subject):
        #This method loads all samples in a list. Input_eeg_data = [[sample_1],[sample_2],[sample_3] ... ]
        input_eeg_data = []
        for i in range(self.number_of_samples):
            sample = self.load_sample(i+1,subject)
            input_eeg_data.append(sample)
        return input_eeg_data

    def load_sample(self,sample_index,subject):
        #Loads EEG signals from csv files and creates a list of channels, each channel being a list of values from the respective signal. sample = [[channel_1],[channel_2],[channel_3] ... ]
        path_to_file = os.path.join(self.path_to_neucube_folder,'input_stage_1',subject,'sam_'+str(sample_index)+'.csv')
        #print('opening file ' + path_to_file)
        datapoints = np.floor_divide(self.signal_duration,self.signal_timestep)
        n_inp = self.number_of_input_channels

        csv_file = open(path_to_file, 'r')
        reader = csv.reader(csv_file, delimiter=',')

        sample = []
        for line in reader:
            channel = []
            for i in range(int(datapoints)):
                channel.append(float(line[i]))
            sample.append(channel)
        return sample

    def info(self):
        #This method displays all internal information of the filter.
        print("")
        print("Information about the Encoder!")
        print('Path to input folder:          ' +self.path_to_input_signal_folder)
        print('Number of input channels:      ' +str(self.number_of_input_channels   ))
        print('Number of samples:             ' +str(self.number_of_samples          ))
        print('Duration of the signal in ms:  ' +str(self.signal_duration            ))
        print('Timestep of the signal in ms:  ' +str(self.signal_timestep            ))
        print("")
        print("Parameters for BSA encoding:")
        print('Filter:                        ' +str(self.filter           ))
        print('BSA-Threshold:                 ' +str(self.threshold_BSA    ))
        print("")
        print("Parameters for TD encoding:")
        print('TD-Threshold:                  ' +str(self.threshold_TD     ))
        print("")
        print("Parameters for mod_TD encoding:")
        print('mod_TD-Threshold:              ' +str(self.threshold_mod_TD ))

    def set_BSA_parameters(self,win_size,cutoff_freq,thresh):
        self.filter = firwin(win_size,cutoff_freq)
        self.threshold_BSA = thresh

    def set_TD_parameters(self,thresh):
        self.threshold_TD = thresh

    def set_mod_TD_parameters(self,thresh):
        self.threshold_mod_TD = thresh

    def encode(self, method,subject):
        #This method encodes the input samples into spike trains based on the method of choice.
        #return: list of lists of spike source arrays (list) containing the spike times.
        print ("Encoding the signal using " + method + "...")
        data = self.load_input_data(subject)
        SSA_data = []
        for s in range(self.number_of_samples):
            sample = data[s]
            SSA_sample = []
            for c in range(self.number_of_input_channels):
                signal = sample[c]
                if method == 'BSA':
                    SSA = self.use_BSA_encoding(signal)
                elif method == 'TD':
                    SSA = self.use_TD_encoding(signal)
                elif method == 'mod_TD':
                    SSA = self.use_mod_TD_encoding(signal)
                else:
                    print('Error! The method you chose is not implemented. Please choose among BSA, TD, and mod_TD.')
                    break
                SSA_sample.append(SSA)
            SSA_data.append(SSA_sample)
        self.SSA_data = SSA_data
        print ("Signal encoded! Ready for stage 2 or decoding.")
        return SSA_data

    def encode_sample(self, sample, method):
        #This method encodes one single sample into spike trains based on the method of choice.
        #return: lists of spike source arrays (list) containing the spike times.
        print ("Encoding the sample using " + method + "...")
        SSA_sample = []
        for c in range(len(sample)):
            signal = sample[c]
            if method == 'BSA':
                SSA = self.use_BSA_encoding(signal)
            elif method == 'TD':
                SSA = self.use_TD_encoding(signal)
            elif method == 'mod_TD':
                SSA = self.use_mod_TD_encoding(signal)
            else:
                print('Error! The method you chose is not implemented. Please choose among BSA, TD, and mod_TD.')
                break
            SSA_sample.append(SSA)
        print ("Sample encoded!")
        return SSA_sample

    def use_BSA_encoding(self,signal):
        #This method encodes the input samples into spike trains based on the Ben's Spiker algorithm.
        #return: spike source array (list) containing the spike times.
        SSA = []
        for i in range(1,len(signal)):
            error1 = 0
            error2 = 0
            for j in range(1,len(self.filter)):
                if i+j-1 <= len(signal):
                    error1 += abs(signal[i+j-2]-self.filter[j])
                    error2 += abs(signal[i+j-2])
            if error1 <= (error2-self.threshold_BSA):
                SSA.append(i*self.signal_timestep)
                for j in range(1,len(self.filter)):
                    if i+j-1 <= len(signal):
                        signal[i+j-2] -= self.filter[j]
        return SSA

    def use_TD_encoding(self, signal):
        #This method encodes the input samples into spike trains based on the Temporal Difference algorithm.
        #return: spike source array (list) containing the spike times. The sign of each element defines an excitatory (+) or inhibitory spike (-). The value defines the time of the spike.
        # timepoints = [ abs(x) for x in SSA]
        # spikes = [ sign(x) for x in SSA]
        SSA = []
        last_value = 0.0
        for i in range(len(signal)):
            diff = signal[i]-last_value
            if abs(diff)>=self.threshold_TD:
                SSA.append(round(i*self.signal_timestep*np.sign(diff),3)) # round to eliminate error from floating point operation
            last_value = signal[i]
        return SSA

    def use_mod_TD_encoding(self, signal):
        #This method encodes the input samples into spike trains based on the Temporal Difference algorithm.
        #return: spike source array (list) containing the spike times. The sign of each element defines an excitatory (+) or inhibitory spike (-). The value defines the time of the spike.
        # timepoints = [ abs(x) for x in SSA]
        # spikes = [ sign(x) for x in SSA]
        SSA = []
        last_value = 0.0
        res = 0.0
        for i in range(len(signal)):
            res += signal[i]-last_value
            if abs(res)>=self.threshold_mod_TD:
                SSA.append(round(i*self.signal_timestep*np.sign(res),3)) # round to eliminate error from floating point operation
                res = np.sign(res)*(abs(res)-self.threshold_mod_TD)
            last_value = signal[i]
        return SSA

    def decode(self, method):
        #This method decodes the spike trains into analog signals based on the method of choice.
        #return: list of lists of reconstructed signals (list) containing the analog values.
        print ("Decoding the signal using " + method + "...")
        SSA_data = self.SSA_data
        rec_sig_data = []
        for s in range(self.number_of_samples):
            SSA_sample = SSA_data[s]
            rec_sig_sample = []
            for c in range(self.number_of_input_channels):
                SSA = SSA_sample[c]
                if method == 'BSA':
                    rec_sig = self.use_BSA_decoding(SSA)
                elif method == 'TD':
                    rec_sig = self.use_TD_decoding(SSA)
                elif method == 'mod_TD':
                    rec_sig = self.use_mod_TD_decoding(SSA)
                else:
                    print('Error! The method you chose is not implemented. Please choose among BSA, TD, and mod_TD.')
                    break
                rec_sig_sample.append(rec_sig)
            rec_sig_data.append(rec_sig_sample)
        self.reconstructed_data = rec_sig_data
        print ("Spike Train decoded! Ready for error calculation.")
        return rec_sig_data

    def use_BSA_decoding(self,SSA):
        #This method decodes a spike train and reconstructs the original signal based on the Ben's Spiker algorithm.
        #result: list with reconstructed signal values.
        datapoints = int(np.floor_divide(self.signal_duration,self.signal_timestep))
        rec_sig = np.zeros(datapoints)
        for i in SSA:
            for j in range(len(self.filter)):
                if int(round(i/self.signal_timestep))-1+j < datapoints:
                    rec_sig[(int(round(i/self.signal_timestep)))-1+j] += self.filter[j]
        return rec_sig


    def use_TD_decoding(self,SSA):
        #This method decodes a spike train and reconstructs the original signal based on the TD algorithm.
        #result: list with reconstructed signal values.
        datapoints = np.floor_divide(self.signal_duration,self.signal_timestep)
        rec_sig = 0.0*np.ones(datapoints)
        for i in SSA:
            rec_sig[abs(int(round(i/self.signal_timestep))):] += np.sign(i)*self.threshold_TD
        return rec_sig

    def use_mod_TD_decoding(self,SSA):
        #This method decodes a spike train and reconstructs the original signal based on the modified TD algorithm.
        #result: list with reconstructed signal values.
        datapoints = np.floor_divide(self.signal_duration,self.signal_timestep)
        rec_sig = np.zeros(datapoints)
        for i in SSA:
            rec_sig[abs(int(round(i/self.signal_timestep))):] += np.sign(i)*self.threshold_mod_TD
        return rec_sig

    def calc_error(self):
        #This method calculates the error between the original and the reconstructed signals
        #result: sample_list of channel_list of relative errors.
        print("Calculating the encoding error...")
        original = self.input_eeg_data
        reconstructed = self.reconstructed_data
        error_data = []
        for s in range(self.number_of_samples):
            ori_sample = original[s]
            rec_sample = reconstructed[s]
            error_sample = []
            for c in range(self.number_of_input_channels):
                ori_sig = ori_sample[c]
                rec_sig = rec_sample[c]
                sum_dif = 0
                sum_sig = 0
                for i in range(len(ori_sig)):
                    sum_dif += abs(ori_sig[i]-rec_sig[i])
                    sum_sig += abs(ori_sig[i])
                error_channel = sum_dif/sum_sig
                error_sample.append(error_channel)
            error_data.append(error_sample)
        self.error_data = error_data
        print("The mean error is " + str(np.mean(error_data)) + " with a variance of " + str(np.var(error_data)) + ".")
        return error_data

    def save_output(self):
        # This method saves all signals and spike trains to csv files
        SSA_data = self.SSA_data
        for s in range(self.number_of_samples):
            SSA_sample = SSA_data[s]
            with open(os.path.join(self.path_to_neucube_folder,'input_stage_2','sam_' + str(s+1) + '.csv'), 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                for c in range(self.number_of_input_channels):
                    SSA = SSA_sample[c]
                    writer.writerow(SSA)
        print("Spike traines saved!")
        return None

    def save_rec_sig(self):
        # This method saves all signals and spike trains to csv files
        signal_data = self.reconstructed_data
        error_data = self.error_data
        with open(os.path.join(self.path_to_neucube_folder,'memory_stage_1','error.csv'), 'wb') as csvfile:
            writer_error = csv.writer(csvfile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            for s in range(self.number_of_samples):
                writer_error.writerow(error_data[s])
                signal_sample = signal_data[s]
                with open(os.path.join(self.path_to_neucube_folder,'memory_stage_1','sam_' + str(s+1) + '.csv'), 'wb') as csvfile:
                    writer_rec = csv.writer(csvfile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                    for c in range(self.number_of_input_channels):
                        SSA = signal_sample[c]
                        writer_rec.writerow(SSA)
        print("Reconstructed signals and errors saved!")
        return None

    def plot_rec_sig(self):
        plt.figure()
        sam = np.min([3,self.number_of_samples])
        chan= np.min([3,self.number_of_input_channels])
        format = sam*100+chan*10
        time = np.linspace(0,self.signal_duration,np.floor_divide(self.signal_duration,self.signal_timestep))
        for s in range(sam):
            for c in range(chan):
                n = s*chan+(c+1)
                plt.subplot(format+n)
                plt.xlabel('time [ms]')
                plt.ylabel('normalized EEG')
                plt.title('sample ' + str(s+1) + ', channel ' + str(c+1))
                plt.plot(time, self.input_eeg_data[s][c],'b-', time, self.reconstructed_data[s][c],'r-.')
                blue_patch = mpatches.Patch(color='blue', label='original signal')
                red_patch = mpatches.Patch(color='red', label='reconstructed signal')
                plt.legend(handles=[blue_patch,red_patch])
                plt.axis([0, self.signal_duration, 0, 1])
        plt.suptitle('Decoded signals for input EEG data', fontsize=16)
        plt.show()
        return None

    def plot_output(self,encoding_method):
        plt.figure()
        sam = np.min([3,self.number_of_samples])
        chan= np.min([3,self.number_of_input_channels])
        format = sam*100+chan*10
        for s in range(sam):
            for c in range(chan):
                n = s*chan+(c+1)
                plt.subplot(format+n)
                plt.xlabel('time [ms]')
                plt.ylabel('spikes')
                plt.title('sample ' + str(s+1) + ', channel ' + str(c+1))
                if self.SSA_data[s][c]:
                    markerline, stemlines, baseline = plt.stem(np.absolute(self.SSA_data[s][c]), np.sign(self.SSA_data[s][c]),linefmt='b-',markerfmt=',')
                    plt.setp(stemlines, 'linewidth', 0.5)
                else:
                    plt.plot([0,self.signal_duration], [0,0],'r-')
                if encoding_method == 'BSA':
                    plt.axis([0, self.signal_duration, 0, 1.1])
                else:
                    plt.axis([0, self.signal_duration, -1.1, 1.1])
        plt.suptitle('Encoded spike trains from input EEG data', fontsize=16)
        plt.show()
        return None
if __name__ == "__main__":
    '''
    Test Encoder, calculate error, and plot results:
    '''
    encoder = Encoder(os.getcwd(),number_of_input_channels=3,number_of_samples=100,signal_duration=1000,signal_timestep=4)
    method = 'BSA'
    spike_trains = encoder.encode(method)
    rec_sig = encoder.decode(method)
    error = encoder.calc_error()
    encoder.save_output()
    encoder.save_rec_sig()
    encoder.plot_output(method)
    encoder.plot_rec_sig()
