'''
created by Jan Behrenbeck
'''

import os, time
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNNClassifier
from Encoder import Encoder
from Reservoir import NeuCubeReservoir
from Classifier import Output_Neuron, Classifier

class NeuCube():
    """
    This class integrates all stages of the NeuCube model.
    """
    def __init__(self,input_electrodes,number_of_training_samples,signal_duration,signal_timestep,simulation_timestep,subject):

        self.path = os.getcwd()
        self.input_electrodes = input_electrodes
        self.number_of_training_samples = number_of_training_samples
        self.encoder = Encoder(self.path,len(input_electrodes),number_of_training_samples,signal_duration,signal_timestep,subject)
        self.reservoir = NeuCubeReservoir(self.path,simulation_timestep)
        self.classifier = Classifier()

    def encode_eeg_input(self,encoding_method,save_data,plot_data,subject):
        #This method encodes the EEG data in the input_stage_1 folder using the defined encoding_method and saves / plots the data if needed.
        spike_trains = self.encoder.encode(encoding_method,subject)
        if save_data or plot_data:
            rec_sig = self.encoder.decode(encoding_method)
            error = self.encoder.calc_error()
            if save_data:
                self.encoder.save_output()
                self.encoder.save_rec_sig()
            if plot_data:
                self.encoder.plot_output(encoding_method)
                self.encoder.plot_rec_sig()
        return spike_trains

    def create_reservoir(self,new_reservoir,plot_stability,input_electrodes,inhibitory_split, connection_probability, small_world_conn_factor, max_syn_len, w_dist_ex_mean, w_dist_inh_mean, save_structure):
        if new_reservoir:
            self.reservoir.initialize_reservoir_structure(input_electrodes,inhibitory_split, connection_probability, small_world_conn_factor, max_syn_len, w_dist_ex_mean, w_dist_inh_mean, save_structure)
            if plot_stability:
                self.reservoir.reservoir_structure.calculate_stability(inhibitory_split,w_dist_ex_mean,w_dist_inh_mean)
        else:
            self.reservoir.load_reservoir_structure()

    def train_reservoir_STDP(self,use_STDP,encoding_method,simulation_time,number_of_neurons_per_core,number_of_training_samples,spike_train_data,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,save_training_result,plot_spikes,plot_voltage):
        if use_STDP:
            self.reservoir.train_network_STDP(encoding_method,simulation_time,number_of_neurons_per_core,number_of_training_samples,spike_train_data,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,save_training_result,plot_spikes,plot_voltage)

    def train_deSNN(self,load_spikes,save_reservoir_spikes,save_neurons,encoding_method,simulation_time,number_of_neurons_per_core,number_of_training_samples,spike_train_data,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,alpha,mod,drift_up,drift_down,number_of_classes,plot_spikes,plot_voltage):
        print('Training the deSNN network...')
        # Read target_class_labels:
        tar_class_labels = []
        for item in csv.reader(open(os.path.join(self.path,'input_stage_3','tar_class_labels.txt'), 'r'), delimiter=' '):
            tar_class_labels.append(item)
        if len(tar_class_labels)<number_of_training_samples:
            print('Error: Not enough class lables for number of samples!')
        else:
            tar_class_labels = tar_class_labels[0:number_of_training_samples]
            if load_spikes: #load spikes from storage
                for s in range(number_of_training_samples):
                    sample_spikes = self.classifier.load_reservoir_spikes(os.path.join(self.path,'input_stage_3','reservoir_spikes_sam_'+str(s+1)+'.txt'))
                    neuron = Output_Neuron(sample_spikes,len(self.reservoir.reservoir_structure.get_positions()),alpha,mod,drift_up,drift_down,tar_class_labels[s])
                    if save_neurons:
                        neuron.save_whole_neuron(self.path, s)
                    self.classifier.add_neuron(neuron)
            else: #create spikes from reservoir
                STDP = False            #enable/disable STDP during deSNN training
                reservoir_spikes = self.reservoir.train_network_deSNN(encoding_method,simulation_time,number_of_neurons_per_core,number_of_training_samples,spike_train_data,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,STDP,plot_spikes,plot_voltage,save_reservoir_spikes)
                for s in range(number_of_training_samples):
                    sample_spikes = []
                    for spike in reservoir_spikes:
                        if spike[1] >= s*1.5*simulation_time and spike[1] <= (s*1.5+1)*simulation_time:
                            sample_spikes.append(spike)
                    neuron = Output_Neuron(sample_spikes,len(self.reservoir.reservoir_structure.get_positions()),alpha,mod,drift_up,drift_down,tar_class_labels[s])
                    if save_neurons:
                        neuron.save_whole_neuron(self.path, s)
                    self.classifier.add_neuron(neuron)
            print('Added all samples/neurons to the deSNN classifier!')
        return self.classifier.separation(self.path,number_of_training_samples,number_of_classes,len(self.reservoir.reservoir_structure.get_positions()))

    def classify(self,subject,save_reservoir_spikes,first_test_sample_index,number_of_test_samples,encoding_method,simulation_time,number_of_neurons_per_core,number_of_training_samples,alpha,mod,drift_up,drift_down,feature,k_neighbors):
        # Classify test samples
        labels = []
        for test_sample_index in range(first_test_sample_index,first_test_sample_index+number_of_test_samples):
            sample_EEG = self.encoder.load_sample(test_sample_index,subject)
            sample_SSA = self.encoder.encode_sample(sample_EEG,encoding_method)
            sample_reservoir_spikes = self.reservoir.filter_sample(encoding_method,test_sample_index,sample_SSA,simulation_time,number_of_neurons_per_core,save_reservoir_spikes)
            test_neuron = Output_Neuron(sample_reservoir_spikes,len(self.reservoir.reservoir_structure.get_positions()),alpha,mod,drift_up,drift_down)
            fitting_type = 'normal' #'normal' for fitting to all neurons, 'COM' for fitting to only center of mass vectors
            class_label = self.classifier.classify(test_neuron, feature, k_neighbors,fitting_type)
            labels.append(class_label[0])
        print('Predicted labels for all samples: ' + str(labels))
        # Calculate Accuracy
        tar_class_labels = []
        for item in csv.reader(open(os.path.join(self.path,'input_stage_3','tar_class_labels.txt'), 'r'), delimiter=' '):
            tar_class_labels.append(item[0])
        count = 0
        for i in range(number_of_test_samples):
            if labels[i] == tar_class_labels[number_of_training_samples+i]:
               count += 1
        print('Real labels for all samples:      '+ str(tar_class_labels[first_test_sample_index-1:first_test_sample_index+number_of_test_samples-1]))
        accuracy = count/float(number_of_test_samples)
        print('Accuracy: ' + str(accuracy))
        return(accuracy)

def run_NeuCube_simulation(input_electrodes,number_of_training_samples,subject,number_of_classes,signal_duration,signal_timestep,encoding_method,save_data,plot_data,new_reservoir,plot_stability,inhibitory_split,connection_probability,small_world_conn_factor,max_syn_len,w_dist_ex_mean,w_dist_inh_mean,save_structure,number_of_neurons_per_core,simulation_timestep,simulation_time,save_training_result,plot_spikes,plot_voltage,use_STDP,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,load_reservoir_spikes,save_reservoir_spikes,save_neurons,k_neighbors,alpha,mod,drift_up,drift_down,feature,first_test_sample_index,number_of_test_samples):
        neucube = NeuCube(input_electrodes,number_of_training_samples,signal_duration,signal_timestep,simulation_timestep,subject)
        neucube.encode_eeg_input(encoding_method, save_data, plot_data,subject)
        neucube.create_reservoir(new_reservoir,plot_stability,input_electrodes,inhibitory_split, connection_probability, small_world_conn_factor, max_syn_len, w_dist_ex_mean, w_dist_inh_mean, save_structure)
        neucube.train_reservoir_STDP(use_STDP,encoding_method,simulation_time,number_of_neurons_per_core,number_of_training_samples,neucube.encoder.SSA_data,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,save_training_result,plot_spikes,plot_voltage)
        separation = neucube.train_deSNN(load_reservoir_spikes,save_reservoir_spikes,save_neurons,encoding_method,simulation_time,number_of_neurons_per_core,number_of_training_samples,neucube.encoder.SSA_data,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,alpha,mod,drift_up,drift_down,number_of_classes,plot_spikes,plot_voltage)
        accuracy = neucube.classify(subject,save_reservoir_spikes,first_test_sample_index,number_of_test_samples,encoding_method,simulation_time,number_of_neurons_per_core,number_of_training_samples,alpha,mod,drift_up,drift_down,feature,k_neighbors)
        return accuracy, separation

if __name__ == "__main__":

    '''
    Set NeuCube parameters:
    '''
    input_electrodes                        = ['C3', 'Cz', 'C4']        # EEG: ['C3', 'Cz', 'C4']     #EMG: ['C5', 'Pz', 'C6', 'Fz']      #Emotiv-EEG: ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    number_of_training_samples              = 80
    subject                                 = 'B04T_4-5_f01-60'
    number_of_classes                       = 2
    signal_duration                         = 1000                      # in msec
    signal_timestep                         = 4                         # 7.8125 = 1000/128 Hz   4 = 1000/250Hz sampling frequency   2 = 1000/512Hz sampling frequency
    '''
    Set Encoding parameters:
    '''
    encoding_method                         = 'BSA'                     # 'BSA', 'TD', 'mod_TD'
    save_data                               = False
    plot_data                               = True
    '''
    Set Reservoir structure parameters:
    '''
    new_reservoir                           = True
    if new_reservoir:
        plot_stability                      = False
        inhibitory_split                    = 0.2
        connection_probability              = 0.25               # C
        small_world_conn_factor             = 0.5               # lambda
        max_syn_len                         = 0.5               # maximum synapse length relative to biggest neuron distance in the model
        w_dist_ex_mean                      = 3.0               # is normalized to the square root of the number of reservoir neurons
        w_dist_inh_mean                     = w_dist_ex_mean*(1-inhibitory_split)/inhibitory_split #for balanced networks
        save_structure                      = False
    '''
    Set Simulation parameters:
    '''
    number_of_neurons_per_core              = 30
    simulation_timestep                     = 1                 # [ms] should be = 1.0 (ms) for realtime applications
    simulation_time                         = signal_duration
    save_training_result                    = False             # takes some time
    plot_spikes                             = True
    plot_voltage                            = False             # takes some time
    '''
    Set STDP parameters:
    '''
    use_STDP                                = False
    tau_plus                                = 10.0
    tau_minus                               = 10.0
    A_plus                                  = 0.01
    A_minus                                 = 0.01
    w_min                                   = 0.0
    w_max                                   = 0.1
    '''
    Set Classifier parameters:
    '''
    load_reservoir_spikes                   = False             # if True, loads spikes from storage. If False, creates spikes by running samples through liquid
    save_reservoir_spikes                   = False             # saves reservoir spikes to storage
    save_neurons                            = False             # saves state vectors for deSNN network to storage
    k_neighbors                             = 1
    alpha                                   = 1
    mod                                     = 0.9   #0.9
    drift_up                                = 0.01 #0.08
    drift_down                              = 0.01 #0.01
    feature                                 = 'final_weights'   #'final_weights''initial_weights''spike_count':
    '''
    Set Test parameters:
    '''
    first_test_sample_index                 = number_of_training_samples+1
    number_of_test_samples                  = 20 #int(np.floor(1/4.0*number_of_training_samples)) #80-20 training-testing split
    number_of_simulations                   = 10
    '''
    Run NeuCube:
    '''
    accuracy = []
    separation = []
    for i in range(number_of_simulations):
        acc, sep = run_NeuCube_simulation(input_electrodes,number_of_training_samples,subject,number_of_classes,signal_duration,signal_timestep,encoding_method,save_data,plot_data,new_reservoir,plot_stability,inhibitory_split,connection_probability,small_world_conn_factor,max_syn_len,w_dist_ex_mean,w_dist_inh_mean,save_structure,number_of_neurons_per_core,simulation_timestep,simulation_time,save_training_result,plot_spikes,plot_voltage,use_STDP,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,load_reservoir_spikes,save_reservoir_spikes,save_neurons,k_neighbors,alpha,mod,drift_up,drift_down,feature,first_test_sample_index,number_of_test_samples)
        accuracy.append(acc)
        separation.append(sep)
    '''
    Save results:
    '''
    with open(os.path.join(os.getcwd(),'results','test-'+subject+'-'+time.strftime("%Y%m%d-%H%M%S")+'-acc_'+str(np.mean(accuracy))+'_'+str(np.var(accuracy))+'_'+str(accuracy)+'.txt'), 'w') as thefile:
        print>>thefile, 'Mean Accuracy:                ', np.mean(accuracy)
        print>>thefile, 'Variance:                     ', np.var(accuracy)
        print>>thefile, 'Accuracies:                   ', accuracy
        print>>thefile, '_________________________________________________'
        print>>thefile, '_________________________________________________'
        for i in range(number_of_simulations):
            print>>thefile, 'Separation Initial Weights:   ', i+1, ': ', separation[i][0][0]
            print>>thefile, 'Separation Final Weights:     ', i+1, ': ', separation[i][0][1]
            print>>thefile, 'Separation Spike Count:       ', i+1, ': ', separation[i][0][2]
            print>>thefile, '_________________________________________________'
            print>>thefile, 'Expansion Initial Weights:    ', i+1, ': ', separation[i][1][0]
            print>>thefile, 'Expansion Final Weights:      ', i+1, ': ', separation[i][1][1]
            print>>thefile, 'Expansion Spike Count:        ', i+1, ': ', separation[i][1][2]
            print>>thefile, '_________________________________________________'
            print>>thefile, 'Final Metric Initial Weights: ', i+1, ': ', separation[i][2][0]
            print>>thefile, 'Final Metric Final Weights:   ', i+1, ': ', separation[i][2][1]
            print>>thefile, 'Final Metric Spike Count:     ', i+1, ': ', separation[i][2][2]
            print>>thefile, '_________________________________________________'
            print>>thefile, '_________________________________________________'
        print>>thefile, 'NeuCube parameters:           '
        print>>thefile, 'input_electrodes:             ', input_electrodes
        print>>thefile, 'number_of_training_samples    ', number_of_training_samples
        print>>thefile, 'number_of_classes             ', number_of_classes
        print>>thefile, 'signal_duration               ', signal_duration
        print>>thefile, 'signal_timestep               ', signal_timestep
        print>>thefile, '_________________________________________________'
        print>>thefile, 'Encoding parameters:          '
        print>>thefile, 'encoding_method               ', encoding_method
        print>>thefile, 'save_data                     ', save_data
        print>>thefile, 'plot_data                     ', plot_data
        print>>thefile, '_________________________________________________'
        print>>thefile, 'Reservoir structure parameters:'
        print>>thefile, 'new_reservoir                 ', new_reservoir
        print>>thefile, 'plot_stability                ', plot_stability
        print>>thefile, 'inhibitory_split              ', inhibitory_split
        print>>thefile, 'connection_probability        ', connection_probability
        print>>thefile, 'small_world_conn_factor       ', small_world_conn_factor
        print>>thefile, 'max_syn_len                   ', max_syn_len
        print>>thefile, 'w_dist_ex_mean                ', w_dist_ex_mean
        print>>thefile, 'w_dist_inh_mean               ', w_dist_inh_mean
        print>>thefile, 'save_structure                ', save_structure
        print>>thefile, '_________________________________________________'
        print>>thefile, 'Simulation parameters:        '
        print>>thefile, 'number_of_neurons_per_core    ', number_of_neurons_per_core
        print>>thefile, 'simulation_timestep           ', simulation_timestep
        print>>thefile, 'simulation_time               ', simulation_time
        print>>thefile, 'save_training_result          ', save_training_result
        print>>thefile, 'plot_spikes                   ', plot_spikes
        print>>thefile, 'plot_voltage                  ', plot_voltage
        print>>thefile, '_________________________________________________'
        print>>thefile, 'STDP parameters:              '
        print>>thefile, 'use_STDP                      ', use_STDP
        print>>thefile, 'tau_plus                      ', tau_plus
        print>>thefile, 'tau_minus                     ', tau_minus
        print>>thefile, 'A_plus                        ', A_plus
        print>>thefile, 'A_minus                       ', A_minus
        print>>thefile, 'w_min                         ', w_min
        print>>thefile, 'w_max                         ', w_max
        print>>thefile, '_________________________________________________'
        print>>thefile, 'Classifier parameters:            '
        print>>thefile, 'load_reservoir_spikes         ', load_reservoir_spikes
        print>>thefile, 'save_reservoir_spikes         ', save_reservoir_spikes
        print>>thefile, 'save_neurons                  ', save_neurons
        print>>thefile, 'k_neighbors                   ', k_neighbors
        print>>thefile, 'alpha                         ', alpha
        print>>thefile, 'mod                           ', mod
        print>>thefile, 'drift_up                      ', drift_up
        print>>thefile, 'drift_down                    ', drift_down
        print>>thefile, 'feature                       ', feature
        print>>thefile, '_________________________________________________'
        print>>thefile, 'Test parameters:              '
        print>>thefile, 'first_test_sample_index       ', first_test_sample_index
        print>>thefile, 'number_of_test_samples        ', number_of_test_samples
