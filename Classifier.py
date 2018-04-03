'''
created by Jan Behrenbeck 01.02.2018
'''
import os
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNNClassifier

class Output_Neuron():
    """
    This class specifies weights for the connection of a classification neuron to a reservoir to classify spike trains that come from a SNN reservoir using the deSNN algorithm.
    """
    def __init__(self,spike_trains,num_neurons=5,alpha=1,mod=0.9,drift_up=0.08, drift_down=0.01, label='Test'):
        self.class_label           = label
        self.spike_trains          = self.disconnect_input_neurons(spike_trains)
        self.chronological_spikes  = self.sort_chrono(self.spike_trains)
        self.neuron_rank_list      = self.get_rank_order(self.chronological_spikes)
        self.initial_weights       = self.calc_init_weights(alpha, mod, num_neurons, self.neuron_rank_list)
        self.spikes_per_neuron     = self.count_spikes(self.spike_trains, num_neurons)
        self.first_spikes          = self.first_spike_timing(self.spike_trains, num_neurons)
        self.final_weights         = self.SDSP_weight_adjustment(drift_up, drift_down, num_neurons, self.initial_weights, self.spikes_per_neuron, self.neuron_rank_list, self.first_spikes)

    def disconnect_input_neurons(self,spike_trains):
        # this method disconnects the output classification neurons from the input reservoir neurons by deleting the respective reservoir spikes.
        result = []
        for spike in spike_trains:
            if spike[0] not in [105 , 730 , 1365]:
                result.append(spike)
        return result

    def sort_chrono(self,spike_trains):
        #This method sorts the reservoir spike train list chronologically.
        sortable = []
        for spike in spike_trains:
            sortable.append(list(spike[::-1]))
        sorted_list = sorted(sortable)
        final = []
        for spike in sorted_list:
            final.append(spike[::-1])
        return final

    def get_rank_order(self,sorted_spike_trains):
        #This method creates a table that contains the neuron index, first spike time, and rank for each firing reservoir neuron.
        rank_order_list = []
        neurons_done = []
        for spike in sorted_spike_trains:
            if spike[0] not in neurons_done:
                rank_order_list.append([spike[0],spike[1],sorted_spike_trains.index(spike)]) # [ neuron_index, spike_time, synapse_rank]
                neurons_done.append(spike[0])
        return rank_order_list

    def calc_init_weights(self, alpha, mod, num_neurons, spike_order_list):
        #This method calculates the initial weights for the connection between reservoir and output neuron according to the rank order learning principle.
        init_weights_for_neurons = np.zeros(num_neurons)
        for element in spike_order_list:
            init_weights_for_neurons[int(element[0])] = alpha * mod ** element[2]
        return init_weights_for_neurons

    def count_spikes(self,spike_train,num_neurons):
        #This method counts how many spikes are fired by each neuron in the reservoir during one sample presentation.
        spikes_per_neuron = np.zeros(num_neurons)
        for spike in spike_train:
            neuron_index = int(spike[0])
            spikes_per_neuron[neuron_index] +=1
        return spikes_per_neuron

    def first_spike_timing(self,spike_trains,num_neurons):
        #This method extracts the timing for the first spike on each reservoir neuron. If 0 there was no spike.
        first_spike_time_per_neuron = np.zeros(num_neurons)
        if spike_trains:
            n_idx = int(spike_trains[0][0])
            first_spike_time_per_neuron[n_idx] = spike_trains[0][1]
            for spike in spike_trains:
                if spike[0] != n_idx:
                    n_idx = int(spike[0])
                    first_spike_time_per_neuron[n_idx] = spike[1]
        return first_spike_time_per_neuron

    def SDSP_weight_adjustment(self,drift_up,drift_down,num_neurons,init_weights,spikes_per_neuron,neuron_rank_list,first_spikes):
        #This method uses SDSP learning to adjust the initial weights with respect to the remaining spike train on each synapse
        final_weights = np.zeros(num_neurons)
        if self.chronological_spikes:
            last_spike_time = self.chronological_spikes[-1][1]
            for i in range(num_neurons):
                if spikes_per_neuron[i] != 0:
                    duration_after_first_spike = last_spike_time - first_spikes[i]
                    timesteps_with_spike = spikes_per_neuron[i]-1
                    timesteps_without_spike = duration_after_first_spike - timesteps_with_spike
                    final_weights[i] = init_weights[i] + drift_up*timesteps_with_spike - drift_down*timesteps_without_spike
        return final_weights

    def save_file(self, variable, filename, line_elements, type):
        #This method saves a variable to a file
        thefile = open(filename, 'w')
        if type=='list':
            for item in variable:
                if line_elements == 1:
                    print>>thefile, item
                elif line_elements == 2:
                    print>>thefile, item[0], item[1]
                elif line_elements == 3:
                    print>>thefile, item[0], item[1], item[2]
                elif line_elements == 4:
                    print>>thefile, item[0], item[1], item[2], item[3]
        if type=='string':
            print>>thefile, variable

    def save_whole_neuron(self,path, sample_number='0'):
        self.save_file(self.neuron_rank_list,  os.path.join(path,'memory_stage_3',str(sample_number+1)+'_neuron_rank_list.txt'),3,'list')
        self.save_file(self.initial_weights,   os.path.join(path,'memory_stage_3',str(sample_number+1)+'_initial_weights.txt'),1,'list')
        self.save_file(self.spikes_per_neuron, os.path.join(path,'memory_stage_3',str(sample_number+1)+'_spikes_per_neuron.txt'),1,'list')
        self.save_file(self.final_weights,     os.path.join(path,'memory_stage_3',str(sample_number+1)+'_final_weights.txt'),1,'list')

class Classifier():
    """
    This class contains methods to classify spike trains that come from a SNN reservoir using the deSNN algorithm.
    """
    def __init__(self):
        self.neuron_final_weights_list = []
        self.neuron_initial_weights_list = []
        self.neuron_total_spike_count_list = []
        self.target_list = []
        self.deSNN = []
        self.kNN_clf = None
	self.COM = []

    def load_reservoir_spikes(self,filename):
        #This method loads a spike train file containing all the fired spikes within a reservoir
        csv_file = open(filename, 'r')
        reader = csv.reader(csv_file, delimiter=' ')
        # Read from file
        spike_trains = []
        for line in reader:
            n = [float(line[0]),float(line[1])]
            spike_trains.append(n)
        return spike_trains

    def add_neuron(self, neuron):
        #This method adds a neuron's weight vector and label to the classifier
        self.neuron_final_weights_list.append(neuron.final_weights)
        self.neuron_initial_weights_list.append(neuron.initial_weights)
        self.neuron_total_spike_count_list.append(neuron.spikes_per_neuron)
        self.target_list.append(neuron.class_label)
        self.deSNN.append(neuron)
        return 'Neuron added!'

    def fit_model_COM(self, feature):
        #This method fits the model and lets it learn only the center of mass neuron for each class. Choose a feature type.
        if feature == 'final_weights':
            self.kNN_clf.fit([self.COM[0][1] , self.COM[1][1]], ['1', '2'])              #Fit the model using X as training data and y as target values/labels
        elif feature == 'initial_weights':
            self.kNN_clf.fit([self.COM[0][0] , self.COM[1][0]], ['1', '2'])
        if feature == 'spike_count':
            self.kNN_clf.fit([self.COM[0][2] , self.COM[1][2]], ['1', '2'])
        return 'Model fitting completed!'

    def fit_model(self, feature):
        #This method fits the model and lets it learn all neurons that have been added. Choose a feature type.
        if feature == 'final_weights':
            self.kNN_clf.fit(self.neuron_final_weights_list, np.ravel(self.target_list))              #Fit the model using X as training data and y as target values/labels
        elif feature == 'initial_weights':
            self.kNN_clf.fit(self.neuron_initial_weights_list, np.ravel(self.target_list))
        if feature == 'spike_count':
            self.kNN_clf.fit(self.neuron_total_spike_count_list, np.ravel(self.target_list))
        return 'Model fitting completed!'

    def classify(self, neuron, feature, k_neighbors, fitting_type):
        #This method predicts the label for a neuron.
        self.kNN_clf = kNNClassifier(k_neighbors)
	if fitting_type == 'normal':
       	    self.fit_model(feature)
	elif fitting_type == 'COM':
       	    self.fit_model_COM(feature)

        if feature == 'final_weights':
            class_label = self.kNN_clf.predict([neuron.final_weights])
        elif feature == 'initial_weights':
            class_label = self.kNN_clf.predict([neuron.initial_weights])
        elif feature == 'spike_count':
            class_label = self.kNN_clf.predict([neuron.spikes_per_neuron])
        return class_label

    def separation(self,path,num_samples,num_classes,num_neurons):
        # calculate center of masses
        center_of_mass = []       # list containing a tuple for each class with the respective COMs for each feature (init, final, count)
        for i in range(num_classes):        # iterate over classes: 0,1
            elem_per_class = 0
            vector_sum_ini = np.zeros(num_neurons)
            vector_sum_fin = np.zeros(num_neurons)
            vector_sum_cou = np.zeros(num_neurons)
            for s in range(num_samples):
                if self.target_list[s][0] == str(i+1):
                    vector_sum_ini += self.neuron_initial_weights_list[s]
                    vector_sum_fin += self.neuron_final_weights_list[s]
                    vector_sum_cou += self.neuron_total_spike_count_list[s]
                    elem_per_class += 1
            com_ini = vector_sum_ini / elem_per_class
            com_fin = vector_sum_fin / elem_per_class
            com_cou = vector_sum_cou / elem_per_class
            center_of_mass.append((com_ini, com_fin, com_cou))
	self.COM = center_of_mass
        print('Center of masses have been calculated.')
        # calculate sparation
        sum_ini = 0
        sum_fin = 0
        sum_cou = 0
        for i in range(num_classes):
            for j in range(num_classes):
                sum_ini += np.linalg.norm(center_of_mass[i][0]-center_of_mass[j][0],2)  #euclidean distance
                sum_fin += np.linalg.norm(center_of_mass[i][1]-center_of_mass[j][0],2)
                sum_cou += np.linalg.norm(center_of_mass[i][2]-center_of_mass[j][0],2)
        sep_ini = sum_ini/num_classes**2
        sep_fin = sum_fin/num_classes**2
        sep_cou = sum_cou/num_classes**2
        #print('Separation for feature "initial weights": ' + str(sep_ini))
        #print('Separation for feature "final weights":   ' + str(sep_fin))
        #print('Separation for feature "spike count":     ' + str(sep_cou))
        # calculate expansion
        sum_ini = 0
        sum_fin = 0
        sum_cou = 0
        av_dist = []            # list containing a tuple for each class with the respective average distance betw. state vector and COM for each feature (init, final, count)
        for i in range(num_classes): # iterate over classes: 0,1
            elem_per_class = 0
            dist_sum_ini = 0
            dist_sum_fin = 0
            dist_sum_cou = 0
            for s in range(num_samples):
                if self.target_list[s][0] == str(i+1):
                    dist_sum_ini += np.linalg.norm(self.neuron_initial_weights_list[s]-center_of_mass[i][0],2)
                    dist_sum_fin += np.linalg.norm(self.neuron_final_weights_list[s]-center_of_mass[i][1],2)
                    dist_sum_cou += np.linalg.norm(self.neuron_total_spike_count_list[s]-center_of_mass[i][2],2)
                    elem_per_class += 1
            av_dist_ini = dist_sum_ini / elem_per_class
            av_dist_fin = dist_sum_fin / elem_per_class
            av_dist_cou = dist_sum_cou / elem_per_class
            av_dist.append((av_dist_ini, av_dist_fin, av_dist_cou))
        for i in range(num_classes):
            sum_ini += av_dist[i][0]
            sum_fin += av_dist[i][1]
            sum_cou += av_dist[i][2]
        exp_ini = sum_ini/num_classes
        exp_fin = sum_fin/num_classes
        exp_cou = sum_cou/num_classes
        #print('Expansion for feature "initial weights": ' + str(exp_ini))
        #print('Expansion for feature "final weights":   ' + str(exp_fin))
        #print('Expansion for feature "spike count":     ' + str(exp_cou))
        # calculate optimization function
        print('Optimization Function for feature "initial weights": ' + str(sep_ini-2*exp_ini))
        print('Optimization Function for feature "final weights":   ' + str(sep_fin-2*exp_fin))
        print('Optimization Function for feature "spike count":     ' + str(sep_cou-2*exp_cou))

        return [(sep_ini,sep_fin,sep_cou),(exp_ini,exp_fin,exp_cou),(sep_ini-2*exp_ini,sep_fin-2*exp_fin,sep_cou-2*exp_cou)]

if __name__ == "__main__":

    classifier = Classifier()
    # Test network
    out_neuron_1 = Output_Neuron(classifier.load_reservoir_spikes(os.path.join(os.getcwd(),'input_stage_3','deSNN_Test_files','Test_1.txt')),label='1')
    out_neuron_2 = Output_Neuron(classifier.load_reservoir_spikes(os.path.join(os.getcwd(),'input_stage_3','deSNN_Test_files','Test_2.txt')),label='2')
    test_neuron  = Output_Neuron(classifier.load_reservoir_spikes(os.path.join(os.getcwd(),'input_stage_3','deSNN_Test_files','Test_3.txt')),label='2')

    classifier.add_neuron(out_neuron_1)
    classifier.add_neuron(out_neuron_2)
    print(classifier.classify(test_neuron,'final_weights',1))
