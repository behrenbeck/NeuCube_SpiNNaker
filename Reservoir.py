import os
import spynnaker8 as sim
from spynnaker8.utilities import neo_convertor
import csv
import random as rand
import numpy as np
import scipy.spatial as scipy_spatial
import matplotlib.pyplot as plt
import pyNN.utility.plotting as plot

class NetworkStructure():
    """
    This class contains methods to load a 3D neuron structure, calculate inter-neuron distances and generate connections
    based on these distances.
    """
    def __init__(self, path_to_neucube_folder, simulation_timestep):
        self.path_to_neucube_folder         = path_to_neucube_folder                # 'C:\.....'
        self.positions_list                 = []
        self.input_positions                = []
        self.distances                      = []
        self.max_distance                   = 0
        self.input_connection_list          = []
        self.inhibitory_connection_list     = []
        self.excitatory_connection_list     = []
        self.delay_factor                   = 16              #144 * simulation_timestep   # 16 ms is the max delay supported normally in SpiNNaker

    def structure_file_load(self,filename):
        """
        Loads a position file from path filename
        """
        csv_file = open(filename, 'r')
        reader = csv.reader(csv_file, delimiter=',')
        # Read neuron positions from file
        positions = []
        for line in reader:
            p = (float(line[0]), float(line[1]), float(line[2]))
            positions.append(p)
        return positions

    def calculate_distances(self):
        """
        Calculates the distances between all neurons within the loaded 3D locations file.
        This method should check if a saved version of the distances for the given dataset already exists, and only
        calculate these if it does not. It should then save a file containing them. This saves time on large datasets.
        """
        '''
        original by Nathan Scott
        modified by Jan Behrenbeck
        '''
        pos_list = np.array(self.positions_list)
        distances = scipy_spatial.distance.pdist(pos_list, metric='euclidean')
        # np.savetxt("distances.csv", self.distances, delimiter=",")
        return distances

    def calculate_connection_matrix(self, inhibitory_split, connection_probability, small_world_conn_factor, max_syn_len, w_dist_ex_mean, w_dist_inh_mean):
        """
        Calculate the connection matrixes based on distance dependent probability and a given inhibitory split.
        :param inhibitory_split: the proportion of inhibitory:excitatory neurons in the population
        """
        '''
        original by Nathan Scott
        modified by Jan Behrenbeck
        '''
        #print(self.positions_list)
        #print(len(self.distances))
        dist_mat = scipy_spatial.distance.squareform(self.distances) #creates squareform of distance matrix for easier indexing
        inp_idx = self.find_input_neurons()
        for i, presynaptic_pos in enumerate(self.positions_list):
            inhibitory_neuron = False
            for j, postsynaptic_pos in enumerate(self.positions_list):
                if i is not j and j not in inp_idx: # no self-connections and no connections to input neurons
                    normalised_distance = dist_mat[i][j] / self.max_distance
                    if normalised_distance <= max_syn_len: # distance limit for synapse creation
                        """
                        Choose the connectivity formula of your choice:
                        """
                        # Distance dependence from Neuronal Dynamics by Gerstner, Kistler, et. al.
                        #conn_prob = connection_probability * np.e ** (-normalised_distance)
                        # Distance dependence from Verstraeten, David, et al. 2007 An experimental unification...
                        lam = small_world_conn_factor;
                        conn_prob = connection_probability * np.e ** (-(normalised_distance/lam)**2)
                        if i in inp_idx:
                            conn_prob = 5*conn_prob         # doubles connection probability for synapses with input neurons to increase penetration of the liquid
                        if rand.random() < conn_prob:
                            # This ensures the the synaptic delay scales linearly with distance and also will never go above
                            # the 16ms max delay normally implemented in SpiNNaker
                            delay = normalised_distance * self.delay_factor
                            #delay = 2.0 #only for testing
                            if rand.random() < inhibitory_split and i not in inp_idx: # assure inhibitory split and always excitatory input neurons
                                self.inhibitory_connection_list.append([i, j, abs(rand.gauss(w_dist_inh_mean/np.sqrt(len(self.positions_list)), 1.0/len(self.positions_list))), delay]) #rand.gauss(0.4, 0.2)
                            else:
                                if i in inp_idx: # use higher initial weights for input neurons
                                    self.excitatory_connection_list.append([i, j, abs(rand.gauss(5*w_dist_ex_mean/np.sqrt(len(self.positions_list)), 1.0/len(self.positions_list))), delay]) #rand.gauss(0.5, 0.3)
                                else:
                                    self.excitatory_connection_list.append([i, j, abs(rand.gauss(w_dist_ex_mean/np.sqrt(len(self.positions_list)), 1.0/len(self.positions_list))), delay]) #rand.gauss(0.5, 0.3)

    def calculate_stability(self,inhibitory_split,w_dist_ex_mean,w_dist_inh_mean):
        ex_syn = self.get_excitatory_connection_list()
        inh_syn = self.get_inhibitory_connection_list()
        weight_matrix = np.zeros((len(self.get_positions()),len(self.get_positions())))
        for syn in ex_syn:
            weight_matrix[syn[0],syn[1]] = syn[2]
        for syn in inh_syn:
            weight_matrix[syn[0],syn[1]] = -syn[2]
        eigenvalues = np.linalg.eigvals(weight_matrix)
        radius = np.sqrt(1+(1-inhibitory_split)*w_dist_ex_mean**2 + inhibitory_split * w_dist_inh_mean**2)
        fig,ax = plt.subplots()
        plt.scatter(eigenvalues.real, eigenvalues.imag, marker='.')
        #ax.add_artist(plt.Circle((0, 0), radius, color='b',fill=False))
        ax.add_artist(plt.Circle((0, 0), 1, color='r',fill=False))
        #plt.suptitle('Eigenvalues of the reservoir weight matrix')
        plt.xlabel('real(eig)')
        plt.ylabel('imag(eig)')
        plt.show()

    def get_excitatory_connection_list(self):
        """
        Returns a list of connections for excitatory neurons.
        n-length list of [presynaptic index, postsynaptic index, weight, distance dependent delay]
        """
        return self.excitatory_connection_list

    def get_inhibitory_connection_list(self):
        """
        Returns a list of connections for inhibitory neurons.
        n-length list of [presynaptic index, postsynaptic index, weight, distance dependent delay]
        """
        return self.inhibitory_connection_list

    def get_input_connection_list(self):
        """
        Returns a list of connections for input neurons.
        n-length list of [source index, input neuron index, weight, distance dependent delay]
        """
        # Make the input connections
        input_neuron_indexes = self.find_input_neurons()
        input_connection_list = []
        for index, input_neuron_index in enumerate(input_neuron_indexes):
            input_connection_list.append((index, input_neuron_index, 5.0, 1.0))
        return input_connection_list

    def get_positions(self):
        """
        Returns the list of 3D positions of the neurons in the network
        n-length list of (x,y,z) locations
        """
        return self.positions_list

    def find_input_neurons(self, query_list=None, input_neighbourhood=1):
        """
        Find the nearest neurons from a list of 3D points and a search neigbourhood, to define which neurons in the 3D
        NeuCube reservoir should be connected to the spike sources based on our a-priori knowledge
        of the data.
        :param query_list: an n length list of [x,y,z] positions
        :param input_neighbourhood: the number of closest neurons to return for each point, default = 1
        :return: a list of lists (possibly of lists, if input_neighbourhood > 1) of neuron indexes in query list
        """
        locations = []
        if not query_list:
            query_list = self.input_positions
        for location in query_list:
            locations.append(self.find_closest_neuron(location, input_neighbourhood))
        return locations

    def find_closest_neuron(self, query_position, search_neigbourhood=1):
        """
        Find the nearest neuron from a given 3D point in a given neighbourhood. This is used to connect the input
        neurons to their closest location in the 3D structure.
        :param query_position: list [x,y,z] positions to query from
        :param search_neigbourhood: the number of neighbours query for, default = 1
        :return: a single list [x,y,z] of the nearest neuron positon, or a list of lists of the same
        """
        positions_tree = scipy_spatial.cKDTree(self.positions_list)
        closest_neuron_index = positions_tree.query(query_position, k=search_neigbourhood)
        return closest_neuron_index[1]

    def save_structure(self):
        '''
        Saves lists of synapses and input positions for visualization

        synapse list = [index_presynaptic_neuron    index_postsynaptic_neuron    weight    delay]
        input_list = [input_neuron_index]
        '''
        thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','ex_syn_list_pre_training.txt'), 'w')
        for item in self.excitatory_connection_list:
            print>>thefile, item[0], item[1], item[2], item[3]
        thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','inh_syn_list_pre_training.txt'), 'w')
        for item in self.inhibitory_connection_list:
            print>>thefile, item[0], item[1], item[2], item[3]
        thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','input_positions.txt'), 'w')
        for item in self.input_positions:
            print>>thefile, item[0], item[1], item[2]
        thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','neuron_positions.txt'), 'w')
        for item in self.positions_list:
            print>>thefile, item[0], item[1], item[2]
        thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','inp_conn_list.txt'), 'w')
        for item in self.get_input_connection_list():
            print>>thefile, item[0], item[1], item[2], item[3]

    def info(self):
        print('Number of reservoir neurons:         ' + str(len(self.positions_list)))
        print('Maximum distance:                    ' + str(np.round(self.max_distance,2)))
        print('Number of input electrodes/neurons:  ' + str(len(self.input_positions)))
        print('Number of excitatory synapses:       ' + str(len(self.excitatory_connection_list)))
        print('Number of inhibitory synapses:       ' + str(len(self.inhibitory_connection_list)))

'''
############################################################################################################################################################
'''

class NeuCubeReservoir():
    """
    This class contains methods to set up and run a simulation on a 3D SNN reservoir.
    """
    def __init__(self,path_to_neucube_folder,simulation_timestep):
        self.path_to_neucube_folder     = path_to_neucube_folder
        self.simulation_timestep        = simulation_timestep
        self.reservoir_structure        = NetworkStructure(self.path_to_neucube_folder,self.simulation_timestep)

    def initialize_reservoir_structure(self, input_electrodes, inhibitory_split, connection_probability, small_world_conn_factor, max_syn_len, w_dist_ex_mean, w_dist_inh_mean, save_structure):
        print('Initializing Stage 2: SNN Reservoir ...')
        self.reservoir_structure.positions_list                 = self.reservoir_structure.structure_file_load(os.path.join(self.path_to_neucube_folder,'setup_stage_2','neuron_positions.txt'))
        print('Neuron positions loaded.')
        self.reservoir_structure.input_positions                = self.find_input_positions(self.path_to_neucube_folder,input_electrodes)
        self.reservoir_structure.input_connection_list          = self.reservoir_structure.get_input_connection_list()
        print('Input positions loaded.')
        self.reservoir_structure.distances                      = self.reservoir_structure.calculate_distances()
        self.reservoir_structure.max_distance                   = max(self.reservoir_structure.distances)
        print('Distances Calculated.')
        self.reservoir_structure.calculate_connection_matrix(inhibitory_split, connection_probability, small_world_conn_factor, max_syn_len, w_dist_ex_mean, w_dist_inh_mean)
        # in case of STDP modular test:
        #self.reservoir_structure.excitatory_connection_list = [[105,505,5.0,1.0],[730,505,0.1,1.0]]
        #self.reservoir_structure.inhibitory_connection_list = [[1365,505,2.5,1.0]]
        print('Excitatory and inhibitory synapse lists created.')
        if save_structure:
            self.reservoir_structure.save_structure()
            print('Structure files saved.')
        print('Stage 2 successfully initialized:')
        self.reservoir_structure.info()
        print('Ready for training..')

    def find_input_positions(self, filename, input_electrodes):
        #load electrode list
        all_electrodes = []
        csv_file = open(os.path.join(filename,'setup_stage_2','electrode_order.txt'), 'r')
        reader = csv.reader(csv_file, delimiter=' ')
        # Read electrode information from file
        for line in reader:
            all_electrodes.append(line)
        #load all electrode positions
        all_electrode_positions = []
        csv_file = open(os.path.join(filename,'setup_stage_2','electrode_positions.txt'), 'r')
        reader = csv.reader(csv_file, delimiter=',')
        # Read electrode information from file
        for line in reader:
            all_electrode_positions.append([float(line[0]), float(line[1]), float(line[2])])
        #extract input electrode positions:
        input_positions = []
        for elec in input_electrodes:
            input_positions.append(all_electrode_positions[all_electrodes.index([elec])])
        return input_positions

    def read_pos_file(self, filename):
        csv_file = open(filename, 'r')
        reader = csv.reader(csv_file, delimiter=' ')
        # Read neuron positions from file
        positions = []
        for line in reader:
            p = [float(line[0]), float(line[1]), float(line[2])]
            positions.append(p)
        return positions

    def read_syn_list_file(self, filename):
        csv_file = open(filename, 'r')
        reader = csv.reader(csv_file, delimiter=' ')
        # Read neuron positions from file
        syn_list = []
        for line in reader:
            syn = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            syn_list.append(syn)
        return syn_list

    def read_input_spike_train(self, filename, timestep):
        '''
        Loads spike trains from csv files and creates input spike source arrays.
        '''
        csv_file = open(filename, 'r')
        reader = csv.reader(csv_file, delimiter=' ')
        spike_train = []
        for line in reader:
            spike_train.append(float(line[0]))
        SSA=[]
        for i in range(len(spike_train)):
            if spike_train[i]:
                SSA.append(i*timestep)
        #print(SSA)
        return SSA

    def load_reservoir_structure(self):
        print('Initializing Stage 2: SNN Reservoir ...')
        self.reservoir_structure.positions_list                 = self.read_pos_file(os.path.join(self.path_to_neucube_folder,'memory_stage_2','neuron_positions.txt'))
        print('Neuron positions loaded.')
        self.reservoir_structure.input_positions                = self.read_pos_file(os.path.join(self.path_to_neucube_folder,'memory_stage_2','input_positions.txt'))
        self.reservoir_structure.input_connection_list          = self.read_syn_list_file(os.path.join(self.path_to_neucube_folder,'memory_stage_2','inp_conn_list.txt'))
        print('Input positions/connections loaded.')
        self.reservoir_structure.distances                      = self.reservoir_structure.calculate_distances()
        self.reservoir_structure.max_distance                   = max(self.reservoir_structure.distances)
        print('Distances Calculated.')
        self.reservoir_structure.excitatory_connection_list     = self.read_syn_list_file(os.path.join(self.path_to_neucube_folder,'memory_stage_2','ex_syn_list_pre_training.txt'))
        self.reservoir_structure.inhibitory_connection_list     = self.read_syn_list_file(os.path.join(self.path_to_neucube_folder,'memory_stage_2','inh_syn_list_pre_training.txt'))
        print('Excitatory and inhibitory synapse lists loaded.')
        print('Stage 2 successfully loaded:')
        self.reservoir_structure.info()
        print('Ready for training..')

    def merge_samples_for_training(self,simulation_time,spike_train_data,number_of_samples):
        # This method merges all samples into one input spike source array. Between the samples there is a pause of the simulation_time.
        merged_samples = spike_train_data[0]
        for s in range(1,number_of_samples):
            sample = spike_train_data[s]
            for c in range(len(sample)):
                merged_samples[c] += ([x + 1.5*s*simulation_time for x in sample[c]])
        return merged_samples

    def train_network_STDP(self,encoding_method,simulation_time,number_of_neurons_per_core,number_of_samples,spike_train_data,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,save_training_result,plot_spikes,plot_voltage):
        print('Training the reservoir using STDP...')
        # Set up the hardware
        sim.setup(self.simulation_timestep)
        sim.set_number_of_neurons_per_core(sim.IF_curr_exp(), number_of_neurons_per_core)
        # Create a population of neurons for the reservoir
        reservoir_pop = sim.Population(len(self.reservoir_structure.get_positions()), sim.IF_curr_exp(), label="Reservoir")
        # Set up STDP learning
        timing_rule = sim.SpikePairRule(tau_plus, tau_minus, A_plus, A_minus)
        #weight_rule = sim.AdditiveWeightDependence(w_min, w_max)
	weight_rule = sim.MultiplicativeWeightDependence(w_min, w_max)
        stdp_model  = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule, weight = 0.25, delay = 5.0) #weight and delay are ignored if specified by synapse list
        #Connect Neurons within the Reservoir
        excitatory_connector = sim.FromListConnector(self.reservoir_structure.get_excitatory_connection_list())
        inhibitory_connector = sim.FromListConnector(self.reservoir_structure.get_inhibitory_connection_list())
        excitatory_projection = sim.Projection(reservoir_pop, reservoir_pop, excitatory_connector, synapse_type = stdp_model)
        inhibitory_projection = sim.Projection(reservoir_pop, reservoir_pop, inhibitory_connector, synapse_type = stdp_model, receptor_type='inhibitory')
        # Merge data for training
        input = self.merge_samples_for_training(simulation_time, spike_train_data, number_of_samples)
        # Create input population
        if encoding_method == 'BSA':
            input_pop = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(input), label="Input")
            # Connect the input spike trains with the "input" neurons in the reservoir
            input_projection = sim.Projection(input_pop, reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list))
        elif encoding_method == 'TD' or encoding_method == 'mod_TD':
            # sort ex and inh spikes
            input_ex  = [[],[],[]]
            input_inh = [[],[],[]]
            for i in range(len(input)):
                for spike in input[i]:
                    if spike >= 0:
                        input_ex[i].append(spike)
                    else:
                        input_inh[i].append(-spike)
            input_pop_ex  = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(input_ex), label="Input")
            input_pop_inh = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(input_inh), label="Input")
            # Connect the input spike trains with the "input" neurons in the reservoir
            input_projection_ex  = sim.Projection(input_pop_ex,  reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list))
            input_projection_inh = sim.Projection(input_pop_inh, reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list), receptor_type='inhibitory')
        else:
            print('Unsupported enconding method. Choose TD, mod_TD, or BSA.')

        # Run the Simulation and record the spike trains, membrane voltage and weights of all neurons
        reservoir_pop.record(["spikes","v"])

        if save_training_result:
            for s in range(number_of_samples):
                print('Training with sample ' + str(s+1) + '...')
                # Run the actual simulation
                sim.run(1.5*simulation_time)
                # Save weights between samples
                excitatory_connection_list = self.reservoir_structure.get_excitatory_connection_list()
                inhibitory_connection_list = self.reservoir_structure.get_inhibitory_connection_list()
                # Get final weights from projections
                ex_syn_res = excitatory_projection.get(attribute_names=['weight'],format='list')
                in_syn_res = inhibitory_projection.get(attribute_names=['weight'],format='list')
                # Update excitatory_connection_list with new weights
                for i in range(len(excitatory_connection_list)):
                    excitatory_connection_list[i][2] = ex_syn_res[i][2]
                # Update inhibitory_connection_list with new weights
                for i in range(len(inhibitory_connection_list)):
                    inhibitory_connection_list[i][2] = in_syn_res[i][2]
                # Save lists and spike activity to files
                thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','ex_syn_list_post_training_sam_' + str(s+1) + '.txt'), 'w')
                for item in excitatory_connection_list:
                    print>>thefile, item[0], item[1], item[2], item[3]
                thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','inh_syn_list_post_training_sam_' + str(s+1) + '.txt'), 'w')
                for item in inhibitory_connection_list:
                    print>>thefile, item[0], item[1], item[2], item[3]
        else:
            print('Training with ' + str(number_of_samples) + ' samples ...')
            sim.run(number_of_samples*1.5*simulation_time)
        # Get the output spikes and membrane voltages (only if necessary)
        print('Collecting data from SpiNNaker ...')
        if save_training_result:
            neo_reservoir = reservoir_pop.get_data(variables=["spikes","v"])
            result_spikes = neo_reservoir.segments[0].spiketrains
            result_v_memb = neo_reservoir.segments[0].filter(name='v')[0]
        else:
            if plot_spikes:
                neo_reservoir = reservoir_pop.get_data(variables=["spikes"])
                result_spikes = neo_reservoir.segments[0].spiketrains
            if plot_voltage:
                neo_reservoir = reservoir_pop.get_data(variables=["v"])
                result_v_memb = neo_reservoir.segments[0].filter(name='v')[0]
        # Update weights in Network Structure
        # Get original list
        excitatory_connection_list = self.reservoir_structure.get_excitatory_connection_list()
        inhibitory_connection_list = self.reservoir_structure.get_inhibitory_connection_list()
        # Get final weights from projections
        ex_syn_res = excitatory_projection.get(attribute_names=['weight'],format='list')
        in_syn_res = inhibitory_projection.get(attribute_names=['weight'],format='list')
        # Update excitatory_connection_list with new weights
        for i in range(len(excitatory_connection_list)):
            excitatory_connection_list[i][2] = ex_syn_res[i][2]
        # Update inhibitory_connection_list with new weights
        for i in range(len(inhibitory_connection_list)):
            inhibitory_connection_list[i][2] = in_syn_res[i][2]
        # Set weights in Network Structure
        self.reservoir_structure.excitatory_connection_list = excitatory_connection_list
        self.reservoir_structure.inhibitory_connection_list = inhibitory_connection_list
        print('Reservoir weights have been updated.')
        # Save spikes
        if save_training_result:
            # Save lists and spike activity to files
            thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','ex_syn_list_post_training_final.txt'), 'w')
            for item in excitatory_connection_list:
                print>>thefile, item[0], item[1], item[2], item[3]
            thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','inh_syn_list_post_training_final.txt'), 'w')
            for item in inhibitory_connection_list:
                print>>thefile, item[0], item[1], item[2], item[3]
            spikes = neo_convertor.convert_data(neo_reservoir,'spikes')
            thefile = open(os.path.join(self.path_to_neucube_folder,'memory_stage_2','spikes_post_training.txt'), 'w')
            for item in spikes:
                print>>thefile, item[0], item[1]
            print('All files have been saved.')
        # Visualize spike trains and membrane potentials
        if plot_spikes:
            plot.Figure(plot.Panel(result_spikes, yticks=True, xticks=True, marker = '|', markersize=1, xlim=(0,number_of_samples*1.5*simulation_time)))
            plt.show()
        if plot_voltage:
            plot.Figure(plot.Panel(result_v_memb, yticks=True, xticks=True, markersize=5, xlim=(0,number_of_samples*1.5*simulation_time),legend=False))
            plt.show()
        sim.end()
        print('STDP training is finished! Ready for training of deSNN.')

    def train_network_deSNN(self,encoding_method,simulation_time,number_of_neurons_per_core,number_of_samples,spike_train_data,tau_plus,tau_minus,A_plus,A_minus,w_min,w_max,STDP,plot_spikes,plot_voltage,save_reservoir_spikes):
        # This method runs all samples through the reservoir and records the spiking behaviour of the reservoir. STDP learning can be turned on and off.
        # Set up the hardware
        sim.setup(self.simulation_timestep)
        sim.set_number_of_neurons_per_core(sim.IF_curr_exp(), number_of_neurons_per_core)
        # Create a population of neurons for the reservoir
        reservoir_pop = sim.Population(len(self.reservoir_structure.get_positions()), sim.IF_curr_exp(), label="Reservoir")

        if STDP:
            # Set up STDP learning
            timing_rule = sim.SpikePairRule(tau_plus, tau_minus, A_plus, A_minus)
            weight_rule = sim.AdditiveWeightDependence(w_min, w_max)
            stdp_model  = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule, weight = 0.25, delay = 5.0) #weight and delay are ignored if specified by synapse list
            #Connect Neurons within the Reservoir
            excitatory_connector = sim.FromListConnector(self.reservoir_structure.get_excitatory_connection_list())
            inhibitory_connector = sim.FromListConnector(self.reservoir_structure.get_inhibitory_connection_list())
            excitatory_projection = sim.Projection(reservoir_pop, reservoir_pop, excitatory_connector, synapse_type = stdp_model)
            inhibitory_projection = sim.Projection(reservoir_pop, reservoir_pop, inhibitory_connector, synapse_type = stdp_model, receptor_type='inhibitory')
        else:
            #Connect Neurons within the Reservoir
            excitatory_connector = sim.FromListConnector(self.reservoir_structure.get_excitatory_connection_list())
            inhibitory_connector = sim.FromListConnector(self.reservoir_structure.get_inhibitory_connection_list())
            excitatory_projection = sim.Projection(reservoir_pop, reservoir_pop, excitatory_connector)
            inhibitory_projection = sim.Projection(reservoir_pop, reservoir_pop, inhibitory_connector, receptor_type='inhibitory')

        # Merge data for training
        input = self.merge_samples_for_training(simulation_time, spike_train_data, number_of_samples)

        # Create input population
        if encoding_method == 'BSA':
            input_pop = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(input), label="Input")
            # Connect the input spike trains with the "input" neurons in the reservoir
            input_projection = sim.Projection(input_pop, reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list))
        elif encoding_method == 'TD' or encoding_method == 'mod_TD':
            # sort ex and inh spikes
            input_ex  = [[],[],[]]
            input_inh = [[],[],[]]
            for i in range(len(input)):
                for spike in input[i]:
                    if spike >= 0:
                        input_ex[i].append(spike)
                    else:
                        input_inh[i].append(-spike)
            input_pop_ex  = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(input_ex), label="Input")
            input_pop_inh = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(input_inh), label="Input")
            # Connect the input spike trains with the "input" neurons in the reservoir
            input_projection_ex  = sim.Projection(input_pop_ex,  reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list))
            input_projection_inh = sim.Projection(input_pop_inh, reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list), receptor_type='inhibitory')
        else:
            print('Unsupported enconding method. Choose TD, mod_TD, or BSA.')

        # Run the Simulation and record the spike trains, membrane voltage and weights of all neurons
        reservoir_pop.record(["spikes","v"])
        print('Running samples through reservoir...')
        # Run the actual simulation
        sim.run(number_of_samples*1.5*simulation_time)

        # Get the output spikes and membrane voltages (only if necessary)
        # Save lists and spike activity to files
        neo_reservoir = reservoir_pop.get_data(variables=["spikes"])#,"v"])
        spikes = neo_convertor.convert_data(neo_reservoir,'spikes')
        if save_reservoir_spikes:
            for s in range(number_of_samples):
                thefile = open(os.path.join(self.path_to_neucube_folder,'input_stage_3','reservoir_spikes_sam_' + str(s+1) + '.txt'), 'w')
                for item in spikes:
                    if item[1] >= s*1.5*simulation_time and item[1] <= (s*1.5+1)*simulation_time:
                        print>>thefile, item[0], (item[1]-s*1.5*simulation_time)
            print('Reservoir spikes have been saved.')
        # Visualize spike trains
        if plot_spikes:
            plot.Figure(plot.Panel(neo_reservoir.segments[0].spiketrains, yticks=True, xticks=True, marker = '|', markersize=1, xlim=(0,number_of_samples*1.5*simulation_time)))
            plt.show()
        if plot_voltage:
            neo_reservoir = reservoir_pop.get_data(variables=["v"])
            plot.Figure(plot.Panel(neo_reservoir.segments[0].filter(name='v')[0], yticks=True, xticks=True, markersize=5, xlim=(0,number_of_samples*1.5*simulation_time),legend=False))
            plt.show()
        sim.end()
        return spikes

    def filter_sample(self,encoding_method,sample_number,sample,simulation_time,number_of_neurons_per_core,save_reservoir_spikes):
        print('Filtering sample ...')
        # Set up the hardware
        sim.setup(self.simulation_timestep)
        sim.set_number_of_neurons_per_core(sim.IF_curr_exp(), number_of_neurons_per_core)
        # Create a population of neurons for the reservoir
        reservoir_pop = sim.Population(len(self.reservoir_structure.get_positions()), sim.IF_curr_exp(), label="Reservoir")
        #Connect Neurons within the Reservoir
        excitatory_connector = sim.FromListConnector(self.reservoir_structure.get_excitatory_connection_list())
        inhibitory_connector = sim.FromListConnector(self.reservoir_structure.get_inhibitory_connection_list())
        excitatory_projection = sim.Projection(reservoir_pop, reservoir_pop, excitatory_connector)
        inhibitory_projection = sim.Projection(reservoir_pop, reservoir_pop, inhibitory_connector, receptor_type='inhibitory')
        # Create input population
        if encoding_method == 'BSA':
            input_pop = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(sample), label="Input")
            # Connect the input spike trains with the "input" neurons in the reservoir
            input_projection = sim.Projection(input_pop, reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list))
        elif encoding_method == 'TD' or encoding_method == 'mod_TD':
            # sort ex and inh spikes
            input_ex  = [[],[],[]]
            input_inh = [[],[],[]]
            for i in range(len(sample)):
                for spike in sample[i]:
                    if spike >= 0:
                        input_ex[i].append(spike)
                    else:
                        input_inh[i].append(-spike)
            input_pop_ex  = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(input_ex), label="Input")
            input_pop_inh = sim.Population(len(self.reservoir_structure.get_input_connection_list()), sim.SpikeSourceArray(input_inh), label="Input")
            # Connect the input spike trains with the "input" neurons in the reservoir
            input_projection_ex  = sim.Projection(input_pop_ex,  reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list))
            input_projection_inh = sim.Projection(input_pop_inh, reservoir_pop, sim.FromListConnector(self.reservoir_structure.input_connection_list), receptor_type='inhibitory')
        else:
            print('Unsupported enconding method. Choose TD, mod_TD, or BSA.')

        # Run the Simulation and record the spike trains, membrane voltage and weights of all neurons
        reservoir_pop.record(["spikes","v"])
        # Run the actual simulation
        sim.run(simulation_time)
        # Get the output spikes and membrane voltages (only if necessary)
        print('Collecting data from SpiNNaker ...')
        neo_reservoir = reservoir_pop.get_data(variables=["spikes"])#,"v"])
        result_spikes = neo_reservoir.segments[0].spiketrains
        #result_v_memb = neo_reservoir.segments[0].filter(name='v')[0]
        # Save lists and spike activity to files
        spikes = neo_convertor.convert_data(neo_reservoir,'spikes')
        if save_reservoir_spikes:
            thefile = open(os.path.join(self.path_to_neucube_folder,'input_stage_3','reservoir_spikes_sam_' + str(sample_number) + '.txt'), 'w')
            for item in spikes:
                print>>thefile, item[0], item[1]
            print('Reservoir spikes have been saved.')
        sim.end()
        return spikes

if __name__ == "__main__":
    ncr = NeuCubeReservoir(os.getcwd(),1)
    ncr.initialize_reservoir_structure(['C3','C4','Cz'],0.2, 0.13, 0.2, 0.3, 0.2, 0.8,False)
    #ncr.load_reservoir_structure()
    ncr.reservoir_structure.calculate_stability(0.2,0.2,0.8)
