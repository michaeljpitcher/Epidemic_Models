
# coding: utf-8

# In[1]:

# maths packages
import math
import numpy
import collections

# network simulation
from networkx import *
import time

# data analysis
import pandas
import pickle


# In[2]:

class GraphWithDynamics(networkx.Graph):
    '''A extension to a NetworkX undirected network with associated dynamics. This
    class combines two sets of entwined functionality: a network and
    the dynamical process being studied. This is the base class
    for studying different kinds of dynamics.'''
    
    ''' This class provides an extension to a NetworkX.Graph undirected network by
    associating disease dynamics to each node. This class combines two sets of 
    entwined functionality: a network and the dynamical process being studied. 
    This is the base class for studying different kinds of dynamics. '''
    
    # keys for node and edge attributes
    OCCUPIED = 'occupied'     # edge has been used to transfer infection or not
    DYNAMICAL_STATE = 'state'   # dynamical state of a node
    # list of states that nodes can be in
    STATES = []
    # the stats to be returned to the user
    STATISTICS = dict()
    # the current population of network (split between states)
    POPULATION = dict()
    # the current timestep of the simulation
    CURRENT_TIMESTEP = 0

    def __init__( self, graph = None , time_limit = 20000, states = [], rates = dict()):
        '''Create a graph, optionally with nodes and edges copied from
        the graph given.
        
        graph: graph to copy (optional)
        time_limit: maximum number of timesteps
        states: list of the possible node states (e.g. susceptible, infected, recovered)
        rates: the probability factors associated with the model
        '''
        Graph.__init__(self, graph)
        # Graph provided, so copy into model
        if graph is not None:
            self.copy_from(graph)
        # Set time limit
        self._time_limit = time_limit
        # Created the event distribution (ordered by timestep when event occurred)
        self._event_dist = collections.OrderedDict()
        # Historical record of each sub-population (to see how diseases spreads)
        self._pop_dist = dict()
        # Set states
        self.STATES = states
        # For each state, create a population history dictionary
        for (i) in self.STATES:
            self._pop_dist[i] = collections.OrderedDict()
        # Add the rates into the statistics    
        for (k) in rates.keys():
            self.STATISTICS[k] = rates[k]
        
    def copy_from( self, g ):
        '''Copy the nodes and edges from another graph into us.
        g: the graph to copy from
        returns: the graph'''
        
        # copy in nodes and edges from source network
        self.add_nodes_from(g.nodes_iter())
        self.add_edges_from(g.edges_iter())
        
        # remove self-loops
        es = self.selfloop_edges()
        self.remove_edges_from(es)
        
        return self
    
    def remove_all_nodes( self ):
        '''Remove all nodes and edges from the graph.'''
        self.remove_nodes_from(self.nodes())

    def at_equilibrium( self ):
        '''Test whether the model is an equilibrium. Default is just whether the 
        current timestep is greater than the set limit (can be overriden at lower
        level).
        
        timestep: the current simulation timestep
        returns: True if simulation is finished'''
        return (self.CURRENT_TIMESTEP >= self._time_limit)

    def before( self ):
        '''Run the before process. Appends the start time to the stats, then delegates to lower methods.'''
        # Add start time to stats
        self.STATISTICS['start_time'] = time.clock()
        # Run the before processes specific to the model
        self._before()
        # Calculate the initial populations
        self.POPULATION = self.calculate_populations()
        for (s) in self.STATES:
            self._pop_dist[s][self.CURRENT_TIMESTEP] = len(self.POPULATION[s])
        
    def _before( self ):
        '''Internal function defining the process to run before simulation.
        To be overriden at lower level with specifics.'''
        raise NotYetImplementedError('_before()')

    def after( self ):
        '''Run the after process. Delegates to lower methods, then appends end time and duration to stats.'''
        self._after()
        self.STATISTICS['end_time'] = time.clock()
        self.STATISTICS['duration'] = self.STATISTICS['end_time'] - self.STATISTICS['start_time']
    
    def _after( self ):
        '''Internal function defining the process to run after simulation.
        To be overriden at lower level with specifics.'''
        raise NotYetImplementedError('_after()')
        
    def dynamics( self ):
        '''Run a number of iterations of the model over the network. 
        returns: a dict of statistic'''
        
        # Run the before processes
        self.before()
        
        #Run the specific system dynamics, which returns a set of properties relevant to the model chosen
        stats = self._dynamics()
        
        # Run the after processes
        self.after()
        
        # Append those stats to the overall stats
        self.STATISTICS.update(stats)
        
        # Write the event distribution to stats
        self.STATISTICS['event_distribution'] = self._event_dist
        
        # Write each of the population distributions to the stats
        for (s) in self.STATES:
            key_string = s + '_distribution'
            self.STATISTICS[key_string] = self._pop_dist[s]
        
        return self.STATISTICS

    def _dynamics( self ):
        '''Internal function defining the way the dynamics works.
        To be overriden at lower level with specifics.'''
        raise NotYetImplementedError('_dynamics()')
    
    def skeletonise( self ):
        '''Remove unoccupied edges from the network.
        returns: the network with unoccupied edges removed'''
        
        # find all unoccupied edges
        edges = []
        for n in self.nodes_iter():
            for (np, m, data) in self.edges_iter(n, data = True):
                if (self.OCCUPIED not in data.keys()) or (data[self.OCCUPIED] != True):
                    # edge is unoccupied, mark it to be removed
                    # (safe because there are no parallel edges)
                    edges.insert(0, (n, m))
                    
        # remove them
        self.remove_edges_from(edges)
        return self
    
    def calculate_populations( self ):
        '''Return a count of the number of nodes in each dynamical state.
        returns: a dict'''
        pops = dict()
        
        # Initialise populations with empty lists
        for s in self.STATES:
            pops[s] = []
        
        # Loop through nodes and add to relevant list based on state
        for n in self.nodes_iter():
            state = self.node[n][self.DYNAMICAL_STATE]
            pops[state].append(n)
        
        return pops
                
    def rewire(self, node = 0):
        '''Placeholder to be run during simulation, Default does nothing.'''
        pass
    
    def increment_timestep(self, dt = 0.0, update_dist = True):
        '''Increment the time step, and updates the population history if requried
        Flag is needed as some model methods can result in small timestep jumps 
        where nothing happens. The default is to record the population.'''
        if update_dist:
            # Loop through states and records
            for (s) in self.STATES:
                self._pop_dist[s][self.CURRENT_TIMESTEP] = len(self.POPULATION[s])
        # Increase timestep by amount required
        self.CURRENT_TIMESTEP += dt
        
    def update_node(self, changed_node = 0, state_before = 'before', state_after = 'after'):
        '''Change a node from one state to another, and update population'''
        # Change node
        self.node[changed_node][self.DYNAMICAL_STATE] = state_after
        # Remove from previous sub-population
        self.POPULATION[state_before].remove(changed_node)
        # Add to new sub-population
        self.POPULATION[state_after].insert(0, changed_node)
        
    def reset(self):
        ''' For parallel processing. Rather than building a new network
        each time, allows the current network to reset itself back to a
        blank state. The network can then be built again and dynamics 
        run on it'''
        
        # Clear the nodes
        self.remove_all_nodes()
        
        # Remove the population distribution history
        self._pop_dist = dict()
        for (i) in self.STATES:
            self._pop_dist[i] = collections.OrderedDict()
            
        # Clear dictionaries    
        self.STATISTICS.clear()
        self.POPULATION.clear()
        
        # Reset the timestep
        self.CURRENT_TIMESTEP = 0
        
    def rebuild_barabasi_albert(self, N, M):
        ''' For parallelism. Allows the building of a new Barabasi-Albert
        network to serve as the basis for the graph.'''
        # Create BA network
        graph = barabasi_albert_graph(N,M)
        # Copy from it
        self.copy_from(graph)

