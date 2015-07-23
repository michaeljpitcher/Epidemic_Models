
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

# plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'svg'")


# In[2]:

class GraphWithDynamics(networkx.Graph):
    '''A NetworkX undirected network with associated dynamics. This
    class combines two sets of entwined functionality: a network and
    the dynamical process being studied. This is the base class
    for studying different kinds of dynamics.'''

    # keys for node and edge attributes
    OCCUPIED = 'occupied'     # edge has been used to transfer infection or not
    DYNAMICAL_STATE = 'state'   # dynamical state of a node
    STATES = []
    STATISTICS = dict()
    POPULATION = dict()
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
        if graph is not None:
            self.copy_from(graph)
        self._time_limit = time_limit
        self._event_dist = collections.OrderedDict()
        self._pop_dist = dict()
        self.STATES = states
        # For each state, create a population history dictionary
        for (i) in self.STATES:
            self._pop_dist[i] = collections.OrderedDict()
        # Add the reates into the statistics    
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
        '''Internal function defining the process to run before simulation.'''
        raise NotYetImplementedError('_before()')

    def after( self ):
        '''Run the after process. Delegates to lower methods, then appends end time and duration to stats.'''
        self._after()
        self.STATISTICS['end_time'] = time.clock()
        self.STATISTICS['duration'] = self.STATISTICS['end_time'] - self.STATISTICS['start_time']
    
    def _after( self ):
        '''Internal function defining the process to run after simulation.'''
        raise NotYetImplementedError('_after()')
        
    def dynamics( self ):
        '''Run a number of iterations of the model over the network. 
        returns: a dict of properties'''
        
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
            string = s + '_distribution'
            self.STATISTICS[string] = self._pop_dist[s]
        
        return self.STATISTICS

    def _dynamics( self ):
        '''Internal function defining the way the dynamics works.
        returns: a dict of properties'''
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
        '''Placeholder to be run after simulation, Defaults does nothing.'''
        pass
    
    def increment_timestep(self, dt = 0.0, update_dist = True):
        '''Increment the time step, updates the population history'''
        # Update population history if requested
        if update_dist:
            for (s) in self.STATES:
                self._pop_dist[s][self.CURRENT_TIMESTEP] = len(self.POPULATION[s])
        # Increase timestep by 1
        self.CURRENT_TIMESTEP += dt
        
    def update_node(self, changed_node = 0, state_before = 'before', state_after = 'after'):
        '''Change a node from one state to another, and update population'''
        self.node[changed_node][self.DYNAMICAL_STATE] = state_after
        self.POPULATION[state_before].remove(changed_node)
        self.POPULATION[state_after].insert(0, changed_node)
        
    def plot_distributions(self, title_string = ''):
        fig = plt.figure(figsize = (8, 5))
        for i in self.STATES:
            plt.plot(self._pop_dist[i].keys(),self._pop_dist[i].values(), label=i)
        plt.ylabel('Number of nodes')
        plt.xlabel('Timestep')
        plt.title(title_string)
        plt.legend(loc=5)
        plt.show()

