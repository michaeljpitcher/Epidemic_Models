
# coding: utf-8

# In[1]:

from GraphWithStochasticDynamics import *


# In[3]:

class SIRStochasticDynamics(GraphWithStochasticDynamics):
    '''An SIR dynamics with stochastic simulation.'''

    # the possible dynamics states of a node for SIR dynamics
    SUSCEPTIBLE = 'susceptible'
    INFECTED = 'infected'
    RECOVERED = 'recovered'
    
    # list of SI edges connecting a susceptible to an infected node
    _si = []
        
    def __init__( self, time_limit = 10000, p_infect = 0.0, p_recover = 1.0, p_infected = 0.0, graph = None ):
        '''Generate a graph with dynamics for the given parameters.
        
        p_infect: infection probability (defaults to 0.0)
        p_recover: probability of recovery (defaults to 1.0)
        p_infected: initial infection probability (defaults to 0.0)
        graph: the graph to copy from (optional)'''
        states = {self.SUSCEPTIBLE,self.INFECTED,self.RECOVERED}
        rates = dict()
        rates['p_infect'] = p_infect
        rates['p_recover'] = p_recover
        rates['p_infected'] = p_infected
        GraphWithStochasticDynamics.__init__(self, time_limit = time_limit, graph = graph, states = states, rates = rates)
        self.p_infected = p_infected
        self.p_infect = p_infect
        self.p_recover = p_recover
        
    def before( self ):
        '''Seed the network with infected nodes, extract the initial set of
        SI nodes, and mark all edges as unoccupied by the dynamics.'''
        self._si = []
        
        # infect nodes
        for n in self.node.keys():
            if numpy.random.random() <= self.p_infected:
                self.node[n][self.DYNAMICAL_STATE] = self.INFECTED
            else:
                self.node[n][self.DYNAMICAL_STATE] = self.SUSCEPTIBLE
                
        self.POPULATION = self.calculate_populations()        
        
        # extract the initial set of SI edges
        for (n, m, data) in self.edges_iter(self.POPULATION[self.INFECTED], data = True):
            if self.node[m][self.DYNAMICAL_STATE] == self.SUSCEPTIBLE:
                self._si.insert(0, (n, m, data))
        
        # mark all edges as unoccupied
        for (n, m, data) in self.edges_iter(data = True):
            data[self.OCCUPIED] = False
    
    def after(self):
        pass
    
    def at_equilibrium( self):
        '''SIR dynamics is at equilibrium if there are no more infected nodes left
        in the network, no susceptible nodes adjacent to infected nodes, or if we've
        exceeded the default simulation length.
        
        returns: True if the model has stopped'''
        if self.CURRENT_TIMESTEP >= self._time_limit:
            return True
        else:
            return ((len(self.POPULATION[self.INFECTED]) == 0))
                    # or (len(self._si) == 0))

    def infect( self ):
        '''Infect a node chosen at random from the SI edges.'''
         
        # choose an SI edge
        i = numpy.random.randint(len(self._si))
        (n, m, data) = self._si[i]
        
        # infect the susceptible end
        self.update_node(m,self.SUSCEPTIBLE,self.INFECTED)
        
        # label the edge we traversed as occupied
        data[self.OCCUPIED] = True
        
        # remove all edges in the SI list from an infected node to this one
        self._si = [ (np, mp, data) for (np, mp, data) in self._si if m != mp ]
        
        # add all the edges incident on this node connected to susceptible nodes
        for (_, mp, datap) in self.edges_iter(m, data = True):
            if self.node[mp][self.DYNAMICAL_STATE] == self.SUSCEPTIBLE:
                self._si.insert(0, (m, mp, datap))
                
    def recover( self ):
        '''Cause a node to recover.'''
        
        # choose an infected node at random
        i = numpy.random.randint(len(self.POPULATION[self.INFECTED]))
        n = self.POPULATION[self.INFECTED][i]
        
        # mark the node as recovered
        self.update_node(n,self.INFECTED,self.RECOVERED)
        
        # remove all edges in the SI list incident on this node
        self._si = [ (np, m, e) for (np, m, e) in self._si if np != n ]
        
        
    def transitions( self ):
        '''Return the transition vector for the dynamics.
        
        t: time (ignored)
        returns: the transition vector'''
        
        # transitions are expressed as rates, whereas we're specified
        # in terms of probabilities, so we convert the latter to the former.
        return [ (len(self._si) * self.p_infect,        lambda t: self.infect()),
                 (len(self.POPULATION[self.INFECTED]) * self.p_recover, lambda t: self.recover()) ]
            


# In[ ]:



