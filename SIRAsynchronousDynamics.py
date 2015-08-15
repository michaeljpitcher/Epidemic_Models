
# coding: utf-8

# In[1]:

from GraphWithAsynchronousDynamics import *


# In[2]:

class SIRAsynchronousDynamics(GraphWithAsynchronousDynamics):
    '''A graph with a particular SIR dynamics. We use probabilities
    to express infection and recovery per timestep, and run the system
    using synchronous dynamics.'''
    
    # the possible dynamics states of a node for SIR dynamics
    SUSCEPTIBLE = 'susceptible'
    INFECTED = 'infected'
    RECOVERED = 'recovered'
    
    # list of infected nodes, the sites of all the dynamics
    _infected = []
    _susceptible = []
    _recovered = []
    
    
    def __init__( self, time_limit = 10000, p_infect = 0.0, p_recover = 1.0, p_infected = 0.0, graph = None ):
        '''Generate a graph with dynamics for the given parameters.
        
        pInfect: infection probability (defaults to 0.0)
        pRecover: probability of recovery (defaults to 1.0)
        pInfected: initial infection probability (defaults to 0.0)
        g: the graph to copy from (optional)'''
        states = {self.SUSCEPTIBLE,self.INFECTED,self.RECOVERED}
        rates = dict()
        rates['p_infect'] = p_infect
        rates['p_recover'] = p_recover
        rates['p_infected'] = p_infected
        GraphWithAsynchronousDynamics.__init__(self, time_limit = time_limit, graph = graph, states = states, rates = rates)
        self._p_infect = p_infect
        self._p_recover = p_recover
        self._p_infected = p_infected
            
    def _before( self ):
        '''Seed the network with infected nodes, and mark all edges
        as unoccupied by the dynamics.'''
        self._infected = []       # in case we re-run from a dirty intermediate state
        for n in self.node.keys():
            if numpy.random.random() <= self._p_infected:
                self.node[n][self.DYNAMICAL_STATE] = self.INFECTED
            else:
                self.node[n][self.DYNAMICAL_STATE] = self.SUSCEPTIBLE
        for (n, m, data) in self.edges_iter(data = True):
            data[self.OCCUPIED] = False
    
    def _after(self):
        pass
    
    def set_timestep_rate(self):
        '''Calculate the minimum time-jump. Choose the maximum from the recovery rate
        or the max node degree * the infection rate. Timestep will be 1 over this value'''
        max_degree = max(self.degree().values())
        return max(max_degree*self._p_infect, self._p_recover)
     
    def at_equilibrium( self ):
        '''SIR dynamics is at equilibrium if there are no more
        infected nodes left in the network or if we've exceeded
        the default simulation length.
        
        returns: True if the model has stopped'''
        
        '''if(t%1000 == 0.0):
            print("Processing timestep ",t)'''
        
        if self.CURRENT_TIMESTEP >= self._time_limit:
            return True
        else:
            return (len(self.POPULATION[self.INFECTED]) == 0)
            
    
    def asyn_node_action(self, node = 0, dt = 0.0, r = 0.0):
        
        # If chosen node is infected
        if(self.node[node][self.DYNAMICAL_STATE] == self.INFECTED):
            # If random number lower than Number of nodes x rate recovery * timestep
            if(r < self.order()*self._p_recover*dt):
                # Node recovers
                self.update_node(node,self.INFECTED,self.RECOVERED)
                return True
        # If chosen node is susceptible
        elif(self.node[node][self.DYNAMICAL_STATE] == self.SUSCEPTIBLE):
            infected_neighbours = 0
            # Count neighbours - TODO - better way to do this? Have to count the neighbours each time
            for (_, mp, datap) in self.edges_iter(node, data = True):
                if self.node[mp][self.DYNAMICAL_STATE] == self.INFECTED:
                        infected_neighbours += 1
            # If random number lower than Number of nodes x rate infection * timestep * infected_neighbours
            if(r < self.order()*self._p_infect*dt*infected_neighbours):
                # Node gets infected
                self.update_node(node,self.SUSCEPTIBLE,self.INFECTED)
                return True
        
        return False

