
# coding: utf-8

# In[1]:

from GraphWithSynchronousDynamics import *


# In[2]:

class SIRSynchronousDynamics(GraphWithSynchronousDynamics):
    '''A graph with a particular SIR dynamics. We use probabilities
    to express infection and recovery per timestep, and run the system
    using synchronous dynamics.'''
    
    # the possible dynamics states of a node for SIR dynamics
    SUSCEPTIBLE = 'susceptible'
    INFECTED = 'infected'
    RECOVERED = 'recovered'
    
    
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
        GraphWithSynchronousDynamics.__init__(self, time_limit = time_limit, graph = graph, states = states, rates = rates)
        self.p_infected = p_infected
        self.p_infect = p_infect
        self.p_recover = p_recover
        
            
    def _before( self ):
        '''Seed the network with infected nodes, and mark all edges
        as unoccupied by the dynamics.'''
        self._infected = []       # in case we re-run from a dirty intermediate state
        for n in self.node.keys():
            if numpy.random.random() <= self.p_infected:
                self.node[n][self.DYNAMICAL_STATE] = self.INFECTED
            else:
                self.node[n][self.DYNAMICAL_STATE] = self.SUSCEPTIBLE
        for (n, m, data) in self.edges_iter(data = True):
            data[self.OCCUPIED] = False

    def _after(self):
        '''No processing to do after completion'''
        pass
            
    def _dynamics_step( self ):
        '''Optimised per-step dynamics that only runs the dynamics at infected
        nodes, since they're the only places where state changes originate. At the
        end of each timestep we re-build the infected node list.
        returns: the number of events that happened in this timestep'''
        events = 0
        
        # run model dynamics on all infected nodes
        for n in self.POPULATION[self.INFECTED]:
            events += self.model(n)
        return events
            
    def model( self, node_selected ):
        '''Apply the SIR dynamics to node n. From the re-definition of dynamics_step()
        we already know this node is infected.

        n: the node
        returns: the number of changes made'''
        events = 0
        
        # infect susceptible neighbours with probability pInfect
        for (_, neighbour, data) in self.edges_iter(node_selected, data = True):
            if self.node[neighbour][self.DYNAMICAL_STATE] is self.SUSCEPTIBLE:
                if numpy.random.random() <= self.p_infect:
                    events += 1
                    
                    # infect the node
                    self.update_node(neighbour,self.SUSCEPTIBLE,self.INFECTED)
                        
                    # label the edge we traversed as occupied
                    data[self.OCCUPIED] = True
    
        # recover with probability pRecover
        if numpy.random.random() <= self.p_recover:
            # recover the node
            events = events + 1
            self.update_node(node_selected,self.INFECTED,self.RECOVERED)
                
        return events
            
    def at_equilibrium( self ):
        '''SIR dynamics is at equilibrium if there are no more
        infected nodes left in the network or if we've exceeded
        the default simulation length.
        
        returns: True if the model has stopped'''
        
        if self.CURRENT_TIMESTEP >= self._time_limit:
            return True
        else:
            return (len(self.POPULATION[self.INFECTED]) == 0)

