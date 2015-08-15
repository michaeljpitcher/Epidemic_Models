
# coding: utf-8

# In[9]:

from GraphWithDynamics import *


# In[10]:

class GraphWithAsynchronousDynamics(GraphWithDynamics):
    '''A graph with a dynamics that runs asynchronously,
    by calculating the minimum time an event could occur
    in, then incrementing by that amount if an event did 
    indeed occur and what it was.'''

    # Timestep to increment
    DT = 0
    
    def __init__( self, graph = None, time_limit = 10000, states = [], rates = dict() ):
        '''Create a graph, optionally with nodes and edges copied from
        the graph given.
        
        g: graph to copy (optional)'''
        GraphWithDynamics.__init__(self, graph, time_limit, states = states, rates = rates)
        
    def model( self, n ):
        '''The dynamics function that's run over the network. This
        is a placeholder to be re-defined by sub-classes.
        
        n: the node being simulated'''
        raise NotYetImplementedError('model()')

    def set_timestep_rate( self ):
        '''To calculate the timestep jump, need to know the minimum time needed for an event to happen. Timestep
        will be calculated by 1 over the value returned here.
        '''
        raise NotYetImplementedError('model()')
        

    def _dynamics( self ):
        '''Asynchronous dynamics. We apply _dynamics_step() at each timestep (calculated based on infection / recovery rates)
        and then check for completion using at_equilibrium().
        
        returns: a dict of simulation properties'''

        # Initialise values
        rc = dict()
        timestepEvents = 0
        events = 0
        eventDist = dict()
        
        # Calculate the maximum rate
        max_trans_rate = self.set_timestep_rate()
        
        # Timestep = 1 over number of nodes by maximum transition rate
        self.DT = 1.0/(self.order()*max_trans_rate);
        
        # Run continuously until equilibrium reached
        while True:
            
            #Pick a node at random
            n = numpy.random.randint(self.order())
            # Random number for probability distribution
            r = numpy.random.random()
            # Run an action on the node dependent on it's current state
            event = self.asyn_node_action(n, self.DT, r)
            
            # Increment the timestep, if nothing happened, don't update the population distribution history
            self.increment_timestep(self.DT, event)
            
            # test for termination
            if self.at_equilibrium():
                break
            
        # return the simulation-level results
        rc['timesteps'] = self.CURRENT_TIMESTEP
        rc['events'] = events
        rc['event_distribution'] = eventDist
        
        # compute the limits and means
        cs = sorted(networkx.connected_components(self.skeletonise()), key = len, reverse = True)
        max_outbreak_size = len(cs[0])
        max_outbreak_proportion = (max_outbreak_size + 0.0) / self.order()
        mean_outbreak_size = numpy.mean([ len(c) for c in cs ])
        
        # add parameters and metrics for this simulation run
        rc['number_of_nodes'] = self.order(),
        rc['mean_outbreak_size'] = mean_outbreak_size,
        rc['max_outbreak_size'] = max_outbreak_size,
        rc['max_outbreak_proportion'] = max_outbreak_proportion
        
        return rc
    
    def asyn_node_action(self, node = 0, dt = 0.0, r = 0.0):
        '''Internal function defining what to do for the node'''
        raise NotYetImplementedError('asyn_node_action()')

