
# coding: utf-8

# In[2]:

from GraphWithDynamics import *


# In[3]:

class GraphWithSynchronousDynamics(GraphWithDynamics):
    '''A graph with a dynamics that runs synchronously,
    incrementing the timestep by 1 each time and running
    through every single node in the network appying
    the necessary dynamics.'''
        
    def __init__( self, graph = None, time_limit = 10000, states = [], rates = dict() ):
        '''Create a graph, delegates to superclass'''
        GraphWithDynamics.__init__(self, graph, time_limit, states = states, rates = rates)
        
    def model( self, node ):
        '''The dynamics function that's run over the network. This
        is a placeholder to be re-defined by sub-classes.
        
        node: the node being simulated'''
        raise NotYetImplementedError('model()')
        
    def _dynamics_step( self ):
        '''Run a single step of the model over the network.
        
        returns: the number of dynamic events that happened in this timestep'''
        #Initialise events
        events = 0
        # Run through every node in the network
        for i in self.node.keys():
            # Run the model, add any event done to the events total
            events = events + self.model(i)
        return events    
    
    def _dynamics( self ):
        '''Synchronous dynamics. We apply _dynamics_step() at each timestep
        and then check for completion using at_equilibrium().
        
        returns: a dict of simulation properties'''
        
        # initialise properties, which records the statistics to be returned to user
        properties = dict()

        # initialise event counter
        events = 0
        timestep_events = 0
        
        # Run continuously until equilibrium reached
        while True:
            # run a step
            new_events = self._dynamics_step()
            # If events have happened, update
            if new_events > 0:
                events += new_events
                timestep_events += 1
                # Record event distribution history
                self._event_dist[self.CURRENT_TIMESTEP] = new_events
        
            # Increment timestep (synchronous maps each step, so increment by 1)
            self.increment_timestep(1)
            
            # test for termination
            if self.at_equilibrium():
                break
        
        # Calculate outbreak sizes
        cs = sorted(networkx.connected_components(self.skeletonise()), key = len, reverse = True)
        max_outbreak_size = len(cs[0])
        max_outbreak_proportion = (max_outbreak_size + 0.0) / self.order()
        mean_outbreak_size = numpy.mean([ len(c) for c in cs ])
        
        # add parameters and metrics for this simulation run
        properties['number_of_nodes'] = self.order(),
        properties['mean_outbreak_size'] = mean_outbreak_size,
        properties['max_outbreak_size'] = max_outbreak_size,
        properties['max_outbreak_proportion'] = max_outbreak_proportion
        
        # return the simulation-level results
        properties['timesteps'] = self.CURRENT_TIMESTEP
        properties['events'] = events
        properties['timesteps_with_events'] = timestep_events
        
        # Return the statistics
        return properties

