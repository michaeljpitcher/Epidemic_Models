
# coding: utf-8

# In[1]:

from GraphWithDynamics import *


# In[2]:

class GraphWithStochasticDynamics(GraphWithDynamics):
    '''A graph with a dynamics that runs stochastically, whereby
    the next expected event to happen is calculated based on the 
    transition rates and the model then jumps to that step and 
    performs an action on a node'''
        
    def __init__( self, graph = None, time_limit = 10000, states = [], rates = dict() ):
        '''Create a graph, optionally with nodes and edges copied from
        the graph given.
        
        grpah: graph to copy (optional)
        time_limit: maximum number of timesteps(optional)'''
        GraphWithDynamics.__init__(self, graph, time_limit, states = states, rates = rates)

    def transitions( self, t ):
        '''Return the transition vector, a sequence of (r, f) pairs
        where r is the rate at which a transition happens and
        f is the transition function called to make it happen. Note that
        it's a rate we want, not a probability.
        
        It's important that the transitions always come in the same order
        in the vector, even though the rates (and indeed functions) can
        change over time.
        
        t: timestep for which we want the transitions
        returns: the transition vector'''
        raise NotYetImplementedError('transitions()')
        
    def _dynamics( self ):
        '''Stochastic dynamics.
        
        returns: a dict of simulation properties'''
        properties = dict()
        
        # set up the priority list
        transitions = self.transitions()
        pr = range(len(transitions))
        
        # run the dynamics
        events = 0
        
        # Run continuously until equilibrium reached
        while True:
                
            # pull the transition dynamics
            transitions = self.transitions()
            # Sum up total transition rate
            tot = 0.0
            for (r, _) in transitions:
                tot = tot + r
                
            # calculate the timestep delta
            x = numpy.random.random()
            tau = (1.0 / tot) * math.log(1.0 / x)
            
            # calculate which transition happens 
            # Generate random number between 0 and tot
            x = numpy.random.random() * tot
            k = 0
            # Pull the first transition
            (xs, f) = transitions[pr[k]]
            while xs < x:
                # Keep processing transitions until the random number is hit
                k = k + 1
                (xsp, f) = transitions[pr[k]]
                xs = xs + xsp
            
            # perform the transition
            f(self.CURRENT_TIMESTEP)
            
            # if we used a low-priority transition, swap it up the priority queue
            if k > 0:
                p = pr[k - 1]
                pr[k - 1] = pr[k]
                pr[k] = p
            
            # Record the event
            self._event_dist[self.CURRENT_TIMESTEP] = 1
            # Increment the timestep by the delta
            self.increment_timestep(tau)
            # Increment event total
            events += 1
            
            # check for termination
            if self.at_equilibrium():
                break
        
        # compute the limits and means
        cs = sorted(networkx.connected_components(self.skeletonise()), key = len, reverse = True)
        max_outbreak_size = len(cs[0])
        max_outbreak_proportion = (max_outbreak_size + 0.0) / self.order()
        mean_outbreak_size = numpy.mean([ len(c) for c in cs ])
        
        properties['mean_outbreak_size'] = mean_outbreak_size,
        properties['max_outbreak_size'] = max_outbreak_size,
        properties['max_outbreak_proportion'] = max_outbreak_proportion
        
        # complete statistics
        properties['timesteps'] = self.CURRENT_TIMESTEP
        properties['events'] = events
        return properties

