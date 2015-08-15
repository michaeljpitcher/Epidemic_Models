
# coding: utf-8

# In[4]:

from GraphWithStochasticDynamics import *
import operator
import csv


# In[5]:

class SIRStochasticDynamicsRewireDegree(GraphWithStochasticDynamics):
    '''An SIR dynamics with stochastic simulation.'''

    # the possible dynamics states of a node for SIR dynamics
    SUSCEPTIBLE = 'susceptible'
    INFECTED = 'infected'
    RECOVERED = 'recovered'
    degree_dict = dict()
    
    # list of SI edges connecting a susceptible to an infected node
    _si = []
        
    def __init__( self, time_limit = 10000, p_infect = 0.0, p_recover = 1.0, p_infected = 0.0, graph = None, p_rewire = 0.0 ):
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
        rates['p_rewire'] = p_rewire
        GraphWithStochasticDynamics.__init__(self, time_limit = time_limit, graph = graph, states = states, rates = rates)
        self.p_infected = p_infected
        self.p_infect = p_infect
        self.p_recover = p_recover
        self.p_rewire = p_rewire
        
        for i in self.nodes():
            self.degree_dict[i] = self.degree(i)
            
        
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
        
        returns: True if the model has stopped
        
        Extension: stop once the infection population drops below 80% of the current maximum
        '''
        if len(self.POPULATION[self.INFECTED]) < (0.8 * max(self._pop_dist[self.INFECTED].values())):
            self.STATISTICS['peak_infection'] = max(self._pop_dist[self.INFECTED].iteritems(), key=operator.itemgetter(1))
            return True
        
        if self.CURRENT_TIMESTEP >= self._time_limit:
            self.STATISTICS['peak_infection'] = max(self._pop_dist[self.INFECTED].iteritems(), key=operator.itemgetter(1))
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
    
    def rewire( self ):
        '''Cause a node to rewire.'''
       
        # choose an SI edge
        i = numpy.random.randint(len(self._si))
        # remove the edge from the si list and from the overall graph structure
        (n, m, data) = self._si.pop(i)
        
        previous_degree = self.degree(n)
        
        self.remove_edges_from([(n, m)])
        self.degree_dict[n] -= 1
        
        # Add a link to a new node
        
        # Clone the list of susceptible and recovered nodes - these will be the candidates for the new link
        candidates = list(self.POPULATION[self.SUSCEPTIBLE])
        candidates.extend(list(self.POPULATION[self.RECOVERED]))                
                        
        # Remove the current node (don't want self loops)
        if m in candidates:
            candidates.remove(m)
        
        # Remove neighbours from candidates        
        neighbours = self.neighbors(m)
        candidates = [c for c in candidates if c not in neighbours]
        
        # Create a dict to store how far away each candidate is in terms of degree from the disconnected node
        pref_cand = dict()
        for c in candidates:
            pref_cand[c] = abs(self.degree_dict[c] - previous_degree)
        
        # Reduce the candidates down to only those who are nearest to the disconncected node in terms of degree
        min_distance = min(pref_cand.values()) 
        candidates = [c for c in pref_cand if pref_cand[c] == min_distance]
        
        # Pick a candidate
        if(len(candidates) >= 1):
            new_neighb_index = numpy.random.randint(len(candidates))
            self.add_edge(m, candidates[new_neighb_index])
            self.degree_dict[candidates[new_neighb_index]] += 1
            
        
    def transitions( self ):
        '''Return the transition vector for the dynamics.
        
        returns: the transition vector'''
        
        # transitions are expressed as rates, whereas we're specified
        # in terms of probabilities, so we convert the latter to the former.
        return [ (len(self._si) * self.p_infect,        lambda t: self.infect()),
                 (len(self.POPULATION[self.INFECTED]) * self.p_recover, lambda t: self.recover()),
                 (len(self._si) * self.p_rewire,        lambda t: self.rewire())]
            

