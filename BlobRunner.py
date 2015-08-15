
# coding: utf-8

# In[1]:

from SIRStochasticDynamicsDisconnect import *
from SIRStochasticDynamicsRewire import *
from SIRStochasticDynamicsRewireDegree import *
from SIRStochasticDynamicsRewireNeighbour import *

from IPython.parallel import Client
import time
import csv


# In[2]:

def make_simulation(N, M, desc, model_type, repetitions = 1, offset = 0 ):
    '''Return a function to populate and run the simulation on a graph
    with dynamics. If there is more than one repetition, return the
    average of the outbreak parameters.
    
    N: the network size
    p: model function for degree distribution
    desc: description of distrbution, used for filename generation
    repetitions: (optional) number of repetitions (defaults to 1)
    offset: (optional) start repetition (defaults to 0)'''
    
    def run_simulation( g ):
        
        # Write the results to a csv file
        filename = model_type + '_results.csv'

        #with open(filename, 'wb') as csvfile:
        #        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #        writer.writerow(['p_infect'] + ['p_rewire'] + ['p_recover'] + ['repetitions'] + 
        #                        ['start_time'] + ['end_time'] + ['avg_time_data'] + ['avg_node_data'])
        
        start = time.clock()
        time_results = []
        node_results = []
        r_infinities = []
        for rep in xrange(repetitions):
            start_rep = time.clock()
            # build the network topology using the given degree distribution
            g.reset()
            
            g.rebuild_barabasi_albert(N,M)
            
            # run the simulation dynamics
            steps = g.dynamics()

            # compute the (partial) results
            time_results.append(steps['peak_infection'][0])
            node_results.append(steps['peak_infection'][1])
            r_infinities.append(steps['r_infinity'])
              
        end = time.clock()
        
        with open(filename, 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([N] + [g.p_infect] + [g.p_rewire] + [g.p_recover] + [repetitions] + 
                                [start] + [end] + [numpy.mean(time_results)] + [numpy.mean(node_results)] +
                                [numpy.mean(r_infinities)])
        
        # construct metadata to wrap-up repetition results
        r = dict()
        r['nodes'] = N
        r['p_infect'] = g.p_infect
        r['p_rewire'] = g.p_rewire
        r['p_recover'] = g.p_recover
        r['repetitions'] = repetitions
        r['start_time'] = start
        r['end_time'] = end
        r['avg_time_data'] = numpy.mean(time_results)
        r['avg_node_data'] = numpy.mean(node_results)
        r['r_infinity'] = numpy.mean(r_infinities)
        
        return r
    
    return run_simulation


# In[3]:

def blob_runner( reps = 1, timelim = 10000, numnodes = 5000, seed = 3, startinfected = 0.01, pinf_min = 0.00, pinf_max = 0.02, 
                pinf_num = 10, prew_min = 0.00, prew_max = 0.00, prew_num = 10, prec_min = 0.00, prec_max = 0.00, 
                prec_num = 10, set_alpha = 2, model_type = 'BASE'):
    
    # IPython profile for our remote cluster
    cluster_name = "blob"

    # connect to cluster
    cluster = Client(profile = cluster_name)
    #cluster = Client()
    print("Cluster has {n} engines available".format(n = len(cluster[:])))

    # Add the location of the files on the blob server to python path
    #def parallel(x):
    #    import sys
    #    sys.path.append('//home//mjp22//08_Blob')
    #    return sys.path
        
    #k = cluster[:].map_sync(parallel,range(1))
    
    #print k
    
    d = cluster[:]
    
    # set up imports on cluster machines
    with d.sync_imports():
        import time
        import math
        import numpy
        import mpmath
        import networkx
        import dill
        import collections
        import operator
        import os.path
        import csv
    
    # use Dill as pickler
    d.use_dill()

    # load-balance work across the available compute nodes
    view = cluster.load_balanced_view()
    
    def parallel(x):
    
        filename = model_type + '_results.csv'
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['nodes'] + ['p_infect'] + ['p_rewire'] + ['p_recover'] + ['repetitions'] + 
                            ['start_time'] + ['end_time'] + ['avg_time_data'] + ['avg_node_data'] + ['r_infinity'])
        
        return 'done'
    
    print 'CSV file written: ', view.map_sync(parallel, range(1))
    
    # set up simulation parameters
    repetitions = reps
    time_limit = timelim

    # network parameters
    N = numnodes
    M = seed
    alpha = set_alpha
    p_infected = startinfected
    
    # Parameter spaces
    p_infects = numpy.linspace(pinf_min, pinf_max, endpoint = True, num = pinf_num)
    p_rewires = numpy.linspace(prew_min, prew_max, endpoint = True, num = prew_num)
    p_recovers = numpy.linspace(prec_min, prec_max, endpoint = True, num = prec_num)
    
    print 'p_infects: ', p_infects
    print 'p_rewires: ', p_rewires
    print 'p_recovers: ', p_recovers

    simulations = []
    
    for pi in p_infects :
        for prew in p_rewires:
            for prec in p_recovers:
                
                if(model_type == 'REWIRE'):
                    simulations.append(SIRStochasticDynamicsRewire(time_limit = time_limit, p_infected = p_infected, 
                                         p_infect = pi, p_recover = prec, p_rewire = prew))
                elif(model_type == 'DISCONNECT'):
                    simulations.append(SIRStochasticDynamicsDisconnect(time_limit = time_limit, p_infected = p_infected, 
                                         p_infect = pi, p_recover = prec, p_rewire = prew))
                elif(model_type == 'NEIGHBOUR'):
                    simulations.append(SIRStochasticDynamicsRewireNeighbour(time_limit = time_limit, p_infected = p_infected, 
                                         p_infect = pi, p_recover = prec, p_rewire = prew))
                elif(model_type == 'DEGREE'):
                    simulations.append(SIRStochasticDynamicsRewireDegree(time_limit = time_limit, p_infected = p_infected, 
                                         p_infect = pi, p_recover = prec, p_rewire = prew))
                    
    
   
    sim = make_simulation(N, M, desc = '', repetitions = repetitions, model_type = model_type)
    
    print 'Beginning simulations...'
    
    rc = view.map_async(sim, simulations)
    
    # wait for simulations to complete
    #rc.wait()
    
    elapsed = 0
    while True:
        msgset = set(rc.msg_ids)
        completed = len(msgset.difference(cluster.outstanding))
        pending = len(msgset.intersection(cluster.outstanding))
        print 'After %d secs: %d complete, %d outstanding' % (elapsed, completed, pending)
        if rc.ready():
            break
        time.sleep(60)
        elapsed += 60
    
    # Write the results to a csv file locally and on the server
    filename = '.\\output\\' + model_type + '_results.csv'

    with open(filename, 'wb') as csvfile:
        awriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        awriter.writerow(rc.result[0].keys())
        for i in xrange(0,len(rc.result)):
            awriter.writerow(rc.result[i].values())
    
    print 'Simulations complete'
    

