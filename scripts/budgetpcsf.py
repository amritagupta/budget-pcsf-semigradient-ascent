import collections, copy, csv, itertools, math, operator, os, pickle, sys, time
import igraph
import networkx as nx
import numpy as np
import gurobipy as grb
from networkx.utils import UnionFind
from Queue import PriorityQueue
import multiprocessing
from functools import partial


class PriorityQueueEntry(object):
    def __init__(self, priority, data):
        self.data = data
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority



def evaluate_solution(solution_elements, vertices, demand):
    # Construct the solution graph
    Gsol = nx.Graph(directed=False)
    Gsol.add_nodes_from(vertices)
    for (i,j) in solution_elements:
        Gsol.add_edge(i,j)

    # Find its connected components
    connected_components = [cc for cc in nx.connected_components(Gsol)]
    
    # Sum satisfied demand within pairs of nodes in the same connected component
    satisfied_demand = 0
    for i,cc in enumerate(connected_components):
        cc_nodes = list(cc) # list(cc.vs)
        for n in cc_nodes:
            for m in cc_nodes:
                satisfied_demand += demand[(n,m)]
    return satisfied_demand, Gsol



def evaluate_solution_from_graph(Gsol, vertices, demand):
    # Find its connected components
    connected_components = [cc for cc in nx.connected_components(Gsol)]
    
    # Sum satisfied demand within pairs of nodes in the same connected component
    satisfied_demand = 0
    for i,cc in enumerate(connected_components):
        cc_nodes = list(cc) # list(cc.vs)
        for n in cc_nodes:
            for m in cc_nodes:
                satisfied_demand += demand[(n,m)]
    return satisfied_demand



def budget_pcsf_greedy(G, cost_E, demand, budget, by_ratio=True, instancestring='', logfiledir='', appendlog = '', resultfiledir='', verbose=True):
    '''Solves budget-constrained prize-collecting Steiner forest problem using simple greedy algorithm.
    Args:
        G: igraph object representing the graph
        cost_E: dictionary containing costs associated with each edge
        demand: dictionary with tuple keys (i,j) representing profit for connecting vertex i to j in G
        budget: float representing maximum amount to spend selecting edges
        by_ratio: bool representing whether to order edges by (benefit/cost) ratio (True) or by benefit (False)
        instancestring: string for naming files associated with given G and budget
        logfiledir: string specifying logfile directory
        resultfiledir: string specifying resultfile directory
    '''
    if appendlog == '': # standalone run
        assert instancestring is not '', 'budget_pcsf_greedy: need to provide an instance string for standalone run'
        assert logfiledir is not '', 'budget_pcsf_greedy: need to specify a log file directory for standalone run'
        assert resultfiledir is not '', 'budget_pcsf_greedy: need to specify a result file directory for standalone run'
        solverunid = np.random.randint(10000)
        logfilename = logfiledir+instancestring+'_'+str(solverunid)+'.log'
        sys.stdout = open(logfilename, 'w')
    else: # subroutine run
        logfilename = appendlog
        sys.stdout = open(logfilename, 'a')

    # Construct a priority queue for the candidate edges
    candidate_edges = PriorityQueue()
    V = [v.index for v in G.vs()]
    E = G.get_edgelist()
    for e in E:
        if cost_E[e] < budget:
            improvement = float('inf')
            if by_ratio:
                delta = improvement/cost_E[e]
            else:
                delta = improvement
            pqedge = PriorityQueueEntry(-delta, {'e':e, 'cost':cost_E[e], 'valid':False}) # delta negative because python PriorityQueue returns lowest priority element
            candidate_edges.put(pqedge)
    if verbose:
        print("Initial number of candidate edges: %d"%candidate_edges.qsize())
    
    # Create a copy of the graph with none of the candidate edges added yet
    G_prime = copy.deepcopy(G)
    G_prime = G_prime.__sub__(G.es) # this is the graph with no edges added yet
    if verbose:
        print "Starting initial shortest path computations"
    tic = float(time.time())
    baseline_shortest_paths = G_prime.shortest_paths(source=None, target=None)
    baseline_shortest_paths = np.array(baseline_shortest_paths)
    baseline_shortest_paths_inf = np.isinf(baseline_shortest_paths)
    toc = float(time.time())
    if verbose:
        print "Finished initial shortest path computations in %0.4f seconds" % (toc-tic)
    total_demand = sum([demand[(i,j)] for i in V for j in V])
    current_shortest_paths = baseline_shortest_paths
    current_shortest_paths_inf = baseline_shortest_paths_inf
    current_num_feasible_trips = sum([demand[(i,j)] for i in V for j in V if baseline_shortest_paths_inf[i,j]==False])
    current_num_infeasible_trips = sum([demand[(i,j)] for i in V for j in V if baseline_shortest_paths_inf[i,j]==True])
    if verbose:
        print("Total trip demand: %d"%total_demand)
        print("Current number of feasible trips: %d"%current_num_feasible_trips)
        print("Current number of infeasible trips: %d"%current_num_infeasible_trips)

    connected_components = G_prime.components()
    connected_components_unionfind = UnionFind()
    for i,cc in enumerate(connected_components):
        cc_nodes = list(cc) # list(cc.vs)
        cc_parent_node = cc_nodes[0] #cc_nodes[0]["id"]
        connected_components_unionfind[cc_parent_node]
        for n in cc_nodes:
            connected_components_unionfind.union(cc_parent_node, n) #connected_components_unionfind.union(cc_parent_node, n["id"])
    if verbose:
        print('Graph contains %d connected components'%(i+1))
    
    timer = time.time()
    # Begin greedily adding edges from candidate edges
    edges_added = 0
    num_evaluations = 0
    remaining_budget = budget
    while (remaining_budget > 0) and (not candidate_edges.empty()): # can still add edges
        # set all candidate edges to invalid
        temp_store = []
        while not candidate_edges.empty():
            elem = candidate_edges.get()
            elem.data['valid'] = False
            temp_store.append(elem)
        while len(temp_store) > 0:
            candidate_edges.put(temp_store[0])
            temp_store = temp_store[1:]

        # get the next edge to add
        while True:            
            next_edge = candidate_edges.get()

            if next_edge.data['valid']: # add the edge
                edges_added += 1
                remaining_budget -= next_edge.data['cost']
                e = next_edge.data['e']
                if verbose and edges_added%10 == 0:
                    print('Added %d edges, budget of %0.4f remaining'%(edges_added, remaining_budget))
                G_prime.add_edge(e[0], e[1], weight=next_edge.data['cost'])
                connected_components_unionfind.union(e[0], e[1])

                # remove edges that are no longer within budget or no longer between different connected components
                temp_store_2 = []
                num_too_expensive = 0
                num_same_cc = 0
                while not candidate_edges.empty():
                    elem_2 = candidate_edges.get()
                    e2 = elem_2.data['e']
                    if (elem_2.data['cost'] <= remaining_budget and connected_components_unionfind[e2[0]] != connected_components_unionfind[e2[1]]):
                        temp_store_2.append(elem_2)
                    elif (elem_2.data['cost'] > remaining_budget):
                        num_too_expensive += 1
                    elif (connected_components_unionfind[e2[0]] == connected_components_unionfind[e2[1]]):
                        num_same_cc += 1
                while len(temp_store_2) > 0:
                    candidate_edges.put(temp_store_2[0])
                    temp_store_2 = temp_store_2[1:]
                if verbose:
                    print("1 edge added, %d edges eliminated due to cost, %d edges eliminated due to connected components"%(num_too_expensive, num_same_cc))
                    print("%d edges left as candidates"%candidate_edges.qsize())

                # Recompute the effect of adding this edge -- adds time due to function evaluation calls
                current_shortest_paths = G_prime.shortest_paths(source=None, target=None)
                current_shortest_paths = np.array(current_shortest_paths)
                current_shortest_paths_inf = np.isinf(current_shortest_paths)
                current_num_feasible_trips = sum([demand[(i,j)] for i in V for j in V if current_shortest_paths_inf[i,j]==False])
                current_num_infeasible_trips = sum([demand[(i,j)] for i in V for j in V if current_shortest_paths_inf[i,j]==True])
                break

            else: # validate it
                num_evaluations += 1
                if verbose and num_evaluations%10 == 0:
                    print('\tPerformed %d evaluations'%num_evaluations)
                G_temp = G_prime.copy()
                G_temp.add_edge(next_edge.data['e'][0], next_edge.data['e'][1], cost=next_edge.data['cost'])
                new_shortest_paths = G_temp.shortest_paths(source=None, target=None)
                new_shortest_paths = np.array(new_shortest_paths)
                new_shortest_paths_inf = np.isinf(new_shortest_paths)
                new_num_feasible_trips = sum([demand[(i,j)] for i in V for j in V if new_shortest_paths_inf[i,j]==False])
                new_num_infeasible_trips = sum([demand[(i,j)] for i in V for j in V if new_shortest_paths_inf[i,j]==True])
                improvement = current_num_infeasible_trips - new_num_infeasible_trips
                if by_ratio:
                    delta = improvement/next_edge.data['cost']
                else:
                    delta = improvement
                next_edge.priority = -delta
                next_edge.data['valid'] = True
                candidate_edges.put(next_edge)
    if (appendlog == '') or verbose:
        print('Greedy optimization took %0.2f seconds'%(time.time()-timer))
    sol_edges = G_prime.get_edgelist()
    sol_cost = budget - remaining_budget
    sol_profit = current_num_feasible_trips
    
    if (appendlog == '') or verbose:
        print('Final solution cost: %0.4f'%(sol_cost))
        print('Final solution true value: %0.4f'%(sol_profit))

    # Save results if this is a standalone run
    if appendlog == '':
        resultsfilename = resultfiledir+instancestring+'_'+str(solverunid)+'.csv'
        with open(resultsfilename, 'wb') as results:
            writer = csv.writer(results, delimiter=' ')
            writer.writerow(['e, x_e'])
            for (i,j) in E:
                if (i,j) in sol_edges:
                    writer.writerow([(i,j), np.abs(1)])
                else:
                    writer.writerow([(i,j), np.abs(0)])
    
    return(sol_edges, sol_cost, sol_profit)



def knapsack_mip(items, values, weights, capacity, verbose=True):
    '''MIP for 0-1 knapsack problem. Writes an ILP and calls Gurobi to solve it.
    
    Inputs:
    items    - list of edges that are candidates for selection
    values   - dict containing the modular value of each edge
    weights  - dict containing the cost of each edge
    capacity - budget for selecting edges
    
    Returns:
    knapsack_contents            - set of selected edges
    knapsack_contents_weight     - total cost of selected edges
    knapsack_contents_value      - total value of selected edges    
    '''
    if verbose:
        print('Knapsack capacity: %0.6f'%capacity)
    # Create gurobi variables
    mc_timer = time.time()
    m = grb.Model('knapsack_subproblem')
    if not verbose:
        m.setParam('OutputFlag', False)
    x = m.addVars(items, name="x", vtype=grb.GRB.BINARY)
    m.update()
    
    # Add budget constraint
    m.addConstr(x.prod(weights) <= capacity)
    
    m.setObjective(x.prod(values), grb.GRB.MAXIMIZE)
    if verbose:
        print('Model construction took %0.2f seconds'%(time.time() - mc_timer))
    # m.update()
    timer = time.time()
    m.optimize()
    if verbose:
        print('Solving took %0.2f seconds'%(time.time()-timer))
    
    knapsack_contents = set()
    for i in items:
        if np.abs(x[i].x - 1.0) < 1e-9:
            knapsack_contents.add(i)
    knapsack_contents_weight = sum(weights[i] for i in knapsack_contents)
    knapsack_contents_value = m.objVal

    if verbose:
        print('knapsack contents weight: %0.6f'%knapsack_contents_weight)

    print('diff contents weight and capacity: %0.6f'%(0.0000001 + capacity - knapsack_contents_weight))
    assert knapsack_contents_weight <= (capacity + 0.000001), 'knapsack_mip: selected items (%0.6f) exceed capacity (%0.6f)'%(knapsack_contents_weight,capacity)

    return(knapsack_contents, knapsack_contents_weight, knapsack_contents_value)


def knapsack_avoid_cycles_mip(items, values, weights, capacity, verbose=True):
    '''MIP for 0-1 knapsack problem on edges while avoiding cycles. Writes
    an ILP and calls Gurobi to solve it.
    
    Inputs:
    items    - list of edges that are candidates for selection
    values   - dict containing the modular value of each edge
    weights  - dict containing the cost of each edge
    capacity - budget for selecting edges
    
    Returns:
    knapsack_contents            - set of selected edges
    knapsack_contents_weight     - total cost of selected edges
    knapsack_contents_value      - total value of selected edges    
    '''
    # Get cycle basis
    cb_timer = time.time()
    G = nx.Graph(directed=False)
    for e in items:
        G.add_edge(e[0],e[1])
    simple_cycles = nx.cycle_basis(G)
    if verbose:
        print('Getting all cycles possible in current set of candidates took %0.2f seconds'%(time.time() - cb_timer))

    # Get edges in each simple cycle
    ce_timer = time.time()
    simple_cycle_edges = dict.fromkeys(range(len(simple_cycles)))
    for cno, cycle in enumerate(simple_cycles):
        temp_cycle = tuple(copy.deepcopy(cycle))
        temp_cycle = temp_cycle + (cycle[0],)
        possible_cycle_edges = [(temp_cycle[i], temp_cycle[i+1]) for i in range(len(temp_cycle)-1)]
        actual_cycle_edges = [(eij[0],eij[1]) if (eij[0],eij[1]) in items else (eij[1],eij[0]) for eij in possible_cycle_edges]
        simple_cycle_edges[cno] = actual_cycle_edges
    if verbose:
        print('Getting the actual edges participating in each possible cycle in current candidate set took %0.2f seconds'%(time.time() - ce_timer))
    
    # Create gurobi variables
    mc_timer = time.time()
    m = grb.Model('knapsack_subproblem')
    if not verbose:
        m.setParam( 'OutputFlag', False )
    x = m.addVars(items, name="x", vtype=grb.GRB.BINARY)
    m.update()
    
    # Add budget constraint
    m.addConstr(x.prod(weights) <= capacity)
    # m.update()
    
    # Add simple cycle constraint: in each simple cycle, at most |cycle|-1 edges can be selected
    for sc in simple_cycle_edges:
        m.addConstr(grb.quicksum(x[(i,j)] for (i,j) in simple_cycle_edges[sc]) <= len(simple_cycle_edges[sc]) - 1)
    
    # Add induced cycle constraint: for each pair of simple cycles, at least 2 edges must remain unselected
    for sc1 in simple_cycle_edges:
        for sc2 in simple_cycle_edges:
            if sc1 != sc2:
                edges_in_sc1_and_sc2 = set(simple_cycle_edges[sc1] + simple_cycle_edges[sc2])
                m.addConstr(grb.quicksum((1 - x[(i,j)]) for (i,j) in edges_in_sc1_and_sc2) >= 2)
    # m.update()
    
    m.setObjective(x.prod(values), grb.GRB.MAXIMIZE)
    if verbose:
        print('Model construction took %0.2f seconds'%(time.time() - mc_timer))
    # m.update()
    timer = time.time()
    m.optimize()
    if verbose:
        print('Solving took %0.2f seconds'%(time.time()-timer))
    
    knapsack_contents = set()
    for i in items:
        if np.abs(x[i].x - 1.0) < 1e-9:
            knapsack_contents.add(i)
    knapsack_contents_weight = sum(weights[i] for i in knapsack_contents)
    knapsack_contents_value = m.objVal

    assert knapsack_contents_weight <= capacity + 0.0000000001, 'knapsack_avoid_cycles_mip: selected items exceed capacity'

    return(knapsack_contents, knapsack_contents_weight, knapsack_contents_value)


def fix_cycles_greedy(edge_set, cycle_set, edge_costs, edge_values, obj='budget', verbose=True):
    """ Greedy algorithm for breaking cycles.
    Inputs:
    edge_set    - set of edges that have been selected but contain cycles
    cycle_set   - list of cycles to break
    edge_costs  - dict containing cost of each edge
    edge_values - dict containing modular lower bound values of edges
    obj         - objective for fixing cycles, 'budget' or 'mlbobj'
    
    Returns:
    edges_to_remove - list of edges to remove to break cycles in cycle_set
    """
    edges_to_remove = list()
    fc_timer = time.time()

    assert cycle_set, 'fix_cycles_greedy: the set of cycles is possibly empty'
    
    current_cycle_set = copy.deepcopy(cycle_set)

    # Construct graph with edges participating in cycles
    edges_involved_in_cycles = set()
    for cycle in current_cycle_set:
        assert type(cycle) is tuple, 'fix_cycles_greedy: cycle_set should be a set of tuples'
        temp_cycle1 = copy.deepcopy(cycle)
        temp_cycle1 = temp_cycle1 + (cycle[0],)
        possible_cycle_edges1 = [(temp_cycle1[i], temp_cycle1[i+1]) for i in range(len(temp_cycle1)-1)]
        for possible_edge in possible_cycle_edges1:
            if (possible_edge[0], possible_edge[1]) in edge_set:
                edges_involved_in_cycles.add((possible_edge[0], possible_edge[1]))
            elif (possible_edge[1], possible_edge[0]) in edge_set:
                edges_involved_in_cycles.add((possible_edge[1], possible_edge[0]))
            else:
                raise ValueError('fix_cycles_greedy: cycle edge (%d,%d) not found in edge_set' % (possible_edge[0], possible_edge[1]))
    Gcycles = nx.Graph(directed=False)
    for eij in edges_involved_in_cycles:
        Gcycles.add_edge(eij[0],eij[1])
    cycles_in_Gcycles = nx.cycle_basis(Gcycles)

    if len(cycles_in_Gcycles) > 0:
        all_cycles_fixed = False
    
    passes = 0
    while not all_cycles_fixed: # remove the max cost edge in the remaining list of cycles, update list of cycles
        passes += 1
        if verbose:
            print('pass:%d'%passes)
        
        if obj == 'budget':
            # find most expensive edge out of all edges participating in cycles
            most_expensive_edge = (-np.inf, (None,None))
            for ce in edges_involved_in_cycles:
                edge_cost = edge_costs[ce]
                if edge_cost > most_expensive_edge[0]:
                    most_expensive_edge = (edge_cost, ce)
            # remove the chosen edge
            if most_expensive_edge[1] in edge_set:
                edges_to_remove.append(most_expensive_edge[1])
                Gcycles.remove_edge(most_expensive_edge[1][0], most_expensive_edge[1][1])
            else:
                raise ValueError('fix_cycles_greedy: edge marked for removal not found in edge set')
        
        elif obj == 'mlbobj':
            # find lowest value edge out of all edges participating in cycles
            lowest_value_edge = (np.inf, (None,None))
            for ce in edges_involved_in_cycles:
                edge_value = edge_values[ce]
                if edge_value < lowest_value_edge[0]:
                    lowest_value_edge = (edge_value, ce)
            # remove the chosen edge
            if lowest_value_edge[1] in edge_set:
                edges_to_remove.append(lowest_value_edge[1])
                Gcycles.remove_edge(lowest_value_edge[1][0], lowest_value_edge[1][1])
            else:
                raise ValueError('fix_cycles_greedy: edge marked for removal not found in edge set')
        
        # update the list of cycles and edges involved
        cycles_in_Gcycles = nx.cycle_basis(Gcycles)
        if len(cycles_in_Gcycles) == 0:
            all_cycles_fixed = True
        else:
            current_cycle_set = cycles_in_Gcycles
            current_cycle_set = set(tuple(i) for i in current_cycle_set)
            edges_involved_in_cycles = set()
            for cycle in current_cycle_set:
                assert type(cycle) is tuple, 'fix_cycles_greedy: cycle_set should be a set of tuples'
                temp_cycle1 = copy.deepcopy(cycle)
                temp_cycle1 = temp_cycle1 + (cycle[0],)
                possible_cycle_edges1 = [(temp_cycle1[i], temp_cycle1[i+1]) for i in range(len(temp_cycle1)-1)]
                for possible_edge in possible_cycle_edges1:
                    if (possible_edge[0], possible_edge[1]) in edge_set:
                        edges_involved_in_cycles.add((possible_edge[0], possible_edge[1]))
                    elif (possible_edge[1], possible_edge[0]) in edge_set:
                        edges_involved_in_cycles.add((possible_edge[1], possible_edge[0]))
                    else:
                        raise ValueError('fix_cycles_greedy: cycle edge (%d,%d) not found in edge_set' % (possible_edge[0], possible_edge[1]))
    
    budget_recovered = sum(edge_costs[e] for e in edges_to_remove)
    objective_loss = sum(edge_values[e] for e in edges_to_remove)
    if verbose:
        print('fix_cycles_greedy: took %0.2f seconds'%(time.time()-fc_timer))
        print('fix_cycles_greedy: recovered %0.6f units of budget'%budget_recovered)
        print('fix_cycles_greedy: incurred loss of %0.2f units to objective'%objective_loss)

    return edges_to_remove



def budget_pcsf_mlb_knapsack_repair(items, values, weights, capacity, vertices,                     demand, knapsack_avoid_cycles = True, 
                    fix_cycles_method = 'mip', fix_cycles_obj = 'mlbobj', verbose = True, instancestring = '', logfiledir = '', appendlog = '', resultfiledir = ''):
    '''Maximizes a modular lower bound (MLB) for a supermodular function subject to a combination of a knapsack and a graph matroid constraint. Maximization proceeds in alternating phases of ILP-based maximization wrt the budget constraint, followed by cycle fixing and budget recovery.
    
    Inputs:
    items             - list of edges that are candidates for selection
    values            - dict containing the modular value of each edge
    weights           - dict containing the cost of each edge
    capacity          - budget for selecting edges
    vertices          - list of vertices that make up the graph
    knapsack_avoid_cycles - whether to use modified knapsack mip reducing cycles
    fix_cycles_method - 'mip' or 'greedy' heuristic
    fix_cycles_obj    - 'budget' or 'mlbobj'
    
    Returns:
    selected_items            - set of selected edges
    selected_items_weight     - total cost of selected edges
    selected_items_mlb_value  - total modular value of selected edges
    '''
    
    if appendlog == '': # standalone run
        assert instancestring is not '', 'budget_pcsf_mlb_knapsack_repair: need to provide an instance string for standalone run'
        assert logfiledir is not '', 'budget_pcsf_mlb_knapsack_repair: need to specify a log file directory for standalone run'
        assert resultfiledir is not '', 'budget_pcsf_mlb_knapsack_repair: need to specify a result file directory for standalone run'
        solverunid = np.random.randint(10000)
        logfilename = logfiledir+instancestring+'_'+str(solverunid)+'.log'
        sys.stdout = open(logfilename, 'w')
    else: # subroutine run
        logfilename = appendlog
        sys.stdout = open(logfilename, 'a')
    

    selected_items = set()
    available_budget = capacity
    E_candidates = copy.deepcopy(items)

    min_item_cost = min([weights[e] for e in E_candidates])
    if min_item_cost <= capacity + 0.0000000001:
        can_add_more_edges = True
    
    npasses = 0
    timer = time.time()
    while can_add_more_edges:
        npasses += 1
        if verbose:
            print('----------Knapsack-Repair Pass #%d----------'%npasses)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # KNAPSACK MIP PHASE
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # select subset of edges to add to knapsack
        if verbose:
            print 'KNAPSACK PHASE\n>>Finding edges to add with knapsack MIP from %d candidates with budget %0.6f'%(len(E_candidates), available_budget)

        if knapsack_avoid_cycles:
            kmip_elem, kmip_weight, kmip_val = knapsack_avoid_cycles_mip(E_candidates, values, weights, available_budget, verbose=verbose)
        else:
            kmip_elem, kmip_weight, kmip_val = knapsack_mip(E_candidates, values, weights, available_budget, verbose=verbose)
        if verbose:
            print '>>Proposed to add %d edges from these candidates'%len(kmip_elem)
            print('>>Current MLB objective value: %d'%(sum([values[k] for k in selected_items])+sum([values[k] for k in kmip_elem])))
            print('>>Current cost: %0.6f out of %0.6f capacity'%(sum([weights[k] for k in selected_items])+sum([weights[k] for k in kmip_elem]), capacity))
        if len(kmip_elem) == 0:
            break
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # REPAIR PHASE
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # find cycles in current subset of edges
        if verbose:
            print 'REPAIR PHASE\n>>Finding cycles in solution'
        G = nx.Graph(directed=False)
        for (i,j) in selected_items.union(kmip_elem):
            G.add_edge(i,j)
        cycles = nx.cycle_basis(G) # list of cycles
        cycles = set(tuple(i) for i in cycles) # list of lists to set of tuples
        cycles_copy = copy.deepcopy(cycles) # store cycles found before fixing
                
        # fix cycles in current subset of edges
        if len(cycles_copy)>0:
            if verbose:
                print '>>%d cycles found!'%len(cycles)
                print '>>Finding edges to remove to break cycles in solution'
            if fix_cycles_method == 'mip':
                edges_to_remove = fix_cycles_mip_heuristic(selected_items.union(kmip_elem), cycles, weights, values, obj=fix_cycles_obj)
            elif fix_cycles_method == 'greedy':
                edges_to_remove = fix_cycles_greedy(selected_items.union(kmip_elem), cycles, weights, values, obj=fix_cycles_obj, verbose=verbose)
            selected_items = selected_items.union(kmip_elem)
            for a in range(len(edges_to_remove)):
                selected_items.remove(edges_to_remove[a])
            if verbose:
                print('>>%d edges removed to break cycles'%len(edges_to_remove))
                print('>>Decrease in current MLB objective value: %d'%(sum([values[k] for k in edges_to_remove])))
        else:
            if verbose:
                print '>>No cycles found!'
            selected_items = selected_items.union(kmip_elem)
        
        # end pass and assess whether another pass is possible        
        if verbose:
            print('------End of Knapsack-Repair Pass #%d-------'%npasses)
            print('>>Number of edges added so far: %d'%len(selected_items))

        # compute remaining budget
        available_budget = capacity - sum([weights[e] for e in selected_items])
        if verbose:
            print('>>Budget remaining: %0.6f'%available_budget)
            print('>>Current MLB objective value: %d'%(sum([values[k] for k in selected_items])))
        
        # find remaining edge candidates
        if len(cycles_copy)>0:
            E_candidates = set(E_candidates).difference(selected_items).difference(set(edges_to_remove))
            E_candidates = list(E_candidates)
        elif len(cycles_copy)==0:
            E_candidates = set(E_candidates).difference(selected_items)
            E_candidates = list(E_candidates)
        if verbose:
            print('>>Candidates remaining: %d'%len(E_candidates))

        if (len(E_candidates)==0):
            can_add_more_edges = False
            if verbose:
                print('>>All candidate edges eliminated')
        else:
            min_item_cost = min([weights[e] for e in E_candidates])
            if (available_budget < min_item_cost):
                if verbose:
                    print('>>Not enough budget left to purchase cheapest remaining candidate edge')
                    print('Final remaining budget: %0.4f'%available_budget)
                    print('Final min cost candidate: %0.4f'%min_item_cost)
                can_add_more_edges = False
    if (appendlog == '') or verbose:
        print('Knapsack-repair took %0.2f seconds'%(time.time()-timer))
    selected_items_weight = sum(weights[i] for i in selected_items)
    selected_items_mlb_value = sum(values[i] for i in selected_items)
    
    G = nx.Graph(directed=False)
    G.add_nodes_from(vertices)
    for (i,j) in selected_items:
        G.add_edge(i,j)
    cycles = nx.cycle_basis(G) # list of cycles
    
    assert selected_items_weight <= capacity + 0.0000000001, 'Selected items exceed weight capacity'
    assert len(nx.cycle_basis(G)) == 0, 'Cycles present in selected items solution'

    if (appendlog == '') or verbose:
        print('Final solution cost: %0.4f'%(selected_items_weight))
        print('Final solution MLB value: %0.4f'%(selected_items_mlb_value))

        f_selected_items = evaluate_solution(selected_items, vertices, demand)[0]
        print('Final solution true value: %0.4f'%(f_selected_items))

    # Save results if this is a standalone run
    if appendlog == '':
        resultsfilename = resultfiledir+instancestring+'_'+str(solverunid)+'.csv'
        with open(resultsfilename, 'wb') as results:
            writer = csv.writer(results, delimiter=' ')
            writer.writerow(['e, x_e'])
            for (i,j) in items:
                if (i,j) in selected_items:
                    writer.writerow([(i,j), np.abs(1)])
                else:
                    writer.writerow([(i,j), np.abs(0)])
    
    return selected_items, selected_items_weight, selected_items_mlb_value



def budget_pcsf_mlb_greedy(items, values, weights, capacity, vertices, demand, ratio=True, instancestring = '', logfiledir = '', appendlog = '', resultfiledir = '', verbose = True):
    '''Greedily maximizes a modular lower bound (MLB) for a supermodular function subject to a combination of a knapsack and a graph matroid constraint.
    
    Inputs:
    items    - list of edges that are candidates for selection
    values   - dict containing the modular value of each edge
    weights  - dict containing the cost of each edge
    capacity - budget for selecting edges
    vertices - list of vertices that make up the graph
    ratio    - whether to add edges by value/cost ratio or by value
    
    Returns:
    selected_items            - set of selected edges
    selected_items_weight     - total cost of selected edges
    selected_items_mlb_value  - total modular value of selected edges
    '''

    if appendlog == '': # standalone run
        assert instancestring is not '', 'budget_pcsf_mlb_knapsack_repair: need to provide an instance string for standalone run'
        assert logfiledir is not '', 'budget_pcsf_mlb_knapsack_repair: need to specify a log file directory for standalone run'
        assert resultfiledir is not '', 'budget_pcsf_mlb_knapsack_repair: need to specify a result file directory for standalone run'
        solverunid = np.random.randint(10000)
        logfilename = logfiledir+instancestring+'_'+str(solverunid)+'.log'
        sys.stdout = open(logfilename, 'w')
    else: # subroutine run
        logfilename = appendlog
        sys.stdout = open(logfilename, 'a')

    selected_items = set()
    available_budget = capacity
    G = nx.Graph(directed=False)
    G.add_nodes_from(vertices)
    
    timer = time.time()
    # when greedily maximizing the MLB, 
    # inspect edges in descending order of value/cost or value
    if ratio:
        value_cost_ratio = dict.fromkeys(values.keys())
        for i in values.keys():
            value_cost_ratio[i] = float(values[i])/float(weights[i])
        sorted_items = sorted(value_cost_ratio.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_items = sorted(values.items(), key=operator.itemgetter(1), reverse=True)

    for i in sorted_items:
        can_afford = weights[i[0]] <= available_budget
        G.add_edge(i[0][0], i[0][1])
        cycles = nx.cycle_basis(G) # list of cycles
        does_not_create_cycle = len(cycles)==0
        if can_afford and does_not_create_cycle:
            selected_items.add(i[0])
            available_budget -= weights[i[0]]
        else:
            G.remove_edge(i[0][0], i[0][1])
    
    if (appendlog == '') or verbose:
        print('Greedy MLB maximization took %0.2f seconds'%(time.time()-timer))
    selected_items_weight = sum(weights[i] for i in selected_items)
    selected_items_mlb_value = sum(values[i] for i in selected_items)
        
    assert selected_items_weight <= capacity + 0.0000000001, 'Selected items exceed weight capacity'
    assert len(nx.cycle_basis(G)) == 0, 'Cycles present in selected items solution'
    
    if (appendlog == '') or verbose:
        print('Final solution cost: %0.4f'%(selected_items_weight))
        print('Final solution MLB value: %0.4f'%(selected_items_mlb_value))

        f_selected_items = evaluate_solution(selected_items, vertices, demand)[0]
        print('Final solution true value: %0.4f'%(f_selected_items))

    # Save results if this is a standalone run
    if appendlog == '':
        resultsfilename = resultfiledir+instancestring+'_'+str(solverunid)+'.csv'
        with open(resultsfilename, 'wb') as results:
            writer = csv.writer(results, delimiter=' ')
            writer.writerow(['e, x_e'])
            for (i,j) in items:
                if (i,j) in selected_items:
                    writer.writerow([(i,j), np.abs(1)])
                else:
                    writer.writerow([(i,j), np.abs(0)])

    return selected_items, selected_items_weight, selected_items_mlb_value



def parallel_eval(e, G, V, D):
    '''
    Evaluates the effect of removing edge from a spanning tree.
    Inputs:
    e    - edge to be removed
    G    - spanning tree graph (nx)
    V    - vertices in G
    D    - demand between pair of nodes in G

    Returns:
    f_G_minus_e - demand satisfied when e is removed from G
    '''
    G_temp = copy.deepcopy(G)
    G_temp.remove_edge(e[0], e[1])
    f_G_minus_e = evaluate_solution_from_graph(G_temp, V, D)
    return f_G_minus_e


def evaluate_removing_edges_from_spanning_tree(G_tree, vertices, edges, demand, max_values_current):
#     assert nx.is_connected(G_tree), 'The graph is not connected'
#     assert nx.is_tree(G_tree), 'The graph is not a tree'
#     assert len(G_tree.nodes) == len(vertices)  
    timer = time.time()
    f_G_tree = evaluate_solution_from_graph(G_tree, vertices, demand)
    pool = multiprocessing.Pool(10)
    parallel_eval_e = partial(parallel_eval, G=G_tree, V=vertices, D=demand) # parallel_eval has only one argument x (y is fixed to 10)
    result_list = pool.map(parallel_eval_e, G_tree.edges)
    G_tree_edges_list = list(G_tree.edges)
    for ei in range(len(G_tree_edges_list)):
        if f_G_tree - result_list[ei] > max_values_current[G_tree_edges_list[ei]]:
            max_values_current[G_tree_edges_list[ei]] = f_G_tree - result_list[ei]
    print('Parallelly re-evaluating f for each edge in the tree and comparing to current max values took %0.2f seconds'%(time.time() - timer))
    pool.close()
    return max_values_current



def find_max_values(edges, vertices, demand, nruns = 20, method = 'connected'):
    '''Estimates the maximum possible value each edge adds to a graph with pairwise node profits 
    (restricted supermodular on edges), when the edge is added last. Generates random spanning 
    trees on the graph, and finds the loss in objective caused by removing e.
    
    Inputs:
    edges    - list of all edges in the graph
    vertices - list of all vertices in the graph
    demand   - pairwise node profits
    nruns    - number of times to generate random spanning tree
    method   - 'cycles' if checking cycle_basis, else 'connected' if checking is_connected
    
    Returns:
    e_last_value - the value of e when it is added last to a spanning tree (forest) on the graph
    '''
    max_values = dict.fromkeys(edges)
    for e in edges:
        max_values[e] = 0
        
    master_timer = time.time()
    
    if method == 'connected':
        
        # Construct the full graph
        G_full_orig = nx.Graph(directed=False)
        G_full_orig.add_nodes_from(vertices)
        for (i,j) in edges:
            G_full_orig.add_edge(i,j)
            
        print('graph is connected initially: ', nx.is_connected(G_full_orig))

        # Get edges participating in cycles
        edges_participating_in_cycles = set()
        G_full_orig_cycles = nx.cycle_basis(G_full_orig)
        for c in G_full_orig_cycles:
            cp = tuple(c) + (c[0],)
            c_edges = set([(cp[k], cp[k+1]) if (cp[k], cp[k+1]) in edges else (cp[k+1], cp[k]) for k in range(len(cp)-1)])
            edges_participating_in_cycles = edges_participating_in_cycles.union(c_edges)

        edges_participating_in_cycles = list(edges_participating_in_cycles)

        for run in range(nruns):
            print(run)
            # Copy the full graph
            G_full = copy.deepcopy(G_full_orig)
            G_full_is_tree = nx.is_tree(G_full)

            # Randomly order edges participating in cycles
            timer = time.time()
            edge_order = np.random.permutation(len(edges_participating_in_cycles))
            ordered_edges = [edges_participating_in_cycles[edge_order[j]] for j in range(len(edge_order))]
            # print('order of edges participating in cycles to remove:')
            # print(ordered_edges)
            oe = 0
            while not G_full_is_tree:
                # Try to remove the next ordered edge
                G_full.remove_edge(ordered_edges[oe][0], ordered_edges[oe][1])

                # Check if graph is still connected
                connected = nx.is_connected(G_full)

                if connected:
                    print('removed')
                    G_full_is_tree = nx.is_tree(G_full)
                    oe += 1
                else:
                    print('disconnected')
                    # Put the edge back
                    G_full.add_edge(ordered_edges[oe][0], ordered_edges[oe][1])
                    oe += 1

            # For any edge still present in G_full, evaluate G_full with and without e
            max_values = evaluate_removing_edges_from_spanning_tree(G_full, vertices, edges, demand, max_values)
    
    elif method == 'cycles':

        # Construct the full graph
        G_full_orig = nx.Graph(directed=False)
        G_full_orig.add_nodes_from(vertices)
        for (i,j) in edges:
            G_full_orig.add_edge(i,j)
        
        for run in range(nruns):
            # print(run)            
            # Copy the full graph
            G_full = copy.deepcopy(G_full_orig)
            G_full_cycles = nx.cycle_basis(G_full)
            
            # Randomly remove edges from cycles until no more cycles remain
            while len(G_full_cycles)>0:
                next_cycle = G_full_cycles[0]
                complete_next_cycle = tuple(next_cycle) + (next_cycle[0],)
                complete_next_cycle_edges = []
                for i in range(len(complete_next_cycle)-1):
                    possible_e = (complete_next_cycle[i], complete_next_cycle[i+1])
                    if possible_e in edges:
                        complete_next_cycle_edges.append(possible_e)
                    elif (possible_e[1], possible_e[0]) in edges:
                        complete_next_cycle_edges.append((possible_e[1], possible_e[0]))
                    else:
                        raise ValueError('find_max_value: edge found in cycle that is not found in edge list')
                edge_order = np.random.permutation(len(complete_next_cycle_edges))
                remove_e_candidates = [complete_next_cycle_edges[i] for i in edge_order]
                removed = False
                i = 0
                while not removed and i < len(remove_e_candidates):
                    remove_e_candidate = remove_e_candidates[i]
                    if remove_e_candidate in G_full.edges:
                        G_full.remove_edge(remove_e_candidate[0], remove_e_candidate[1])
                        removed = True
                    else:
                        i += 1
                G_full_cycles = nx.cycle_basis(G_full)

            assert len(G_full_cycles) == 0, 'There are still cycles in the graph'
            
            # For any edge that is present in G_full, evaluate G_full with and without e
            max_values = evaluate_removing_edges_from_spanning_tree(G_full, vertices, edges, demand, max_values)

    assert sum([max_values[k] >= 0 for k in edges]) == len(edges), 'find_max_values: some edges were not evaluated in the sampled random spanning trees'
    print('Finding max values with %d sampled spanning trees took %0.2f seconds'%(nruns, time.time() - master_timer))
    return max_values



def budget_pcsf_semigrad_ascent(G, values, max_values, weights,
                                capacity, demand,
                                init_method='empty', mlb_selection='alternating', mlb_max_method='kr-avoid-cycles', fix_cycles_method = 'greedy', fix_cycles_obj = 'budget',
                                instancestring='', logfiledir='', resultfiledir='', verbose=True):
    '''Semigradient-based supermodular function maximization subject to a combination of a knapsack and a matroid constraint.
    
    Inputs:
    G                 - list of edges that are candidates for selection
    values            - dict containing the modular value of each edge
    weights           - dict containing the cost of each edge
    capacity          - budget for selecting edges
    demand            - demand between pairs of vertices
    init_method       - whether to start with 'empty', 'full', or 'greedy' set
    mlb_max_method    - 'knapsackmip_repair' or 'greedy'
    
    Returns:
    selected_items            - set of selected edges
    selected_items_weight     - total cost of selected edges
    selected_items_mlb_value  - total modular value of selected edges
    selected_items_true_value - true value of selected edges
    '''
    
    solverunid = np.random.randint(10000)
    logfilename = logfiledir+instancestring+'_'+str(solverunid)+'.log'
    sys.stdout = open(logfilename, 'w')
    
    V = [v.index for v in G.vs()]
    # num_vertices = len(V)
    E = G.get_edgelist()
    
    E_candidates = copy.deepcopy(E)
    timer = time.time()
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # INITIALIZE SOLUTION
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if verbose:
        print('Initializing with %s set.'%init_method)
    if init_method is 'empty':
        selected_items_current = set()
    elif init_method is 'full':
        selected_items_current = set(E_candidates)
    elif init_method is 'random':
        edge_order = np.random.permutation(len(E_candidates))
        random_edges = []
        k = 0
        while (sum([weights[i] for i in random_edges])) < capacity:
            random_edges.append(E_candidates[edge_order[k]])
            k+= 1
        selected_items_current = set(random_edges)
    elif init_method is 'greedy':
	# # if greedy solution was not already computed
        # selected_items_current = budget_pcsf_greedy(G, weights, demand, capacity, by_ratio=True, instancestring='', logfiledir='', resultfiledir='', verbose=False)[0]
        # selected_items_current = set(selected_items_current)
	# if greedy solution has already been computed, load from result
	selected_items_current = set()
	greedyrdir = '../notebooks/synthetic_graphs/greedy_results_2/'
	#greedyrfile = ['_'.join((f.split('.csv')[0]).split('_')[:-1]) for f in os.listdir(greedyrdir) if os.path.isfile(os.path.join(greedyrdir,f)) and f[0]=='G']
	greedyrfile = [f for f in os.listdir(greedyrdir) if os.path.isfile(os.path.join(greedyrdir,f)) and f[0]=='G' and ('_'.join((f.split('.csv')[0]).split('_')[:-1]) == instancestring)][0]
        with open(greedyrdir+greedyrfile,'r') as gr:
		for line in gr:
			if 'x_e' in line:
				pass
			else:
				edge_ij = line.strip()[:-1]
				edge_x = line.strip()[-1]
				if edge_x == 1:
					selected_items_current.union(eval(edge_ij))
    f_selected_items_current = evaluate_solution(selected_items_current, V, demand)[0]
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SEMIGRADIENT MLB MAXIMIZATION
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    converged = False
    iteration = 0
    if mlb_selection == 'alternating':
        mlb1_converged = False
        mlb2_converged = False
        current_mlb = 'mlb1' # start with MLB1
        while not converged:
            if verbose:
                print('--------Semigradient Ascent Iteration %d --------'%iteration)
            # pick a subdifferential-based semigradient
            if current_mlb == 'mlb1': # use MLB1
                # compute MLB1 coefficients based on semigradient at current solution
                mlb_coeff = dict.fromkeys(values.keys())
                selected_items_count = 0
                non_selected_items_count = 0
                for e in E_candidates:
                    if e not in selected_items_current:
                        non_selected_items_count += 1
                        selected_items_temp_copy = copy.deepcopy(selected_items_current)
                        G_add_e = evaluate_solution(selected_items_temp_copy.union([e]), V, demand)[1]
                        G_add_e_cycles = nx.cycle_basis(G_add_e)
                        # if adding the edge would create a cycle, set its mlb coefficient to 0
                        if len(G_add_e_cycles)>0:
                            mlb_coeff[e] = 0
                        else:
                            mlb_coeff[e] = values[e]
                        # just set mlb coeff straight to f(j|empty)
                        #mlb_coeff[e] = values[e]
                    else:
                        selected_items_count += 1
                        selected_items_temp_copy = copy.deepcopy(selected_items_current)
                        f_adde_selected_items_minuse = f_selected_items_current - evaluate_solution(selected_items_temp_copy.difference([e]), V, demand)[0]
                        mlb_coeff[e] = f_adde_selected_items_minuse
                assert selected_items_count+non_selected_items_count == len(E_candidates), 'budget_pcsf_semigrad_ascent: during mlb_coeff computation, edges in solution and not in solution do not add up to |E|'

            elif current_mlb == 'mlb2': # use MLB2
                # compute MLB1 coefficients based on semigradient at current solution
                mlb_coeff = dict.fromkeys(values.keys())
                selected_items_count = 0
                non_selected_items_count = 0
                for e in E_candidates:
                    if e not in selected_items_current:
                        non_selected_items_count += 1
                        selected_items_temp_copy = copy.deepcopy(selected_items_current)
                        f_selected_items_adde, G_add_e = evaluate_solution(selected_items_temp_copy.union([e]), V, demand)
                        G_add_e_cycles = nx.cycle_basis(G_add_e)
                        # if adding the edge would create a cycle, set its mlb coefficient to 0
                        if len(G_add_e_cycles)>0:
                            mlb_coeff[e] = 0
                        else:
                            mlb_coeff[e] = f_selected_items_adde - f_selected_items_current
                        # just set mlb coeff straight to f(j|S)
                        #mlb_coeff[e] = f_selected_items_adde - f_selected_items_current
                        # j|S seems problematic in terms of MLB, try j|empty
                        #mlb_coeff[e] = values[e]
                    else:
                        selected_items_count += 1
                        mlb_coeff[e] = max_values[e]
                assert selected_items_count+non_selected_items_count == len(E_candidates), 'budget_pcsf_semigrad_ascent: during mlb_coeff computation, edges in solution and not in solution do not add up to |E|'
                ## DEBUG
                # print('current MLB coeffs:')
                # print(mlb_coeff)

            # maximize the MLB based on the selected semigradient
            if mlb_max_method == 'greedy':
                selected_items_new, selected_items_new_weight, selected_items_new_mlb_value = budget_pcsf_mlb_greedy(E_candidates, mlb_coeff, weights, capacity, V, demand, ratio=True, instancestring = '', logfiledir = '', resultfiledir = '', appendlog=logfilename, verbose = verbose) ### add demand arg
            elif mlb_max_method == 'kr':
                selected_items_new, selected_items_new_weight, selected_items_new_mlb_value = budget_pcsf_mlb_knapsack_repair(E_candidates, mlb_coeff, weights, capacity, V, demand, knapsack_avoid_cycles = False, fix_cycles_method = fix_cycles_method, fix_cycles_obj = fix_cycles_obj, instancestring = '', logfiledir = '', resultfiledir = '', appendlog=logfilename, verbose = verbose)
            elif mlb_max_method == 'kr-avoid-cycles':
                selected_items_new, selected_items_new_weight, selected_items_new_mlb_value = budget_pcsf_mlb_knapsack_repair(E_candidates, mlb_coeff, weights, capacity, V, demand, knapsack_avoid_cycles = True, fix_cycles_method = fix_cycles_method, fix_cycles_obj = fix_cycles_obj, instancestring = '', logfiledir = '', resultfiledir = '', appendlog=logfilename, verbose = verbose)
            else:
                raise ValueError('budget_pcsf_semigrad_ascent: invalid mlb_max_method provided')
            
            ## DEBUG
            # check the bound inequality holds
            # print('current selected items:')
            # print(selected_items_current)
            # print('new selected items:')
            # print(selected_items_new)
            #print('current selected items true value:')
            #print(f_selected_items_current)
            #print('current selected items MLB value:')
            #print(sum([mlb_coeff[i] for i in selected_items_current]))
            #print('new selected items MLB value:')
            #print(selected_items_new_mlb_value)
            # print('new selected items MLB partial value:')
            LHS = f_selected_items_current + selected_items_new_mlb_value - sum([mlb_coeff[i] for i in selected_items_current])
            f_selected_items_new = evaluate_solution(selected_items_new, V, demand)[0]
            if verbose:
                print(f_selected_items_current, LHS, f_selected_items_new)
                if LHS < 0:
                    print('MLB IS NEGATIVE')
                else:
                    print('MLB IS NON-NEGATIVE!!!!!!!!!!!')
            #assert LHS <= f_selected_items_new, 'budget_pcsf_semigrad_ascent: after MLB maximization, MLB inequality violated--check bound computation'
            # check the results are within budget
            assert selected_items_new_weight <= capacity + 0.0000000001, 'budget_pcsf_semigrad_ascent: after MLB maximization, selected items exceed weight capacity'

            # check for convergence                
            if selected_items_new == selected_items_current:
                if current_mlb == 'mlb1':
                    mlb1_converged = True
                    if verbose:
                        print('MLB1 converged')
                    current_mlb = 'mlb2' # switch to mlb2
                    selected_items_current = selected_items_new
                    f_selected_items_current = f_selected_items_new
                    iteration += 1
                elif current_mlb == 'mlb2':
                    mlb2_converged = True
                    if verbose:
                        print('MLB2 converged')
                    current_mlb = 'mlb1' # switch to mlb1
                    selected_items_current = selected_items_new
                    f_selected_items_current = f_selected_items_new
                    iteration += 1
                if mlb1_converged and mlb2_converged:
                    converged = True
                    
            else:
                if verbose:
                    print('Convergence broken')
                selected_items_current = selected_items_new
                f_selected_items_current = f_selected_items_new
                iteration += 1
                mlb1_converged = False
                mlb2_converged = False

                if iteration > 100:
                    break
    
    print('Semigradient maximization took %0.2f seconds'%(time.time()-timer))
    # selected_items_mlb_value = sum(values[i] for i in selected_items)
    
    selected_items_current_weight = sum([weights[i] for i in selected_items_current])
    assert selected_items_current_weight <= capacity + 0.0000000001, 'budget_pcsf_semigrad_ascent: final selected items exceed weight capacity'
    
    print('Final solution cost: %0.4f'%(selected_items_current_weight))
    print('Final solution MLB value: %0.4f'%(LHS))

    f_selected_items = evaluate_solution(selected_items_current, V, demand)[0]
    print('Final solution true value: %0.4f'%(f_selected_items))
    # save the results
    resultsfilename = resultfiledir+instancestring+'_'+str(solverunid)+'.csv'
    with open(resultsfilename, 'wb') as results:
        writer = csv.writer(results, delimiter=' ')
        writer.writerow(['e, x_e'])
        for (i,j) in E:
            if (i,j) in selected_items_current:
                writer.writerow([(i,j), np.abs(1)])
            else:
                writer.writerow([(i,j), np.abs(0)])
    
    return selected_items_current, selected_items_current_weight, LHS, f_selected_items_current
