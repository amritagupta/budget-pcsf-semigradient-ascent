from budgetpcsfcopy import *

for num_edges in [500, 1000, 1500]:
	for instance in range(20):
		for budget_frac in [0.1]:#placeholder
			# Load problem data
			f_G = "../data/synthetic_graphs/planar/graphs/G_n_edges_%d_instance_%d.graphml"%(num_edges, instance)
			G = igraph.Graph.Read_GraphML(f_G)
			f_c = "../data/synthetic_graphs/planar/edge_costs/G_n_edges_%d_instance_%d_cost_data.p"%(num_edges, instance)
			(cost_E, total_cost, total_mst_cost) = pickle.load(open(f_c, "rb"))
			f_p = "../data/synthetic_graphs/planar/demands/G_n_edges_%d_instance_%d_demand_data.p"%(num_edges, instance)
			demand = pickle.load(open(f_p, "rb"))
			f_v = "../data/synthetic_graphs/planar/edge_values/G_n_edges_%d_instance_%d_edge_values_data.p"%(num_edges, instance)

			V = [v.index for v in G.vs()]
			E = G.get_edgelist()

			inststring = 'G_n_edges_%d_instance_%d'%(num_edges, instance)
			
			nsamples = 20
			maxvals_lfd = '../data/synthetic_graphs/planar/edge_max_values_%d_logs/'%nsamples
			logfilename = maxvals_lfd+inststring+'.log'
			sys.stdout = open(logfilename, 'w')
			mvtimer = time.time()
			max_values = find_max_values(E,V,demand,nruns=nsamples,method='connected')
			time_taken = time.time() - mvtimer
        		print('%d, %d, %0.4f'%(num_edges, instance, time_taken))
			pickle.dump(max_values, open("../data/synthetic_graphs/planar/edge_max_values_%d/G_n_edges_%d_instance_%d_edge_max_values_data.p"%(nsamples, len(E), instance), "wb" ))
			


