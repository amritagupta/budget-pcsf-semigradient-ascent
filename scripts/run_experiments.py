from budgetpcsf import *

budgetfraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
for num_edges in [500, 1000, 1500]:
	for instance in range(20):
		for budget_frac in budgetfraction:
			# Load problem data
			f_G = "../data/synthetic_graphs/er/graphs/G_n_edges_%d_instance_%d.graphml"%(num_edges, instance)
			G = igraph.Graph.Read_GraphML(f_G)
			f_c = "../data/synthetic_graphs/er/edge_costs/G_n_edges_%d_instance_%d_cost_data.p"%(num_edges, instance)
			(cost_E, total_cost, total_mst_cost) = pickle.load(open(f_c, "rb"))
			f_p = "../data/synthetic_graphs/er/demands/G_n_edges_%d_instance_%d_demand_data.p"%(num_edges, instance)
			demand = pickle.load(open(f_p, "rb"))
			f_v = "../data/synthetic_graphs/er/edge_values/G_n_edges_%d_instance_%d_edge_values_data.p"%(num_edges, instance)
			values = pickle.load(open(f_v, "rb"))
			f_mv = "../data/synthetic_graphs/er/edge_max_values_50/G_n_edges_%d_instance_%d_edge_max_values_data.p"%(num_edges, instance)
			max_values = pickle.load(open(f_mv, "rb"))

			V = [v.index for v in G.vs()]
			num_vertices = len(V)
			E = G.get_edgelist()
			B = float(budget_frac*total_mst_cost)

			inststring = 'G_n_edges_%d_instance_%d_bfrac_%0.1f'%(num_edges, instance, budget_frac)

			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			# GREEDY
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #	
			# # greedy (by_ratio: true)
			greedy_lfd = '../output/synthetic_graphs/er/greedy_logs_minimal/'
			greedy_rfd = '../output/synthetic_graphs/er/greedy_results_minimal/'
			sol, sol_wt, sol_mlbval = budget_pcsf_greedy(G, cost_E, demand, B, by_ratio=True, instancestring=inststring, logfiledir=greedy_lfd, resultfiledir=greedy_rfd, verbose=False)

			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			# KNAPSACK-REPAIR EXPERIMENTS
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			# # knapsack-repair (avoid_cycles: False, fix_cycles_method: 'greedy', fix_cycles_obj: 'budget')
			kr_fc_greedy_fco_budget_lfd = '../output/synthetic_graphs/er/kr_logs_minimal/'
			kr_fc_greedy_fco_budget_rfd = '../output/synthetic_graphs/er/kr_results_minimal/'
			sol, sol_wt, sol_mlbval = budget_pcsf_mlb_knapsack_repair(E, values, cost_E, B, V, demand, knapsack_avoid_cycles=False, fix_cycles_method='greedy', fix_cycles_obj='budget', verbose=False, instancestring=inststring, logfiledir=kr_fc_greedy_fco_budget_lfd, resultfiledir=kr_fc_greedy_fco_budget_rfd)

			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			# SEMIGRAD ASCENT WITH GREEDY
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			sg_greedy_init_empty_lfd = '../output/synthetic_graphs/er/semigrad_greedy_init_empty_logs_minimal/'
			sg_greedy_init_empty_rfd = '../output/synthetic_graphs/er/semigrad_greedy_init_empty_results_minimal/'
			sol, sol_wt, sol_mlb, sol_true = budget_pcsf_semigrad_ascent(G, values, max_values, cost_E, B, demand, init_method='empty', mlb_selection='alternating',
									mlb_max_method='greedy', fix_cycles_method = 'greedy', fix_cycles_obj = 'budget',
									instancestring=inststring, logfiledir=sg_greedy_init_empty_lfd,
									resultfiledir=sg_greedy_init_empty_rfd, verbose=False)
			
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			# SEMIGRAD ASCENT WITH KNAPSACK-REPAIR
			# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			sg_kr_init_empty_lfd = '../output/synthetic_graphs/er/semigrad_kr_init_empty_logs_minimal/'
			sg_kr_init_empty_rfd = '../output/synthetic_graphs/er/semigrad_kr_init_empty_results_minimal/'
			sol, sol_wt, sol_mlb, sol_true = budget_pcsf_semigrad_ascent(G, values, max_values, cost_E, B, demand,
					init_method='empty', mlb_selection='alternating', mlb_max_method='kr',
					fix_cycles_method = 'greedy', fix_cycles_obj = 'budget', instancestring=inststring,
					logfiledir=sg_kr_init_empty_lfd, resultfiledir=sg_kr_init_empty_rfd, verbose=False)