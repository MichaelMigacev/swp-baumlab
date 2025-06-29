import os
import pdb
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from specify import Specify
from mpl_toolkits.axes_grid1 import make_axes_locatable

class NetAnalyse():
    def __init__(self):
        pass

    def statistic_net(self):
        # READ INFORMATION FROM FILE
        gene_bind_weight_edge_df = pd.read_csv('../analysis_nci/gene_bind_weight_edge.csv')
        gene_bind_degree_df = pd.read_csv('../analysis_nci/gene_weight_bind_degree.csv')
        # BINPLOT OF WEIGHT DISTRIBUTION
        bind_weight_list = []
        for bind_row in gene_bind_weight_edge_df.itertuples():
            bind_weight_list.append(bind_row[3])
        bins = np.arange(-0.5, 3, 0.2) # FIXED BIN SIZE
        plt.xlim([min(bind_weight_list) - 0.2, max(bind_weight_list) + 0.2])
        plt.hist(bind_weight_list, bins = bins)
        plt.title('Gene-Gene Bindflow Weight Distribution')
        plt.ylabel('count')
        plt.show()
        # BINPLOT OF DEGREE DISTRIBUTION
        bind_degree_list = []
        for bind_row in gene_bind_degree_df.itertuples():
            bind_degree_list.append(bind_row[3])
        bins = np.arange(0, 120, 20) # FIXED BIN SIZE
        plt.xlim([min(bind_degree_list) - 5, max(bind_degree_list) + 20])
        plt.hist(bind_degree_list, bins = bins)
        plt.title('Gene Bindflow Degree Distribution')
        plt.ylabel('count')
        plt.show()

    def plot_target_gene(self, node_threshold):
        # READ INFORMATION FROM FILE
        gene_bind_weight_edge_df = pd.read_csv('../analysis_nci/gene_bind_weight_edge.csv')
        gene_bind_degree_df = pd.read_csv('../analysis_nci/gene_weight_bind_degree.csv')
        # MAP GENES WITH INDEX NUM !!! START FROM 1
        kegg_gene_num_dict_df = pd.read_csv('../data-nci/filtered_data/kegg_gene_num_dict.csv')
        kegg_gene_num_dict = dict(zip(kegg_gene_num_dict_df.kegg_gene, kegg_gene_num_dict_df.gene_num))
        kegg_gene_name_dict = dict(zip(kegg_gene_num_dict_df.gene_num, kegg_gene_num_dict_df.kegg_gene))
        # GET GENES TARGETED BY DRUGS
        final_drugbank_df = pd.read_csv('../data-nci/filtered_data/final_drugbank.csv')
        final_drugbank_num_df = pd.read_csv('../data-nci/filtered_data/final_drugbank.csv')
        gene_targeted_name_list = list(set(final_drugbank_df['Target']))
        gene_targeted_num_list = list(set(final_drugbank_df['Target']))
        print('----- TARGETED GENES BY DRUGS -----')
        print(gene_targeted_name_list)

        # BUILD bind NODES DEGREE [DELETION LIST]
        filter_node_count = 0
        bind_node_deletion_index_list = []
        bind_node_index_list = []
        bind_node_name_list = []
        bind_node_degree_list = []
        for bind_row in gene_bind_degree_df.itertuples():
            if bind_row[3] <= node_threshold:
                bind_node_deletion_index_list.append(bind_row[1])
            else:
                filter_node_count += 1
                bind_node_name_list.append(bind_row[2])
                bind_node_index_list.append(bind_row[1])
                bind_node_degree_list.append(bind_row[3])
        # INTERSECT ON [gene_targeted_set](TARGETED NODES) [bind_node_name_set](FILTERED BY THRESHOLD NODES)
        gene_targeted_set = set(gene_targeted_name_list)
        bind_node_name_set = set(bind_node_name_list)
        set1 = gene_targeted_set.intersection(bind_node_name_set)
        intersection_name_list = list(set1)
        intersection_list = [kegg_gene_num_dict[item] for item in intersection_name_list]

        intersection_degree_list = []
        for gene in intersection_list:
            gene_degree = float(gene_bind_degree_df.loc[gene_bind_degree_df['gene_idx'] == gene]['degree'])
            intersection_degree_list.append(gene_degree)
        print('----- GENE INTERSECTED TARGETRED BY DRUG LIST: ' + str(len(intersection_name_list)) + ' -----')
        print(intersection_name_list)
        return intersection_name_list, intersection_list, intersection_degree_list


    def drug_target_interaction(self, cellline_name, topmin_loss, testloss_topminobj, testloss_bottomminobj):
        filtered_drug_target_df = pd.read_csv('../data-nci/filtered_data/final_drugbank.csv')
        print(filtered_drug_target_df)

        # FORM [dl_input_drug_dict] TO MAP GENES WITH INDEX START FROM 2017
        drug_map_df = pd.read_csv('../data-nci/filtered_data/drug_num_dict.csv')
        drug_map_num_list = list(drug_map_df['drug_num'])
        drugbank_druglist = list(drug_map_df['Drug'])
        drugbank_drug_dict = {drugbank_druglist[i - 1] : drug_map_num_list[i - 1] for i in range(1, len(drugbank_druglist)+1)}

        # import pdb; pdb.set_trace() 

        print('\n-------- TEST ' + cellline_name + ' --------\n')

        if topmin_loss == True:
            # These 2 drug names are from [dl_input]
            # But [filtered_drug_target_df]'s Drug Names == dl_Drug Names
            top_testloss_dl_drugA = testloss_topminobj['Drug A']
            top_testloss_dl_drugB = testloss_topminobj['Drug B']
            print(top_testloss_dl_drugA, top_testloss_dl_drugB)
            # Get Cell Line Specific Drug Target Links
            cellline_specific_drugtar_list = []
            for row in filtered_drug_target_df.itertuples():
                if row[1] == top_testloss_dl_drugA or row[1] == top_testloss_dl_drugB:
                    print(row)
                    cellline_specific_drugtar_list.append(row[0])
            drugbank_num_df = pd.read_csv('../data-nci/filtered_data/final_drugbank_num.csv')
            cellline_specific_drugbank_df = drugbank_num_df.loc[drugbank_num_df.index.isin(cellline_specific_drugtar_list)]
            print(cellline_specific_drugbank_df)
            return cellline_specific_drugbank_df

        else:
            # These 2 drug names are from [dl_input]
            # But [filtered_drug_target_df]'s Drug Names == dl_Drug Names
            bottom_testloss_dl_drugA = testloss_bottomminobj['Drug A']
            bottom_testloss_dl_drugB = testloss_bottomminobj['Drug B']
            print(bottom_testloss_dl_drugA, bottom_testloss_dl_drugB)
            # Get Cell Line Specific Drug Target Links
            cellline_specific_drugtar_list = []
            for row in filtered_drug_target_df.itertuples():
                if row[1] == bottom_testloss_dl_drugA or row[1] == bottom_testloss_dl_drugB:
                    print(row)
                    cellline_specific_drugtar_list.append(row[0])
            drugbank_num_df = pd.read_csv('../data-nci/filtered_data/final_drugbank_num.csv')
            cellline_specific_drugbank_df = drugbank_num_df.loc[drugbank_num_df.index.isin(cellline_specific_drugtar_list)]
            print(cellline_specific_drugbank_df)
            return cellline_specific_drugbank_df

    def plot_net2(self, node_threshold, edge_threshold, 
                intersection_list, intersection_degree_list, cellline_specific_drugbank_df,
                topmin_loss, seed, cellline_name, top_n, cutoff_range):
        # READ INFORMATION FROM FILE
        gene_bind_weight_edge_df = pd.read_csv('../analysis_nci/gene_bind_weight_edge.csv')
        gene_bind_degree_df = pd.read_csv('../analysis_nci/gene_weight_bind_degree.csv')
        # FORM THE MAP FOR GENES WITH INDEX NUM !!! START FROM 1
        kegg_gene_num_dict_df = pd.read_csv('../data-nci/filtered_data/kegg_gene_num_dict.csv')
        kegg_gene_num_dict = dict(zip(kegg_gene_num_dict_df.gene_num, kegg_gene_num_dict_df.kegg_gene))

        # BUILD bind NODES DEGREE [DELETION LIST]
        filter_node_count = 0
        bind_node_deletion_index_list = []
        bind_node_index_list = []
        bind_node_name_list = []
        bind_node_degree_dict = {}
        for bind_row in gene_bind_degree_df.itertuples():
            if bind_row[3] <= node_threshold:
                bind_node_deletion_index_list.append(bind_row[1])
            else:
                filter_node_count += 1
                bind_node_name_list.append(bind_row[2])
                bind_node_index_list.append(bind_row[1])
                bind_node_degree_dict[bind_row[1]] = bind_row[3]
        print('----- FILTERED GENES WITH HIGHER DEGREES: ' + str(filter_node_count) + ' -----')

        # REMOVE CERTAIN DELETED GENES IN [cellline_gene_num_dict]
        bind_filter_gene_dict = kegg_gene_num_dict.copy()
        [bind_filter_gene_dict.pop(key) for key in bind_node_deletion_index_list]
        # BUILD bind DIRECTED GRAPH
        bind_digraph = nx.Graph() 
        edge_remove = False # DO NOT REMOVE EDGES, ONLY REMOVE NODES
        bind_edge_deletion_list = []
        for bind_row in gene_bind_weight_edge_df.itertuples():
            if edge_remove == True:
                if bind_row[3] <= edge_threshold:
                    bind_edge_deletion_list.append((bind_row[1], bind_row[2]))
            bind_digraph.add_edge(bind_row[1], bind_row[2], weight = bind_row[3])
        
        # BUILD [drug target]
        node_degree_max = max(list(bind_node_degree_dict.values()))
        edge_weight_max = max(list(gene_bind_weight_edge_df['conv_bind_weight']))
        for row in cellline_specific_drugbank_df.itertuples():
            drug_idx = row[1]
            gene_idx = row[2]
            # degree
            bind_node_degree_dict[drug_idx] = 0.8 * node_degree_max
            # weight
            bind_digraph.add_edge(drug_idx, gene_idx, weight = 0.5 * edge_weight_max)
        # Show [drugbank]'s Drug Names in Final Graph
        drug_map_df = pd.read_csv('../data-nci/filtered_data/drug_num_dict.csv')
        drug_map_num_list = list(drug_map_df['drug_num'])
        drugbank_druglist = list(drug_map_df['Drug'])
        drugbank_num_drug_dict = {drug_map_num_list[i - 1] : drugbank_druglist[i - 1] for i in range(1, len(drugbank_druglist)+1)}
        cellline_specific_druglist = list(set(list(cellline_specific_drugbank_df['Drug'])))
        bind_filter_node_name_dict = bind_filter_gene_dict
        for drug_idx in cellline_specific_druglist:
            drugbank_name = drugbank_num_drug_dict[drug_idx]
            # label
            bind_filter_node_name_dict[drug_idx] = drugbank_name
        
        # import pdb; pdb.set_trace()
        
        # RESTRICT GRAPH WITH NODES DEGREES BY USING [nx.restricted_view]
        if edge_remove == True:
            bind_filtered_digraph = nx.restricted_view(bind_digraph, nodes = bind_node_deletion_index_list, edges = bind_edge_deletion_list)
        else:
            bind_filtered_digraph = nx.restricted_view(bind_digraph, nodes = bind_node_deletion_index_list, edges = [])
        print('----- FILTERED GRAPH\'s EDGES WITH HIGHER NODE DEGREES: ' + str(len(bind_filtered_digraph.edges())) + ' -----')
        edges = bind_filtered_digraph.edges()
        weights = [bind_filtered_digraph[u][v]['weight'] for u, v in edges]

        # import pdb; pdb.set_trace()
        nodes = bind_filtered_digraph.nodes()
        bind_filtered_degree_list = []
        for node_idx in nodes:
            node_degree = bind_node_degree_dict[node_idx]
            bind_filtered_degree_list.append(node_degree)
        
        bind_node_degree_list = bind_filtered_degree_list


        # FIND Critical Nodes
        cellline_specific_targetlist = list(set(list(cellline_specific_drugbank_df['Target'])))
        bind_graph_target_nodes = []
        for node in nodes:
            if node in cellline_specific_targetlist:
                bind_graph_target_nodes.append(node)
        print('----- FILTERED GRAPH\'s DRUG TARGET NODES -----')
        print(bind_graph_target_nodes)
        # FIND Critical Edges
        bind_graph_drugtar_links = []
        for u, v in edges:
            if v in cellline_specific_targetlist or u in cellline_specific_targetlist:
                if u <= 2016 and v <= 2016:
                    continue
                else:
                    bind_graph_drugtar_links.append((u, v))
        print('----- FILTERED GRAPH\'s DRUG TARGET LINKS -----')
        print(bind_graph_drugtar_links)

        # import pdb; pdb.set_trace()

        # FIND Critical Paths
        cellline_specific_druglist = list(set(list(cellline_specific_drugbank_df['Drug'])))
        drug_1 = cellline_specific_druglist[0]
        drug_2 = cellline_specific_druglist[1]
        # import pdb; pdb.set_trace()
        # cutoff==3
        cutoff = 3
        path3_num_count = 0
        bind_graph_target_between_3links = []
        path3 = nx.all_simple_paths(bind_filtered_digraph, source=drug_1, target=drug_2, cutoff=3)
        path3_lists = list(path3)
        for path in path3_lists:
            if len(path) < 3:
                print(path)
                continue
            path3_num_count += 1
            for i in range(1, cutoff):
                node_u = path[i - 1]
                node_v = path[i]
                bind_graph_target_between_3links.append((node_u, node_v))
        bind_graph_target_between_3links = list(set(bind_graph_target_between_3links))
        print('----- DRUG BETWEENESS LINKS OF PATH 3: -----')
        print(path3_lists)
        print(bind_graph_target_between_3links)
        print('----- NUMBERS OF PATH 3: -----')
        print(path3_num_count)
        # cutoff==4
        path4_num_count = 0
        path4_weight_count = 0
        cutoff = 4
        bind_graph_target_between_4links = []
        path4 = nx.all_simple_paths(bind_filtered_digraph, source=drug_1, target=drug_2, cutoff=4)
        path4_lists = list(path4)
        # import pdb; pdb.set_trace()
        for path in path4_lists:
            if len(path) < 5:
                continue
            path4_num_count += 1
            for i in range(1, cutoff):
                node_u = path[i - 1]
                node_v = path[i]
                bind_graph_target_between_4links.append((node_u, node_v))
                this_weight = bind_filtered_digraph[node_u][node_v]['weight']
                path4_weight_count += this_weight
        bind_graph_target_between_4links = list(set(bind_graph_target_between_4links))
        print('----- DRUG BETWEENESS LINKS OF PATH 4: -----')
        print(path4_lists)
        print(bind_graph_target_between_4links)
        print('----- NUMBERS OF PATH 4: -----')
        print(path4_num_count)
        print('----- SUM OF WEIGHTS FOR PATH 4: -----')
        print(path4_weight_count)

        if cutoff_range == True:
            cutoff = 5
            path5_num_count = 0
            bind_graph_target_between_5links = []
            path5 = nx.all_simple_paths(bind_filtered_digraph, source=drug_1, target=drug_2, cutoff=5)
            path5_lists = list(path5)
            # path_lists = path_lists[0:3]
            # print(path5_lists)
            for path in path5_lists:
                if path in path4_lists:
                    continue
                path5_num_count += 1
                for i in range(1, cutoff):
                    node_u = path[i - 1]
                    node_v = path[i]
                    bind_graph_target_between_5links.append((node_u, node_v))
            bind_graph_target_between_5links = list(set(bind_graph_target_between_5links))
            print('----- DRUG BETWEENESS LINKS OF PATH 5: -----')
            print(bind_graph_target_between_5links)
            print('----- NUMBERS OF PATH 5: -----')
            print(path5_num_count)

        # DRAW GRAPHS WITH CERTAIN TYPE
        pos = nx.spring_layout(bind_filtered_digraph, 
                                k = 15.0, 
                                center=(1,20),
                                scale = 10, 
                                iterations = 3500,
                                seed = seed)

        # cmap = plt.cm.Wistia
        cmap = plt.cm.Wistia
        dmin = min(bind_node_degree_list)
        dmax = max(bind_node_degree_list)
        cmap2 = plt.cm.OrRd
        emin = min(weights)
        emax = max(weights)
        fig=plt.figure(figsize=(12, 6)) 

        nx.draw_networkx_edges(bind_filtered_digraph, 
                    pos = pos,
                    # arrowsize = 5,
                    alpha = 0.3,
                    width = [float((weight+0.5)**3) / 0.5 for weight in weights],
                    # edge_color = rerange_weights,
                    edge_color = weights,
                    edge_cmap = cmap2
                    )

        # HIGHLIGHT NODES WITH TARGET (effect = circle those critical nodes)
        hilight_nodes = nx.draw_networkx_nodes(bind_filtered_digraph.subgraph(intersection_list), 
                    pos = pos,
                    nodelist = intersection_list, 
                    # node_size = [float((degree+1)**1.5) / 0.1 for degree in intersection_degree_list],
                    node_size = 70,
                    linewidths = 1.0,
                    node_color = 'white'
                    )
        hilight_nodes.set_edgecolor('red')
        # HIGHLIGHT NODES WITH DRUG TARGET ONLY IN THIS SUBNET (effect = circle those critical nodes)
        hilight_genes = nx.draw_networkx_nodes(bind_filtered_digraph.subgraph(bind_graph_target_nodes), 
                    pos = pos,
                    nodelist = bind_graph_target_nodes, 
                    # node_size = [float(bind_node_degree_dict[gene_idx]**1.75) / 3 for gene_idx in bind_graph_target_nodes],
                    node_size = 70,
                    linewidths = 1.0,
                    node_color = 'white'
                    )
        hilight_genes.set_edgecolor('blue')

        nx.draw_networkx_nodes(bind_filtered_digraph, 
                    pos = pos,
                    node_color = bind_node_degree_list,
                    nodelist = nodes, 
                    # node_size = [float(degree**1.75) / 3 for degree in bind_node_degree_list],
                    node_size = 40,
                    alpha = 0.9,
                    cmap = cmap
                    )

        # HIGHLIGHT DRUUGS
        # Shape = "V" And Plot
        hilight_drugs = nx.draw_networkx_nodes(bind_filtered_digraph.subgraph(cellline_specific_druglist), 
                    pos = pos,
                    nodelist = cellline_specific_druglist, 
                    node_size = [float(bind_node_degree_dict[drug_idx]**1.5) / 0.05 for drug_idx in cellline_specific_druglist],
                    linewidths = 1.5,
                    alpha = 1,
                    node_color = 'orange',
                    node_shape = 'v'
                    )
        hilight_drugs.set_edgecolor('gray')

        # HIGHLIGHT [drug-target-betweeness] EDGES (cutoff==4)
        nx.draw_networkx_edges(bind_filtered_digraph,
                    pos = pos,
                    edgelist = bind_graph_target_between_4links,
                    connectionstyle = 'arc3, rad = 0.3',
                    edge_color = 'lightgreen',
                    width = 1.5)
        
        # HIGHLIGHT [drug-target-betweeness] EDGES (cutoff==5)
        if cutoff_range == True:
            nx.draw_networkx_edges(bind_filtered_digraph,
                        pos = pos,
                        edgelist = bind_graph_target_between_5links,
                        connectionstyle = 'arc3, rad = 0.3',
                        edge_color = 'orange',
                        width = 0.75)

        # HIGHLIGHT [drug-target] EDGES
        nx.draw_networkx_edges(bind_filtered_digraph,
                    pos = pos,
                    edgelist = bind_graph_drugtar_links,
                    connectionstyle = 'arc3, rad = 0.3',
                    edge_color = 'lightblue',
                    width = 2)

        nx.draw_networkx_labels(bind_filtered_digraph,
                    pos = pos,
                    labels = bind_filter_node_name_dict,
                    font_size = 5.0
                    )


        # # import pdb; pdb.set_trace()
        # degree = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = dmin, vmax = dmax))
        # dcl = plt.colorbar(degree)
        # dcl.ax.tick_params(labelsize = 8)
        # dcl.ax.set_ylabel('Nodes Degree')
        # edge = plt.cm.ScalarMappable(cmap = cmap2, norm = plt.Normalize(vmin = emin, vmax = emax))
        # ecl = plt.colorbar(edge)
        # ecl.ax.tick_params(labelsize = 8)
        # ecl.ax.set_ylabel('Edges Weight')
        
        drugA = drugbank_num_drug_dict[cellline_specific_druglist[0]]
        drugB = drugbank_num_drug_dict[cellline_specific_druglist[1]]
        titlename = drugA + ' & ' + drugB + ' target on filtered ' + str(filter_node_count) \
            + ' genes on ' + cellline_name + ' cell line'
        plt.title(titlename) 
        # plt.show()
        if cellline_name == 'A549/ATCC':
            cellline_name = 'A549'
        if topmin_loss == True:
            if cutoff_range == True:
                filename = '../analysis_nci/plot_topmin_'  + cellline_name + '_top_' + str(top_n) + '_range5.png'
            else:
                filename = '../analysis_nci/plot_topmin_'  + cellline_name + '_top_' + str(top_n) + '_range4.png'
        else:
            if cutoff_range == True:
                filename = '../analysis_nci/plot_bottommin_'  + cellline_name + '_bottom_' + str(top_n) + '_range5.png'
            else:
                filename = '../analysis_nci/plot_bottommin_'  + cellline_name + '_bottom_' + str(top_n) + '_range4.png'
        plt.savefig(filename, dpi = 1200)
        plt.close()
        return path4_weight_count



if __name__ == "__main__":
    ###### BASICAL PARAMETERS IN FILES
    # NetAnalyse().statistic_net()
    
    #####
    node_threshold = 1.0
    edge_threshold = 0.1
    # cellline_name = '786-0'
    # cellline_name = 'A498'
    # cellline_name = 'A549/ATCC'
    # cellline_name = 'ACHN'
    # cellline_name = 'BT-549'
    # cellline_name = 'CAKI-1'
    # cellline_name = 'CCRF-CEM'
    # cellline_name = 'COLO 205'
    # cellline_name = 'DU-145'
    # cellline_name = 'EKVX'
    # cellline_name = 'HCC-2998'
    # cellline_name = 'HCT-116'
    # cellline_name = 'HCT-15'
    # cellline_name = 'HOP-62'
    # cellline_name = 'HOP-92'
    # cellline_name = 'HS 578T'
    # cellline_name = 'HT29'
    # cellline_name = 'IGROV1'
    # cellline_name = 'K-562'
    # cellline_name = 'KM12'
    # cellline_name = 'LOX IMVI'
    # cellline_name = 'M14'
    # cellline_name = 'MCF7'
    # cellline_name = 'MDA-MB-231/ATCC'
    # cellline_name = 'MDA-MB-468'
    # cellline_name = 'MOLT-4'
    # cellline_name = 'NCI-H226'
    # cellline_name = 'NCI-H23'
    # cellline_name = 'NCI-H322M'
    # cellline_name = 'NCI-H460'
    # cellline_name = 'NCI-H522'
    # cellline_name = 'OVCAR-3'
    # cellline_name = 'OVCAR-4'
    # cellline_name = 'OVCAR-5'
    # cellline_name = 'OVCAR-8'
    # cellline_name = 'PC-3'
    # cellline_name = 'RPMI-8226'
    # cellline_name = 'RXF 393'
    # cellline_name = 'SF-268'
    # cellline_name = 'SF-295'
    # cellline_name = 'SF-539'
    # cellline_name = 'SK-MEL-2'
    # cellline_name = 'SK-MEL-28'
    # cellline_name = 'SK-MEL-5'
    # cellline_name = 'SK-OV-3'
    # cellline_name = 'SN12C'
    # cellline_name = 'SNB-75'
    # cellline_name = 'SR'
    # cellline_name = 'SW-620'
    cellline_name = 'T-47D'
    # cellline_name = 'TK-10'
    # cellline_name = 'U251'
    # cellline_name = 'UACC-257'
    # cellline_name = 'UACC-62'
    # cellline_name = 'UO-31'

    intersection_name_list, intersection_list, intersection_degree_list = \
        NetAnalyse().plot_target_gene(node_threshold)

    # SET TRAINING/TEST SET
    top_k = 20
    seed = 187

    # topmin_loss = False
    topmin_loss = True

    cutoff_range = True
    # cutoff_range = False
    # GET TESTLOSS TOP/BOTTOM Object List
    testloss_topminobj_list, testloss_bottomminobj_list = Specify().cancer_cellline_specific(top_k, cellline_name)

    path4_weight_count_list = []
    top_n_list = [1, 2, 3, 4, 5]
    for top_n in top_n_list:
        testloss_topminobj = testloss_topminobj_list[top_n - 1]
        testloss_bottomminobj = testloss_bottomminobj_list[top_n - 1]

        cellline_specific_drugbank_df = NetAnalyse().drug_target_interaction(cellline_name, topmin_loss, testloss_topminobj, testloss_bottomminobj)

        path4_weight_count = NetAnalyse().plot_net2(node_threshold, edge_threshold,
            intersection_list, intersection_degree_list, cellline_specific_drugbank_df, topmin_loss, seed, cellline_name, top_n, cutoff_range)
        path4_weight_count_list.append(path4_weight_count)

    print('----- LIST FOR SUM OF WEIGHTS FOR PATH 4: -----')
    print(path4_weight_count_list)