import random
import numpy as np
import networkx as nx


class InvalidParametersException(Exception):
    """Raised when the sum of alpha and beta parameters is greater than 1"""
    pass


def sample_from_normal_distribution(mean=0.5, std=0.5):
    sample = np.random.normal(mean, std)
    # limito i valori tra 0 e 1
    sample = np.clip(sample, 0.0, 1.0)
    return sample


def create_graph(n_nodes, alpha, beta, delta_in, delta_out, homophily):

    if (alpha + beta) > 1:
        raise InvalidParametersException
    
    
    G = nx.DiGraph()
    G.add_node(0)
    #opinion = sample_from_normal_distribution()
    opinion = random.random()
    nx.set_node_attributes(G, {0 : {'opinion': opinion, 'id': 0}})

    while len(G) < n_nodes:

        prob = random.random()

        if prob < alpha:
            # alpha
            add_new_node_with_out_edge(G, delta_in, homophily)
        elif prob < alpha + beta:
            # beta
            add_new_edge_between_old_nodes(G, delta_in, delta_out, homophily)
        else:
            # gamma (dato che alpha+beta+gamma=1 non serve specificare gamma)
            add_new_node_with_in_edge(G, delta_out, homophily)

    set_edges_id(G)

    return G


def add_new_node_with_out_edge(G, delta_in, homophily):
    v = len(G)
    #opinion = sample_from_normal_distribution()
    opinion = random.random()
    
    if len(G) == 1:
        G.add_edges_from([(v, 0)])
        nx.set_node_attributes(G, {v : {'opinion': opinion, 'id': v}})
        return
        
    w = choose_node_by_homophilic_in_degree(G, opinion, delta_in, homophily)
    # evito self-loop
    while (v == w):
        w = choose_node_by_homophilic_in_degree(G, opinion, delta_in, homophily)
    G.add_edges_from([(v, w)])
    nx.set_node_attributes(G, {v : {'opinion': opinion, 'id': v}})


def add_new_edge_between_old_nodes(G, delta_in, delta_out, homophily):
    if len(G) == 1:
        return
            
    v = choose_node_by_out_degree(G, delta_out)
    w = choose_node_by_homophilic_in_degree(G, G.nodes[v]['opinion'], delta_in, homophily)
    # evito self-loop
    while (v == w):
        w = choose_node_by_homophilic_in_degree(G, G.nodes[v]['opinion'], delta_in, homophily)
    G.add_edges_from([(v, w)])


def add_new_node_with_in_edge(G, delta_out, homophily):
    w = len(G)
    #opinion = sample_from_normal_distribution()
    opinion = random.random()
    
    if len(G) == 1:
        G.add_edges_from([(0, w)])
        nx.set_node_attributes(G, {w : {'opinion': opinion, 'id': w}})
        return
        
    v = choose_node_by_homophilic_out_degree(G, opinion, delta_out, homophily)
    # evito self-loop
    while (w == v):
        v = choose_node_by_homophilic_out_degree(G, opinion, delta_out, homophily)
    G.add_edges_from([(v, w)])
    nx.set_node_attributes(G, {w : {'opinion': opinion, 'id': w}}) 


def choose_node_by_homophilic_in_degree(G, opinion, delta_in, homophily):
    nodes = np.array([node for node in range(len(G))])
    probabilities = generate_homophilic_in_probabilities(G, opinion, delta_in, homophily)
    return np.random.choice(nodes, 1, False, probabilities)[0]


def choose_node_by_out_degree(G, delta_out): 
    nodes = np.array([node for node in range(len(G))])
    probabilities = generate_out_probabilities(G, delta_out)
    return np.random.choice(nodes, 1, False, probabilities)[0]       


def choose_node_by_homophilic_out_degree(G, opinion, delta_out, homophily):
    nodes = np.array([node for node in range(len(G))])
    probabilities = generate_homophilic_out_probabilities(G, opinion, delta_out, homophily)
    return np.random.choice(nodes, 1, False, probabilities)[0]    


def generate_homophilic_in_probabilities(G, opinion, delta_in, homophily):
    homophily_factors = np.array([get_homophily_factor(opinion, G.nodes[node]['opinion'], homophily) 
                                 for node in range(len(G))])
    in_degrees = np.array([G.in_degree(node) for node in range(len(G))])
    
    return ((np.multiply(homophily_factors, in_degrees) + delta_in) 
            / np.sum(np.multiply(homophily_factors, in_degrees) + delta_in))


def generate_out_probabilities(G, delta_out):
    out_degrees = np.array([G.out_degree(node) for node in range(len(G))])
    return (out_degrees + delta_out) / np.sum(out_degrees + delta_out)


def generate_homophilic_out_probabilities(G, opinion, delta_out, homophily):
    homophily_factors = np.array([get_homophily_factor(opinion, G.nodes[node]['opinion'], homophily) 
                                 for node in range(len(G))])
    out_degrees = np.array([G.out_degree(node) for node in range(len(G))])
    
    return ((np.multiply(homophily_factors, out_degrees) + delta_out) 
            / np.sum(np.multiply(homophily_factors, out_degrees) + delta_out))


def get_homophily_factor(opinion_1, opinion_2, homophily):
    # homophily_factor = homophily if (opinion_1 == opinion_2) else (1 - homophily)
    # uso il valore assoluto in modo da non avere distanze negative
    distance = np.abs(opinion_1 - opinion_2)
    # limito i valori ammissibili tra 0 e 1
    # Se il valore di homophily è minore di 0.5, allora se i nodi
    # sono distanti la probabilità di collegarli deve crescere.
    # Esempio: homophily 0.8, con distanza 0 -> 0.8 - 0 = 0.8
    # Esempio: homophily 0.8, con distanza 1 -> 0.8 - 1 = 0.2
    # Esempio: homophily 0.2, con distanza 0 -> 0.2 - 0 = 0.2
    # Esempio: homophily 0.2, con distanza 1 -> 0.2 - 1 = -0.8 -> 0.8 in valore assoluto
    homophily_factor = np.clip(np.abs(homophily - distance), 0.0, 1.0)
    return homophily_factor


def set_edges_id(G):

    id_edge = 0
    id_edges = []
    edges = G.edges()
    for i in range(len(edges)):
        id_edges.append(id_edge)
        id_edge = id_edge + 1

    i = 0
    attrs = {}
    for edge in G.edges():
        attrs = {**attrs, edge: {'id': id_edges[i]}}
        i = i + 1

    nx.set_edge_attributes(G, attrs)

