import os
import sys
import math
import random
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append('.')
sys.path.append('../../')
sys.path.append('../network_model/')

notebook_path = os.getcwd()
project_path = os.path.abspath(os.path.join(notebook_path, '..'))
sys.path.append(project_path)

from network_model.network_classes import Network


class NetworksContainer:
    def __init__(self, 
                 n_networks, n_nodes, alpha, beta, 
                 delta_in, delta_out, homophily, 
                 n_commons, n_influencers, n_bots, n_fact_checkers,
                 prob_complaint, prob_infection, prob_vaccination,
                 prob_cure, prob_influencer,
                 exp_decay=True, user_block=False, prob_echo=0.0,
                 epochs=0):
        self.n_networks = n_networks
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.beta = beta
        self.delta_in = delta_in
        self.delta_out = delta_out
        self.homophily = homophily
        self.prob_complaint = prob_complaint        
        self.prob_infection = prob_infection        
        self.prob_vaccination = prob_vaccination    
        self.prob_cure = prob_cure                  
        self.prob_influencer = prob_influencer      
        self.exp_decay = exp_decay 
        self.user_block = user_block 
        self.prob_echo = prob_echo
        self.n_commons = n_commons
        self.n_influencers = n_influencers
        self.n_bots = n_bots
        self.n_fact_checkers = n_fact_checkers
        self.network_list = []
        self.epochs = epochs


    def create_network_list(self, n_initial_infected_nodes):
        for i in range(self.n_networks):
            network = Network(n_nodes=self.n_nodes, 
                              alpha=self.alpha, beta=self.beta, 
                              delta_in=self.delta_in, delta_out=self.delta_out, 
                              homophily=self.homophily, 
                              n_commons=self.n_commons, 
                              n_influencers=self.n_influencers, 
                              n_bots=self.n_bots,
                              n_fact_checkers=self.n_fact_checkers,
                              prob_complaint=self.prob_complaint, 
                              prob_infection=self.prob_infection, 
                              prob_vaccination=self.prob_vaccination,
                              prob_cure=self.prob_cure, 
                              prob_influencer=self.prob_influencer, 
                              exp_decay=self.exp_decay, 
                              user_block=self.user_block, 
                              prob_echo=self.prob_echo)
            network.create_graph()
            initial_infected_nodes = self.get_initial_infected_nodes(network, n_initial_infected_nodes)
            network.set_initial_infected_nodes(initial_infected_nodes)
            self.network_list.append(network)


    def get_initial_infected_nodes(self, network, n_initial_infected_nodes):
        initial_infected_nodes = []
        count = 0
        while (count < n_initial_infected_nodes):
            infected_node = random.sample(range(0, self.n_nodes), 1)
            index = infected_node[0]
            if network.nodes[index].type_sirv == "neutral":
                initial_infected_nodes.append(index)
                count += 1
        return initial_infected_nodes


    def set_probabilities(self, prob_influencer, prob_infection, prob_vaccination, prob_cure):
        self.prob_influencer = prob_influencer
        self.prob_infection = prob_infection
        self.prob_vaccination = prob_vaccination
        self.prob_cure = prob_cure
        for i in range(self.n_networks):
            self.network_list[i].prob_influencer = prob_influencer
            self.network_list[i].prob_infection = prob_infection
            self.network_list[i].prob_vaccination = prob_vaccination
            self.network_list[i].prob_cure = prob_cure
            self.network_list[i].set_nodes_probabilities()

    
    def set_prob_echo(self, prob_echo):
        self.prob_echo = prob_echo
        for i in range(self.n_networks):
            self.network_list[i].set_prob_echo(prob_echo)


    def set_prob_complaint(self, prob_complaint):
        self.prob_complaint = prob_complaint
        for i in range(self.n_networks):
            self.network_list[i].set_prob_complaint(prob_complaint)


    def set_influencers_to_vaccinated(self):
        for i in range(self.n_networks):
            for j in range(self.network_list[i].n_nodes):
                if (self.network_list[i].nodes[j].type_role == "influencer"):
                    self.network_list[i].nodes[j].type_sirv = "vaccinated"
