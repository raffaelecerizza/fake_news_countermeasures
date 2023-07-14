import sys
import math
import random
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

sys.path.append('.')
sys.path.append('../../')

from graph_creation import create_graph


class Edge:
    def __init__(self, start, end, id):
        self.id = id
        self.start = start
        self.end = end
    
    def __str__(self):
        string = f"id: {self.id} \
            \nstart: {self.start} \
            \nend: {self.end}" 
        return  string

     
class Node:
    def __init__(self, id, opinion, 
                 prob_complaint, prob_infection, prob_vaccination,
                 prob_cure, prob_influencer,
                 exp_decay=True, prob_echo=0.0):
        self.id = id
        self.opinion = opinion
        self.type_sirv = "neutral"
        self.type_role = "common"
        self.blocked = False
        self.number_complaints = 0
        self.users_complained = []
        self.prob_complaint = prob_complaint
        self.prob_infection = prob_infection
        self.prob_vaccination = prob_vaccination
        self.prob_cure = prob_cure
        self.prob_influencer = prob_influencer
        self.time_infection = 0
        self.time_vaccination = 0
        self.time_cure = 0
        # nodi verso cui il nodo corrente ha un arco
        self.nodes_connected_to = []
        self.messages = {}
        self.prev_timestep = -1
        self.next_timestep = 0
        self.exp_decay = exp_decay
        self.prob_echo = prob_echo
        self.infected_from_echo = False
        self.vaccinated_from_echo = False
        self.cured_from_echo = False
 

    def update_type_sirv(self, dict_messages, dict_influencers, 
                        timestep, file_complaint):
        # La lunghezza di dict_messages è pari al numero di utenti seguiti.
        for i in range(len(dict_messages)):
            followed_node = self.nodes_connected_to[i].id

            # Fattore che rappresenta la distanza di opinioni tra i nodi.
            # Si usa il valore assoluto per evitare valori negativi di differenza.
            # Si sottrae la differenza a 1 in modo che i nodi con opinioni
            # simili abbiano un fattore di probabilità maggiore. 
            opinion_diff = 1 - np.abs(self.opinion - self.nodes_connected_to[i].opinion)

            # Recupero tutti gli istanti di tempo in cui il nodo seguito
            # ha pubblicato un messaggio.
            keys_list = list(dict_messages[followed_node].keys())

            # Considero solo gli istanti di tempo compresi fra l'ultimo istante
            # in cui il nodo corrente è stato attivo e l'istante di tempo
            # precedente a quello corrente.
            for j in range(self.prev_timestep, timestep, 1):
 
                # Se l'istante di tempo è fra quelli in cui l'utente seguito
                # ha pubblicato, allora si valutano gli effetti.
                if j in keys_list:

                    # Prendo il messaggio che l'utente seguito ha pubblicato
                    # nell'istante di tempo j.
                    message = dict_messages[followed_node][j]

                    # Se un utente neutrale vede un messaggio infetto, allora 
                    # con prob_infection * opinion_diff diventerà infetto.
                    if self.type_sirv == "neutral" and message == "infected":
                        prob = random.random()
                        # Se il messaggio arriva da un influencer, la probabilità di
                        # essere infettato aumenta.
                        if dict_influencers[followed_node] == True:
                            prob_infection = self.prob_infection * opinion_diff
                            # limito la somma delle probabilità a 1
                            prob_infection = np.clip(prob_infection + 
                                                     self.prob_influencer, 
                                                     a_min=None, a_max=1.0)
                        else:
                            prob_infection = self.prob_infection * opinion_diff
                        if prob < prob_infection:
                            self.type_sirv = "infected"
                            self.time_infection = timestep

                    # Se un utente neutrale vede un messaggio infetto, allora 
                    # con prob_vaccination * opinion_diff diventerà vaccinato.
                    if self.type_sirv == "neutral" and message == "infected":
                        prob = random.random()
                        prob_vaccination = self.prob_vaccination * opinion_diff
                        if prob < prob_vaccination:
                            self.type_sirv = "vaccinated"
                            self.time_vaccination = timestep

                    # Se un utente neutrale vede un messaggio vaccinato, allora 
                    # con prob_vaccination * opinion_diff diventerà vaccinato.
                    if self.type_sirv == "neutral" and message == "vaccinated":
                        prob = random.random()
                        # Se il messaggio arriva da un influencer, la probabilità di
                        # essere vaccinato aumenta.
                        if dict_influencers[followed_node] == True:
                            prob_vaccination = (self.prob_vaccination * 
                                                opinion_diff)
                            # limito la somma delle probabilità a 1
                            prob_vaccination = np.clip(prob_vaccination + 
                                                       self.prob_influencer, 
                                                       a_min=None, a_max=1.0)
                        else:
                            prob_vaccination = (self.prob_vaccination * 
                                                opinion_diff) 
                        if prob < prob_vaccination:
                            self.type_sirv = "vaccinated"
                            self.time_vaccination = timestep        

                    # Se un utente infetto vede un messaggio vaccinato, allora 
                    # con prob_cure * opinion_diff diventerà curato.
                    # Non modifico lo stato dei bot.
                    if (
                        self.type_sirv == "infected" and 
                        self.type_role != "bot" and message == "vaccinated"
                    ):
                        prob = random.random()
                        # Se il messaggio arriva da un influencer, la probabilità di
                        # essere curato aumenta.
                        if dict_influencers[followed_node] == True:
                            prob_cure = self.prob_cure * opinion_diff
                            # limito la somma delle probabilità a 1
                            prob_cure = np.clip(prob_cure + 
                                                self.prob_influencer, 
                                                a_min=None, a_max=1.0)
                        else:
                            prob_cure = self.prob_cure * opinion_diff 
                        if prob < prob_cure:
                            self.type_sirv = "cured"
                            self.time_cure = timestep

                    # Se un utente vaccinato vede un messaggio infetto, allora
                    # con prob_complaint invia un complaint all'utente infetto.
                    # La gestione dei complaint viene fatta dalla rete.    
                    if self.type_sirv == "vaccinated" and message == "infected":
                        prob = random.random()
                        if (
                            prob < self.prob_complaint and 
                            (followed_node not in self.users_complained)
                        ):
                            self.users_complained.append(followed_node)
                            file_complaint(followed_node)

        # Se una frazione di utenti seguiti con un certo type_sirv supera il 
        # threshold prob_echo, allora anche il nodo assume lo stesso type_sirv.
        # Non modifico lo stato di bot e fact checkers.
        if (
            self.prob_echo > 0 and self.type_role != "bot" and 
            self.type_role != "fact checker"
        ):
            # Applico il meccanismo delle echo chamber solo se la soglia
            # è maggiore di 0.50.
            if (self.prob_echo > 0.50):
                self.update_echo_chamber()


    def publish_message(self, timestep):
        # Se l'utente è curato, allora non prova a curare o infettare gli altri.
        if self.type_sirv == "cured" or self.blocked == True:
            self.messages[timestep] = ""
        else:
            # Il messaggio pubblicato è semplicemente il proprio stato 
            # del modello SIRV.
            self.messages[timestep] = self.type_sirv


    def update_block_status(self):
        if self.number_complaints >= 3:
            self.blocked = True


    def update_echo_chamber(self):
        # Conta il numero di nodi con type_sirv "infected" e "vaccinated" tra i nodi connessi
        count_infected = sum(1 for node in self.nodes_connected_to 
                             if node.type_sirv == "infected")
        count_vaccinated = sum(1 for node in self.nodes_connected_to 
                               if node.type_sirv == "vaccinated")
     
        # Calcola le frazioni di nodi "infected" e "vaccinated"
        if (len(self.nodes_connected_to) != 0):
            fraction_infected = count_infected / len(self.nodes_connected_to)
            fraction_vaccinated = count_vaccinated / len(self.nodes_connected_to)

            # Aggiorna il type_sirv del nodo se supera la soglia prob_echo
            if (
                fraction_infected >= self.prob_echo and
                self.type_sirv == "neutral"
            ):
                self.type_sirv = "infected"
                self.infected_from_echo = True
                self.vaccinated_from_echo = False
                self.cured_from_echo = False
            elif (
                fraction_vaccinated >= self.prob_echo and 
                self.type_sirv == "neutral"
            ):
                self.type_sirv = "vaccinated"
                self.infected_from_echo = False
                self.vaccinated_from_echo = True
                self.cured_from_echo = False
            elif (
                fraction_vaccinated >= self.prob_echo and 
                self.type_sirv == "infected"
            ):
                self.type_sirv = "cured"
                self.infected_from_echo = False
                self.vaccinated_from_echo = False
                self.cured_from_echo = True


    def update_timestep(self, timestep):
        self.prev_timestep = timestep
        # Assumo che i bot siano attivi in ogni istante di tempo.
        if self.type_role != "bot" and self.exp_decay == True:
            lambda_value = random.uniform(0.01, 0.05)
            # Arrotondo al più piccolo intero che sia maggiore o uguale
            # al timestep corrente. Uso il massimo tra 1 e il round in modo
            # da non avere il prossimo timestep uguale a quello corrente.
            decay_factor = math.exp(-lambda_value * timestep)
            self.next_timestep = timestep + max(1, round(1 / decay_factor))
        else:
            self.next_timestep = timestep + 1


    def update(self, timestep, file_complaint):
        # Eseguo l'update solo se l'utente è attivo nell'istante di tempo.
        dict_messages = {}
        dict_influencers = {}
        for i in range(len(self.nodes_connected_to)):
            node = self.nodes_connected_to[i].id
            messages = self.nodes_connected_to[i].get_messages()
            dict_messages[node] = messages
            if self.nodes_connected_to[i].type_role == "influencer":
                dict_influencers[node] = True
            else:
                dict_influencers[node] = False
        if timestep == self.next_timestep:
            self.update_type_sirv(dict_messages, dict_influencers, 
                                 timestep, file_complaint)
            self.publish_message(timestep)
            self.update_timestep(timestep)
    

    def get_messages(self):
        return self.messages
    

    def reset_node(self):
        self.type_sirv = "neutral"
        self.blocked = False
        self.number_complaints = 0
        self.users_complained = []
        self.time_infection = 0
        self.time_vaccination = 0
        self.time_cure = 0
        self.messages = {}
        self.prev_timestep = -1
        self.next_timestep = 0


    def __str__(self):
        string = f"id: {self.id} \
                 \ntype_sirv: {self.type_sirv} \
                 \ntype_role: {self.type_role} \
                 \nblocked: {self.blocked} \
                 \nnumber_complaints: {self.number_complaints} \
                 \nprob_complaint: {self.prob_complaint} \
                 \nprob_infection: {self.prob_infection} \
                 \nprob_vaccination: {self.prob_vaccination} \
                 \nprob_cure: {self.prob_cure} \
                 \nprob_influencer: {self.prob_influencer} \
                 \ntime_infection: {self.time_infection} \
                 \ntime_vaccination: {self.time_vaccination} \
                 \ntime_cure: {self.time_cure}" 
        return  string


class Network:
    id = itertools.count(0)
    def __init__(self, n_nodes, alpha, beta, 
                 delta_in, delta_out, homophily, 
                 n_commons, n_influencers, n_bots, n_fact_checkers,
                 prob_complaint, prob_infection, prob_vaccination,
                 prob_cure, prob_influencer,
                 exp_decay=True, user_block=False, prob_echo=0.0):
        # Incremento l'id ogni volta che creo una nuova rete.
        self.id = next(Network.id)
        self.G = nx.DiGraph()
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
    
        self.nodes = []
        self.edges = []
        self.n_commons = n_commons
        self.n_influencers = n_influencers
        self.n_bots = n_bots
        self.n_fact_checkers = n_fact_checkers
        self.bots = []
        self.fact_checkers = []
        self.influencers = []
        self.initial_infected_nodes = []
        self.perc_neutral = []
        self.perc_infected = []
        self.perc_vaccinated = []
        self.perc_cured = []
        self.infected_from_echo = 0
        self.vaccinated_from_echo = 0
        self.cured_from_echo = 0
        self.n_blocked_nodes = 0
        self.global_timestep = -1


    def create_graph(self):
        self.G = create_graph(self.n_nodes,
                              self.alpha,
                              self.beta,
                              self.delta_in,
                              self.delta_out,
                              self.homophily)
        self.create_nodes()
        self.create_edges()
        self.add_edges_to_nodes()
        self.create_influencers()
        self.create_bots()
        self.create_fact_checkers()


    def create_nodes(self):
        nodes = self.G.nodes()
        for i in list(nodes):
            node = Node(nodes[i]['id'], nodes[i]['opinion'],
                        self.prob_complaint, self.prob_infection,
                        self.prob_vaccination, self.prob_cure,
                        self.prob_influencer)
            self.nodes.append(node)


    def create_edges(self):
        edges = self.G.edges.data()
        for edge in edges:
            self.edges.append(Edge(edge[0], edge[1], edge[2]['id']))


    def add_edges_to_nodes(self):
        for i in range(len(self.nodes)):
            nodes_connected_to = []
            for j in range(len(self.edges)):
                if self.edges[j].start == self.nodes[i].id:
                    nodes_connected_to.append(self.nodes[self.edges[j].end])
            self.nodes[i].nodes_connected_to = nodes_connected_to     


    def create_influencers(self):
        influencers_list = []
        nodes = np.zeros(len(self.nodes))
        # per ogni nodo conto quanti archi entranti ha
        for i in range(len(self.edges)):
            nodes[self.edges[i].end] += 1
        # riordino con ordine decrescente il vettore col numero
        # di archi entranti di ciascun nodo
        order = np.argsort(nodes, axis=0)[::-1]
        # i primi nodi per archi entranti sono gli influencer
        for j in range(self.n_influencers):
            self.nodes[order[j]].type_role = "influencer"
            influencers_list.append(order[j]) 
        self.influencers = influencers_list

    
    def create_bots(self):
        count = 0
        bots_list = []
        while count < self.n_bots:
            # creo bot con indici casuali
            random_index = np.random.randint(0, self.n_nodes)
            # assumo che i bot non siano né influencer né fact checkers 
            # e faccio in modo che i bot siano diversi dai nodi inizialmente infetti
            if (
                self.nodes[random_index].type_role != "influencer" and 
                self.nodes[random_index].type_role != "fact checker" and 
                self.nodes[random_index].type_sirv != "infected"
            ):
                self.nodes[random_index].type_role = "bot"
                # assumo che i bot siano sempre infetti
                self.nodes[random_index].type_sirv = "infected"
                bots_list.append(random_index)
                count += 1 
        self.bots = bots_list
        self.compute_initial_data_analysis()


    def create_fact_checkers(self):
        count = 0
        fact_checkers_list = []
        while count < self.n_fact_checkers:
            # creo fact checker con indici casuali
            random_index = np.random.randint(0, self.n_nodes)
            # assumo che i fact checker non siano né influencer né bot 
            # e faccio in modo che siano diversi dai nodi inizialmente infetti
            if (
                self.nodes[random_index].type_role != "influencer" and 
                self.nodes[random_index].type_role != "bot" and
                self.nodes[random_index].type_sirv != "infected"
            ):
                self.nodes[random_index].type_role = "fact checker"
                # assumo che i fact cheker siano sempre vaccinati
                self.nodes[random_index].type_sirv = "vaccinated"
                fact_checkers_list.append(random_index)
                count += 1 
        self.fact_checkers = fact_checkers_list
        self.compute_initial_data_analysis()


    def set_initial_infected_nodes(self, list):
        self.initial_infected_nodes = list
        for i in range(len(list)):
            node_index = list[i]
            self.nodes[node_index].type_sirv = "infected"
        self.compute_initial_data_analysis()


    def set_bots(self, list):
        self.n_bots = len(list)
        self.bots = list
        for i in range(len(list)):
            node_index = list[i]
            self.nodes[node_index].type_role = "bot"
            self.nodes[node_index].type_sirv = "infected"
        self.compute_initial_data_analysis()


    def set_fact_checkers(self, list):
        self.n_fact_checkers = len(list)
        self.fact_checkers = list
        for i in range(len(list)):
            node_index = list[i]
            self.nodes[node_index].type_role = "fact checker"
            self.nodes[node_index].type_sirv = "vaccinated"
        self.compute_initial_data_analysis()
            
 
    def plot_graph_sirv(self):
        color_map = []
        for i in range(self.n_nodes):
            if self.nodes[i].type_sirv == "neutral":
                color_map.append("white")    
            if self.nodes[i].type_sirv == "infected":
                color_map.append("red") 
            if self.nodes[i].type_sirv == "vaccinated":
                color_map.append("green")
            if self.nodes[i].type_sirv == "cured":
                color_map.append("blue")  
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='neutral', 
                   markerfacecolor='white', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='infected', 
                   markerfacecolor='red', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='vaccinated', 
                   markerfacecolor='green', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='cured', 
                   markerfacecolor='blue', markeredgecolor='black', 
                   markersize=15),
        ]      
        pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, node_color=color_map, font_color='black', 
                edgecolors='black', node_size=25, font_size=10, pos=pos,
                width=0.3, arrowsize=5)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                   loc='upper left')
        plt.show()


    def plot_graph_sirv_with_labels(self):
        color_map = []
        for i in range(self.n_nodes):
            if self.nodes[i].type_sirv == "neutral":
                color_map.append("white")    
            if self.nodes[i].type_sirv == "infected":
                color_map.append("red") 
            if self.nodes[i].type_sirv == "vaccinated":
                color_map.append("green")
            if self.nodes[i].type_sirv == "cured":
                color_map.append("blue")    
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='neutral', 
                   markerfacecolor='white', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='infected', 
                   markerfacecolor='red', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='vaccinated', 
                   markerfacecolor='green', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='cured', 
                   markerfacecolor='blue', markeredgecolor='black', 
                   markersize=15),
        ]            
        labels = nx.get_node_attributes(self.G, 'id')
        pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, labels=dict([index for index in enumerate(labels)]),
                node_color=color_map, font_color='black', edgecolors='black',
                node_size=200, font_size=7, pos=pos,
                width=0.5, arrowsize=5)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                   loc='upper left')
        plt.show()


    def plot_graph_role(self):
        color_map = []
        for i in range(self.n_nodes):
            if self.nodes[i].type_role == "common":
                color_map.append("white")
            if self.nodes[i].type_role == "influencer":
                color_map.append("orange")      
            if self.nodes[i].type_role == "bot":
                color_map.append("red")
            if self.nodes[i].type_role == "fact checker":
                color_map.append("green")    
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='common', 
                   markerfacecolor='white', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='bot', 
                   markerfacecolor='red', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='influencer', 
                   markerfacecolor='orange', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='fact checker', 
                   markerfacecolor='green', markeredgecolor='black', 
                   markersize=15),       
        ]   
        pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, node_color=color_map, font_color='black', 
                edgecolors='black', node_size=30, font_size=10, pos=pos,
                width=0.3, arrowsize=5)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                   loc='upper left')
        plt.show()


    def plot_graph_role_with_labels(self):
        color_map = []
        for i in range(self.n_nodes):
            if self.nodes[i].type_role == "common":
                color_map.append("white")
            if self.nodes[i].type_role == "influencer":
                color_map.append("orange")      
            if self.nodes[i].type_role == "bot":
                color_map.append("red") 
            if self.nodes[i].type_role == "fact checker":
                color_map.append("green")   
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='common', 
                   markerfacecolor='white', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='bot', 
                   markerfacecolor='red', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='influencer', 
                   markerfacecolor='orange', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='fact checker', 
                   markerfacecolor='green', markeredgecolor='black', 
                   markersize=15),  
        ]     
        labels = nx.get_node_attributes(self.G, 'id')
        pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, labels=dict([index for index in enumerate(labels)]),
                node_color=color_map, font_color='black', edgecolors='black',
                node_size=200, font_size=7, pos=pos,
                width=0.5, arrowsize=5)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                   loc='upper left')
        plt.show()


    def plot_graph_block(self):
        color_map = []
        for i in range(self.n_nodes):
            if self.nodes[i].blocked == False:
                color_map.append("white")
            if self.nodes[i].blocked == True:
                color_map.append("red")  
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='not blocked', 
                   markerfacecolor='white', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='blocked', 
                   markerfacecolor='red', markeredgecolor='black', 
                   markersize=15),    
        ]   
        pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, node_color=color_map, font_color='black', 
                edgecolors='black', node_size=30, font_size=10, pos=pos,
                width=0.3, arrowsize=5)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                   loc='upper left')
        plt.show()   


    def plot_graph_block_with_labels(self):
        color_map = []
        for i in range(self.n_nodes):
            if self.nodes[i].blocked == False:
                color_map.append("white")
            if self.nodes[i].blocked == True:
                color_map.append("red")  
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='not blocked', 
                   markerfacecolor='white', markeredgecolor='black', 
                   markersize=15),
            Line2D([0], [0], marker='o', color='w', label='blocked', 
                   markerfacecolor='red', markeredgecolor='black', 
                   markersize=15),    
        ]   
        labels = nx.get_node_attributes(self.G, 'id')
        pos = nx.kamada_kawai_layout(self.G)
        nx.draw(self.G, labels=dict([index for index in enumerate(labels)]),
                node_color=color_map, font_color='black', edgecolors='black',
                node_size=200, font_size=7, pos=pos,
                width=0.5, arrowsize=5)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                   loc='upper left')
        plt.show()


    def file_complaint(self, node_id):
        self.nodes[node_id].number_complaints += 1


    def block_nodes(self):
        for i in range(len(self.nodes)):
            self.nodes[i].update_block_status()
        count_blocked_nodes = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].blocked:
                count_blocked_nodes += 1
        self.n_blocked_nodes = count_blocked_nodes


    def compute_initial_data_analysis(self):
        neutral = 0
        infected = 0
        vaccinated = 0
        cured = 0
        perc_neutral_list = []
        perc_infected_list = []
        perc_vaccinated_list = []
        perc_cured_list = []
        for i in range(len(self.nodes)):
            if self.nodes[i].type_sirv == "infected":
                infected += 1
            elif self.nodes[i].type_sirv == "vaccinated":
                vaccinated += 1
            elif self.nodes[i].type_sirv == "cured":
                cured += 1
            else:
                neutral += 1
        perc_neutral = (neutral * 100) / len(self.nodes)
        perc_infected = (infected * 100) / len(self.nodes)
        perc_vaccinated = (vaccinated * 100) / len(self.nodes)
        perc_cured = (cured * 100) / len(self.nodes)
        perc_neutral_list.append(perc_neutral)
        perc_infected_list.append(perc_infected)
        perc_vaccinated_list.append(perc_vaccinated)
        perc_cured_list.append(perc_cured)
        self.perc_neutral = perc_neutral_list
        self.perc_infected = perc_infected_list
        self.perc_vaccinated = perc_vaccinated_list
        self.perc_cured = perc_cured_list


    def compute_data_analysis(self):
        neutral = 0
        infected = 0
        vaccinated = 0
        cured = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].type_sirv == "infected":
                infected += 1
            elif self.nodes[i].type_sirv == "vaccinated":
                vaccinated += 1
            elif self.nodes[i].type_sirv == "cured":
                cured += 1
            else:
                neutral += 1
        perc_neutral = (neutral * 100) / len(self.nodes)
        perc_infected = (infected * 100) / len(self.nodes)
        perc_vaccinated = (vaccinated * 100) / len(self.nodes)
        perc_cured = (cured * 100) / len(self.nodes)
        self.perc_neutral.append(perc_neutral)
        self.perc_infected.append(perc_infected)
        self.perc_vaccinated.append(perc_vaccinated)
        self.perc_cured.append(perc_cured)

        infected_from_echo = 0
        vaccinated_from_echo = 0
        cured_from_echo = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].infected_from_echo == True:
                infected_from_echo += 1
            elif self.nodes[i].vaccinated_from_echo == True:
                vaccinated_from_echo += 1
            elif self.nodes[i].cured_from_echo == True:
                cured_from_echo += 1
        self.infected_from_echo = infected_from_echo
        self.vaccinated_from_echo = vaccinated_from_echo
        self.cured_from_echo = cured_from_echo


    def update_nodes(self):
        self.global_timestep += 1
        # Non serve mescolare i nodi in quanto l'aggiornamento di ogni nodo
        # considera solo i messaggi degli istanti precedenti.
        for i in range(len(self.nodes)):
            self.nodes[i].update(self.global_timestep, self.file_complaint)
        # Aggiorno lo status blocked dei nodi solo se il relativo
        # iperparametro è impostato a True.
        if self.user_block == True:
            self.block_nodes()
        self.compute_data_analysis()


    def set_nodes_probabilities(self):
        for i in range(self.n_nodes):
            self.nodes[i].prob_complaint = self.prob_complaint
            self.nodes[i].prob_infection = self.prob_infection 
            self.nodes[i].prob_vaccination = self.prob_vaccination 
            self.nodes[i].prob_cure = self.prob_cure  


    def set_prob_echo(self, prob_echo):
        self.prob_echo = prob_echo
        for i in range(self.n_nodes):
            self.nodes[i].prob_echo = self.prob_echo 


    def set_prob_complaint(self, prob_complaint):
        self.prob_complaint = prob_complaint
        for i in range(self.n_nodes):
            self.nodes[i].prob_complaint = self.prob_complaint 


    def reset_network(self):
        for i in range(len(self.nodes)):
            self.nodes[i].reset_node()
        self.perc_neutral = []
        self.perc_infected = []
        self.perc_vaccinated = []
        self.perc_cured = []
        self.global_timestep = -1
        self.set_initial_infected_nodes(self.initial_infected_nodes)
        self.set_bots(self.bots)
        self.set_fact_checkers(self.fact_checkers)

        