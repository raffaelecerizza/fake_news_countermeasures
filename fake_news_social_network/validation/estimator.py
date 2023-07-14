import os
import sys
import math
import random
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

sys.path.append('.')
sys.path.append('../../')
sys.path.append('../network_model/')

notebook_path = os.getcwd()
project_path = os.path.abspath(os.path.join(notebook_path, '..'))
sys.path.append(project_path)

from network_model.network_classes import Network
from networks_container import NetworksContainer


class Estimator(BaseEstimator):
    def __init__(self, prob_influencer=0, prob_infection=0, 
                 prob_vaccination=0, prob_cure=0, prob_echo=0):
        self.prob_influencer = prob_influencer
        self.prob_infection = prob_infection
        self.prob_vaccination = prob_vaccination
        self.prob_cure = prob_cure
        self.prob_echo = prob_echo


    def fit(self, X, y):
        container = X[0]
        container.set_probabilities(self.prob_influencer,
                                    self.prob_infection,
                                    self.prob_vaccination,
                                    self.prob_cure)
        container.set_prob_echo(self.prob_echo)
        # Return the classifier
        return self


    def predict(self, X):
        total_y_pred_list = []
        container = X[0]

        for i in range(container.n_networks):   
            y_pred_list = []
            container.network_list[i].reset_network()
            for j in range(container.epochs):
                container.network_list[i].update_nodes()
            y_pred_list = container.network_list[i].perc_infected
            total_y_pred_list.append(y_pred_list)

        return np.array(total_y_pred_list)
    

    def score(self, X, y):
        simulator = X[0]
        y_true = [item for sublist in y for item in sublist]
        y_pred = self.predict(X)

        neg_rmse_list = []
        for i in range(simulator.n_networks):
                mse = mean_squared_error(y_true=y_true, y_pred=y_pred[i])
                neg_rmse = - np.sqrt(mse)
                neg_rmse_list.append(neg_rmse)
            
        mean_neg_rmse = np.mean(neg_rmse_list)
   
        return mean_neg_rmse


