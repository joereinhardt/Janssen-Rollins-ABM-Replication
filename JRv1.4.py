# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 07:29:33 2016

@author: Joe
"""


import random
import math

import numpy as np
import pandas as pd

from mesa import Model, Agent
from mesa.datacollection import DataCollector
from itertools import product


"""
Begin of Actual Model
"""


class Resource_Agent(Agent):

    def __init__(self, unique_id, alpha, beta, eta):
        self.unique_id = unique_id
        self.score = 0
        self.endowment = 10
        self.contribution = 0 #random.randrange(0, 10, 1)
        self.takeout_share = 0 #(random.randrange(0, 10, 1)) / 10
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.past_contributions = [self.contribution]
        self.past_takeout_shares = [self.takeout_share]
        self.total_contribution = 0
        self.total_amount_taken_out = 0
        
    def first_round(self, model):
        self.endowment = 10
        self.contribution = np.random.choice(11, p = self.probability_list_function(model, "one", self.determine_wage, self.determine_average_wage))
        self.score += (self.endowment - self.contribution)
        self.past_contributions.append(self.contribution)
        self.total_contribution += self.contribution
        model.sum_of_contributions += self.contribution
        
    def second_round(self, model):
        self.takeout_share = ((np.random.choice(11, p = self.probability_list_function(model, "two", self.round_two_determine_wage, self.round_two_average_wage)) / 10))
        takeout = round(model.resource * self.takeout_share)
        self.score += takeout
        model.resource -= takeout
        self.past_takeout_shares.append(self.takeout_share)
        self.total_amount_taken_out += takeout  
        
    #General functions
    
    def get_id(self):
        return self.unique_id        

    def determine_upstream_players(self, model):
        
        upstream_players = []
        for agent in model.schedule.agents:
            if agent.unique_id < self.get_id():
                upstream_players.append(agent)
        return upstream_players
    
    def determine_downstream_players(self, model):
        
        downstream_players = []
        for agent in model.schedule.agents:
            if agent.unique_id > self.get_id():
                downstream_players.append(agent)
        return downstream_players
    
    #Functions for Round 1
    
    def last_round_sum_of_other_contributions(self, model):
        sum_of_contributions = 0
        for i in self.determine_upstream_players(model):
            sum_of_contributions += i.past_contributions[model.schedule.steps]
        for j in self.determine_downstream_players(model):
            sum_of_contributions += j.past_contributions[model.schedule.steps]
        return sum_of_contributions
    
    def last_round_upstream_agents_combined_takeout_shares(self, model):
        remaining_share = 1
        for i in self.determine_upstream_players(model):
            remaining_share = remaining_share * (1 - i.past_takeout_shares[model.schedule.steps])
        return remaining_share
    
    def determine_available_resource(self, k, model):
        contributions = self.last_round_sum_of_other_contributions(model) + k
        resource = model.created_resource(contributions)
        available_resource = resource * self.last_round_upstream_agents_combined_takeout_shares(model)
        return available_resource
    
    def determine_wage(self, k, model):
        wage_player = (self.endowment - k) + self.determine_available_resource(k, model)
        return wage_player
    
    def determine_average_wage(self, k, model):
        n = 0
        for i in model.schedule.agents:
            n += 1
        generated_cpr = model.created_resource(self.last_round_sum_of_other_contributions(model) + k)
        endowment_not_invested = (n * self.endowment) - (self.last_round_sum_of_other_contributions(model) + k)
        total_wage = generated_cpr + endowment_not_invested
        average_wage = total_wage / n
        return average_wage
    
    def utility_of_x(self, x, model, wage_function, average_wage_function):
        if wage_function(x, model) > average_wage_function(x, model):
            utility = wage_function(x, model) - (self.alpha * (wage_function(x, model) - average_wage_function(x, model)))
        elif wage_function(x, model) == average_wage_function(x, model):
            utility = wage_function(x, model)
        else:
            utility = wage_function(x, model) + (self.beta * (average_wage_function(x, model) - wage_function(x, model)))
        return utility
    
    def utility_transformation(self, x, model, wage_function, average_wage_function):
        transformed_utility = math.exp(self.eta * (self.utility_of_x(x, model, wage_function, average_wage_function)))
        return transformed_utility

    def transformed_utility_sum(self, model, round_number, wage_function, average_wage_function):
        utility_sum = 0
        if round_number == "one":
            for i in range(0, 11, 1):
                calculation = self.utility_transformation(i, model, wage_function, average_wage_function)
                utility_sum += calculation
        elif round_number == "two":
            for i in range(0, 11, 1):
                calculation = self.utility_transformation(i / 10, model, self.round_two_determine_wage, self.round_two_average_wage)
                utility_sum += calculation
        return utility_sum   
    
    def probability_of_x(self, x, model, round_number, wage_function, average_wage_function):
        utility_of_x = self.utility_transformation(x, model, wage_function, average_wage_function)
        probability = utility_of_x / self.transformed_utility_sum(model, round_number, wage_function, average_wage_function)
        return probability
    
    def probability_list_function(self, model, round_number, wage_function, average_wage_function):
        probability_list = []
        if round_number == "one":
            for i in range(0, 11, 1):
                probability_list.append(self.probability_of_x(i, model, round_number, wage_function, average_wage_function))
        elif round_number == "two":
            for i in range(0, 11, 1):
                probability_list.append(self.probability_of_x(i / 10, model, round_number, wage_function, average_wage_function))
        return probability_list
    
    #Functions for Round 2
    
    def round_two_determine_wage(self, l, model):
        round_two_wage = (self.endowment - self.contribution) + (l * model.resource)
        return round_two_wage
     
    def round_two_average_wage(self, l, model):
        n = 0
        for i in model.schedule.agents:
            n += 1
        this_round_contribution_sum = 0
        for i in model.schedule.agents:
            this_round_contribution_sum += i.contribution
        generated_resource = model.created_resource(this_round_contribution_sum)
        endowment_not_invested = (n * self.endowment) - this_round_contribution_sum
        total_wage = generated_resource + endowment_not_invested
        average_wage = total_wage / n
        return average_wage
        
class Resource_Model(Model):
    
    def __init__(self, N, alpha, beta, eta):
        self.num_agents = N
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.schedule = BaseScheduler(self)
        self.create_agents()
        self.resource = 0
        self.sum_of_contributions = 0
        ar = {"Score": lambda ar: ar.score, 
                "Total_Contribution": lambda at: at.total_contribution, 
                "Total_Amount_Taken_Out": lambda au: au.total_amount_taken_out}
        self.dc = DataCollector(agent_reporters = ar)
        # Variables needed for the BatchRunner to work later
        self.total_produced_resource = 0
        ad = {"Generated_CPR": lambda ad: ad.total_produced_resource}
        self.dci = DataCollector(model_reporters = ad)
        self.running = True
        
    def create_agents(self):            
        for i in range(self.num_agents):
            a = Resource_Agent(i, self.alpha, self.beta, self.eta)
            self.schedule.add(a)
            
    def step(self):
        self.schedule.step()
        self.dc.collect(self)
        self.total_produced_resource += self.created_resource(self.sum_of_contributions)
        self.dci.collect(self)
        self.sum_of_contributions = 0
        
    def run_model(self, steps):
        for i in range(steps):
            self.step()

    def determine_generated_cpr(self):
        generated_cpr = self.total_produced_resource
        return generated_cpr
            
    def created_resource(self, sum_of_contributions):
        if sum_of_contributions >= 0 and sum_of_contributions <= 10:
            produced_resource = 0
        elif sum_of_contributions >10 and sum_of_contributions <= 15:
            produced_resource = 5
        elif sum_of_contributions >15 and sum_of_contributions <= 20:
            produced_resource = 20
        elif sum_of_contributions >20 and sum_of_contributions <= 25:
            produced_resource = 40
        elif sum_of_contributions >25 and sum_of_contributions <= 30:
            produced_resource = 60
        elif sum_of_contributions >30 and sum_of_contributions <= 35:
            produced_resource = 75
        elif sum_of_contributions >35 and sum_of_contributions <= 40:
            produced_resource = 85
        elif sum_of_contributions >40 and sum_of_contributions <= 45:
            produced_resource = 95
        elif sum_of_contributions >45 and sum_of_contributions <= 50:
            produced_resource = 100
        return produced_resource
        

"""
Modified mesa modules - replace these with imported modules
"""

class BaseScheduler(object):
    model = None
    steps = 0
    time = 0
    agents = []

    def __init__(self, model):
        self.model = model
        self.steps = 0
        self.time = 0
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def remove(self, agent):
        while agent in self.agents:
            self.agents.remove(agent)

    def step(self):
    #Modified part.
        for agent in self.agents:
            agent.first_round(self.model)
        self.model.resource = self.model.created_resource(self.model.sum_of_contributions)
        for agent in self.agents:            
            agent.second_round(self.model)
        self.steps += 1
        self.time += 1

    def get_agent_count(self):
        return len(self.agents)

class BatchRunner(object):


    model_cls = None
    parameter_values = {}
    iterations = 1

    model_reporters = {}
    agent_reporters = {}

    model_vars = {}
    agent_vars = {}

    def __init__(self, model_cls, parameter_values, iterations=1,
                 max_steps=1000, model_reporters=None, agent_reporters=None):
        self.model_cls = model_cls
        self.parameter_values = {param: self.make_iterable(vals)
                                 for param, vals in parameter_values.items()}
        self.iterations = iterations
        self.max_steps = max_steps

        self.model_reporters = model_reporters
        self.agent_reporters = agent_reporters

        if self.model_reporters:
            self.model_vars = {}

        if self.agent_reporters:
            self.agent_vars = {}
            
    def run_all(self):
        params = self.parameter_values.keys()
        param_ranges = self.parameter_values.values()
        run_count = 0
        n = 1
        g = len(alpha_values) * len(beta_values)
        print("running iteration " + str(n) + " of " + str(iterations) + "...")
        for param_values in list(product(*param_ranges)):
            kwargs = dict(zip(params, param_values))
            for _ in range(self.iterations):
                model = self.model_cls(**kwargs)
                self.run_model(model)
                # Collect and store results:
                if self.model_reporters:
                    key = tuple(list(param_values) + [run_count])
                    self.model_vars[key] = self.collect_model_vars(model)
                if self.agent_reporters:
                    for agent_id, reports in self.collect_agent_vars.items():
                        key = tuple(list(param_values) + [run_count, agent_id])
                        self.agent_vars[key] = reports
                run_count += 1
                if run_count % g == 0:
                    n += 1
                    if n <= iterations:
                        print("running iteration " + str(n) + " of " + str(iterations) + str("..."))
                    else:
                        print("Done!")
                         
    def run_model(self, model):
        while model.running and model.schedule.steps < self.max_steps:
            model.step()
            
    def collect_model_vars(self, model):
        model_vars = {}
        for var, reporter in self.model_reporters.items():
            model_vars[var] = reporter(model)
        return model_vars
        
    def collect_agent_vars(self, model):
        agent_vars = {}
        for agent in model.schedule.agents:
            agent_record = {}
            for var, reporter in self.agent_reporters.items():
                agent_record[var] = reporter(agent)
            agent_vars[agent.unique_id] = agent_record
        return agent_vars
        
    def get_model_vars_dataframe(self):
        index_col_names = list(self.parameter_values.keys())
        index_col_names.append("Run")
        records = []
        for key, val in self.model_vars.items():
            record = dict(zip(index_col_names, key))
            for k, v in val.items():
                record[k] = v
            records.append(record)
        return pd.DataFrame(records)
        
    def get_agent_vars_dataframe(self):
        index_col_names = list(self.parameter_values.keys())
        index_col_names += ["Run", "AgentID"]
        records = []
        for key, val in self.agent_vars.items():
            record = dict(zip(index_col_names, key))
            for k, v in val.items():
                record[k] = v
            records.append(record)
        return pd.DataFrame(records)
        
    @staticmethod
    def make_iterable(val):
        if hasattr(val, "__iter__") and type(val) is not str:
            return val
        else:
            return [val]
                

#Model Parameters
runs = 10
agents = 5
eta = 0.5
iterations = 50

alpha_values = []
for i in range(-10, 11):
    x = i/10
    alpha_values.append(x)

beta_values = []
for i in range(-10, 11):
    x = i/10
    beta_values.append(x)

    
def determine_final_resource(model):
    final_resource = model.determine_generated_cpr()
    return final_resource

def compute_gini_contributions(model):
    agent_contribution_sum = [agent.total_contribution for agent in model.schedule.agents]
    x = sorted(agent_contribution_sum)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x) + 0.0000001)
    return (1 + (1/N) - 2*B)

def compute_gini_takeouts(model):
    agent_takeout_sum = [agent.total_amount_taken_out for agent in model.schedule.agents]
    x = sorted(agent_takeout_sum)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x) + 0.0000001)
    return (1 + (1/N) - 2*B)


param_values = {"N": 5, "alpha": alpha_values, "beta": beta_values, "eta": 0.5}
model_reporter = {"Generated_CPR": determine_final_resource,
                  "Gini_Contributions": compute_gini_contributions, 
                  "Gini_Takeouts": compute_gini_takeouts}
batch = BatchRunner(Resource_Model, param_values, iterations, runs, model_reporter)
batch.run_all()
out = batch.get_model_vars_dataframe()

print(out)
out.to_csv("v1.4_" + str(iterations) + "iterations_" + str(eta) + "eta_plus_ginis_CalcStarting_value.csv")