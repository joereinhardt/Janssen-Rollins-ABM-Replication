# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:07:09 2016

@author: Joe
"""

import random
import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Model, Agent
from itertools import product
from mpl_toolkits.mplot3d import *

class Resource_Agent(Agent):

#Initial Conditions
    def __init__(self, unique_id, alpha, beta, eta):
        self.unique_id = unique_id
        self.score = 0 #accumulating score
        self.subscore = 0 #score of this round only
        self.endowment = 1
        self.contribution = 0#(random.randrange(0, 11, 1)) / 10
        self.takeout_share = (random.randrange(0, 11, 1)) / 10
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.past_contributions = [self.contribution]
        self.past_takeout_shares = [self.takeout_share]
        self.total_contribution = 0
        self.total_amount_taken_out = 0

#Resource Provision Phase      
    def first_round(self, model):
        self.endowment = 1
        #self.subscore = 0
        self.contribution = ((np.random.choice(11, p = self.investment_p(model))) / 10)
        self.score += (self.endowment - self.contribution)
        self.subscore += (self.endowment - self.contribution)
        self.past_contributions.append(self.contribution)
        self.total_contribution += self.contribution
        model.sum_of_contributions += self.contribution
        
#Resource Allocation Phase
    def second_round(self, model):
        takeout_share = ((np.random.choice(11, p = self.extraction_p(model))) / 10)
        takeout = round(model.resource * takeout_share, 1)
        actual_takeout_share = round((takeout / (model.resource + 0.000000001)), 1)
        self.takeout_share = actual_takeout_share
        if takeout_share <= 0:
            takeout_share = 0
        self.past_takeout_shares.append(actual_takeout_share)
        self.score += takeout
        self.subscore += takeout
        model.resource -= takeout
        if model.resource < 0:
            model.resource = 0
        self.total_amount_taken_out += takeout
      
#Variant of the Allocation phase for the linear public good game. The resource
#provision phase is the same in both types of games.      
    def sym_second_round(self, model):
        takeout = model.equal_share
        self.takeout_share = 0.2
        self.past_takeout_shares.append(0.2)
        self.score += takeout
        self.subscore += takeout
        model.resource -= takeout
        if model.resource < 0:
            model.resource = 0
        self.total_amount_taken_out += takeout

        
#General functions, needed for the probabilistic choice calculations   
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
        
    def last_round_sum_of_other_contributions(self, model):
        sum_of_contributions = 0
        for i in self.determine_upstream_players(model):
            sum_of_contributions += i.past_contributions[model.steps_this_round]
        for j in self.determine_downstream_players(model):
            sum_of_contributions += j.past_contributions[model.steps_this_round]
        return sum_of_contributions
    
    def last_round_upstream_agents_combined_takeout_shares(self, model):
        remaining_share = 1
        for i in self.determine_upstream_players(model):
            remaining_share = remaining_share * (1 - i.past_takeout_shares[model.steps_this_round])
        return remaining_share
    
    def this_round_upstream_players_combined_takeout_shares(self, model):
        remaining_share = 1
        for i in self.determine_upstream_players(model):
            remaining_share = remaining_share * (1 - i.takeout_share)
        return remaining_share

#Probabilistic Choice functions        
    def investment_p(self, model):
        xj = self.last_round_sum_of_other_contributions(model)
        yj = self.last_round_upstream_agents_combined_takeout_shares(model)
        u_list = []
        for x in range(0, 11, 1):
            wi = 1 - x/10 + yj * model.created_resource(x/10 + xj)
            wavg = 1 - 0.2 * x/10 - 0.2 * xj + 0.2 * model.created_resource(x/10 + xj)
            u_of_x = wi - self.alpha * max(wi - wavg, 0) + self.beta * max(wavg - wi, 0)
            u_list.append(u_of_x)
            p_list = []
            denominator = 0
            for u in u_list:
                denominator += math.exp(self.eta * u)
            for u in u_list:
                    numerator = math.exp(self.eta * u)
                    p_list.append(numerator / denominator)
        return p_list
        
    def extraction_p(self, model):
        x = self.contribution
        xij = 0
        for i in model.schedule.agents:
            xij += i.contribution
        yj = self.this_round_upstream_players_combined_takeout_shares(model)
        u_list = []
        for y in range(0, 11, 1):
            wi = 1 - x + y/10 * yj * model.created_resource(xij)
            wavg = 1 - 0.2 * xij + 0.2 * model.created_resource(xij)
            u_of_x = wi - self.alpha * max(wi - wavg, 0) + self.beta * max(wavg - wi, 0)
            u_list.append(u_of_x)
            p_list = []
            denominator = 0
            for u in u_list:
                denominator += math.exp(self.eta * u)
            if denominator == 0:
                for u in u_list:
                    p_list.append(1/11)
            else:
                for u in u_list:
                    numerator = math.exp(self.eta * u)
                    p_list.append(numerator / denominator)
        return p_list

class Resource_Model(Model):
    
#Initial Conditions
    def __init__(self, N, alpha, beta, eta, fa, r, group_id = 1):
        self.num_agents = N
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.schedule = BaseScheduler(self)
        self.create_agents()
        self.resource = 0
        self.sum_of_contributions = 0
        # Variables needed for mesa's BatchRunner to work later
        self.total_produced_resource = 0
        self.running = True
        self.fa = fa # frequency of asymmetric games being played. Set to 1 for
        #only asymmetric games, and to 0 for only linear games being played.
        self.r = r #parameter in the resource generation function for the
        #linear games.
        self.equal_share = 0 #Determining the share each player receives in the
        #linear symmetric game.
        self.group_id = group_id
        self.this_round_produced_resource = 0
        self.steps_this_round = 0
        
    def create_agents(self):            
        for i in range(self.num_agents):
            a = Resource_Agent(i, self.alpha, self.beta, self.eta)
            self.schedule.add(a)
            
    def step(self, asym):
        #asym = np.random.choice(2, p = [self.fa, (1 - self.fa)])#Determines 
        #whether an asymmetric or linear game is being played.
        if asym == 0:
            self.schedule.step() #step function defined in mesa's BaseScheduler Class.
            #An asymmetric commons game is being played for this round.
        else:
            self.schedule.symstep() #Manually added function in mesa's BaseScheduler Class.
            #A linear public good game is being played in this round.
        self.total_produced_resource += self.created_resource(self.sum_of_contributions)
        self.this_round_produced_resource += self.created_resource(self.sum_of_contributions)
        self.sum_of_contributions = 0
        self.steps_this_round += 1
        
    def run_model(self, steps):
        asym = np.random.choice(2, p = [self.fa, (1 - self.fa)])
        self.this_round_produced_resource = 0
        self.steps_this_round = 0
        for a in self.schedule.agents:
            a.contribution = 0
            a.takeout_share = random.randrange(0, 11, 1) / 10
            a.past_contributions = [a.contribution]
            a.past_takeout_shares = [a.takeout_share]
            a.subscore = 0
        for i in range(steps):
            self.step(asym)

    def determine_generated_cpr(self): #Needed to collect CPR levels for runs with multiple iterations
        generated_cpr = self.total_produced_resource
        return generated_cpr

         
    def created_resource(self, sum_of_contributions): #resource function
        if sum_of_contributions >= 0 and sum_of_contributions < 1:
            produced_resource = 0
        elif sum_of_contributions >=1 and sum_of_contributions < 1.5:
            produced_resource = 0.5
        elif sum_of_contributions >=1.5 and sum_of_contributions < 2:
            produced_resource = 2
        elif sum_of_contributions >= 2 and sum_of_contributions < 2.5:
            produced_resource = 4
        elif sum_of_contributions >= 2.5 and sum_of_contributions < 3:
            produced_resource = 6
        elif sum_of_contributions >=3 and sum_of_contributions < 3.5:
            produced_resource = 7.5
        elif sum_of_contributions >=3.5 and sum_of_contributions < 4:
            produced_resource = 8.5
        elif sum_of_contributions >=4 and sum_of_contributions < 4.5:
            produced_resource = 9.5
        elif sum_of_contributions >=4.5 and sum_of_contributions <= 5:
            produced_resource = 10
        return produced_resource
      
#Needed to determine distribution of extractions between agent positions      
    def determine_agent_total_takeout(self):
        takeout_list = []
        for i in self.schedule.agents:
            takeout = i.total_amount_taken_out
            takeout_list.append(takeout)
        return takeout_list
        
#Needed to determine distribution of investments between agent positions        
    def determine_agent_total_contribution(self):
        contribution_list = []
        for i in self.schedule.agents:
            contribution = i.total_contribution
            contribution_list.append(contribution)
        return contribution_list

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
    #Modified part. Adding a first (provision) and second(allocation) round
        for agent in self.agents:
            agent.first_round(self.model)
        self.model.resource = self.model.created_resource(self.model.sum_of_contributions)
        for agent in self.agents:            
            agent.second_round(self.model)
        self.steps += 1
        self.time += 1
     
    #Added function for the linear goods game 
    def symstep(self):
        for agent in self.agents:
            agent.first_round(self.model)
        self.model.resource = self.model.r * (self.model.sum_of_contributions)
        self.model.equal_share = (self.model.resource / 5)
        for agent in self.agents:            
            agent.sym_second_round(self.model)
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
            
    def run_all(self): # modified to give some progress info while the model is running
        params = self.parameter_values.keys()
        param_ranges = self.parameter_values.values()
        run_count = 0
        s = 1
        g = len(alpha_values) * len(beta_values)
        if figure3_6 == 1 or figure3_7 == 1:
            print("running...")
        elif figure3_8 ==1:
            pass
        else:
            print("running iteration " + str(s) + " of " + str(iterations) + "...")
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
                if figure3_6 != 1 and figure3_7 != 1 and figure3_8 != 1:
                    if run_count % g == 0:
                        s += 1
                        if s <= iterations:
                            print("running iteration " + str(s) + " of " + str(iterations) + str("..."))
                        else:
                            print("")
                            print("Done!")
                         
    def run_model(self, model):
        while model.running and model.schedule.steps < self.max_steps:
            asym = np.random.choice(2, p = [model.fa, (1 - model.fa)])
            model.step(asym)
            
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
