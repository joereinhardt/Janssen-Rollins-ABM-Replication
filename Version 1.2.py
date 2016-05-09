# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:01:49 2016

@author: Joe
"""

import random
import functools
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



from mesa import Model, Agent
from mesa.datacollection import DataCollector
from itertools import product
from operator import itemgetter, attrgetter, methodcaller

class Resource_Agent(Agent):
    
    def __init__(self, unique_id, alpha, beta, eta):
        
        self.unique_id = unique_id
        self.endowment = 10
        self.score = 0
        self.contribution = 0
        self.takeout = 0
        self.last_takeout_level = 0
        self.takeout_level = 0
        self.past_takeout_levels = [self.takeout_level]
        self.past_contributions = [self.contribution]
        self.sum_of_past_contributions = sum(self.past_contributions)
        self.past_takeouts = [self.takeout]
        self.last_contribution = 0
        self.last_takeout = 0
        self.alpha = alpha
        self.beta = alpha
        self.this_takeout = 0
        self.takeout_sum = 0
        self.contribution_sum = 0
        self.eta = eta
        self.k_probability_list = []
        for s in range(0,11):
            self.k_probability_list.append(0)
        self.l_probability_list = []
        for s in range(0,11):
            self.l_probability_list.append(0)

    def step_one(self, model):
         
        #print("")
        #print("AGENT " + str(self.unique_id))
        self.first_tick(model)
        self.determine_upstream_players(model)
        self.get_last_contribution(model)
        self.last_contribution = self.get_last_contribution(model)
        self.upstream_player_last_contribution(model)
        
    def step_two(self, model):
        
        
        #print("")
        #print("AGENT " + str(self.unique_id))
        self.second_tick(model)
        self.get_last_takeout(model)
        self.last_takeout = self.get_last_takeout(model)
        self.last_takeout_level = self.get_last_takeout_level(model)
        self.this_takeout = self.get_this_takeout(model)
        self.upstream_player_last_takeout(model)
        
    def first_tick(self, model):
        
        self.k_probability_list = self.k_probability_list_function(model)
        self.endowment = 10
        self.contribution = np.random.choice(11, p = self.k_probability_list_function(model))
        self.past_contributions.append(self.contribution)
        self.score += (self.endowment - self.contribution)
        model.contribution_sum += self.contribution
        model.kept_tokens += self.endowment - self.contribution
        self.contribution_sum += self.contribution
        #print("Player " + str(self.unique_id))
        #if self.unique_id == 0:# or self.unique_id == 1 or self.unique_id == 2 or self.unique_id == 3 or self. unique_id == 4:
            #print("Contribution Probabilities: " + str(self.k_probability_list_function(model)))
            #print("contribution: " + str(self.contribution))

    def second_tick(self, model):
        
        #print("multiplied resource: " + str(model.resource))
        #print("Player " + str(self.unique_id))
        #if self.unique_id == 0 or self.unique_id == 1 or self.unique_id == 2 or self.unique_id == 3 or self. unique_id == 4:
            #print("Takeout Probabilities: " + str(self.l_probability_list(model)))
        self.l_probability_list = self.l_probability_list_function(model)
        takeout_level = ((np.random.choice(11, p = self.l_probability_list) / 10))
        self.takeout_level = takeout_level
        self.takeout = round(takeout_level * model.resource)
        if model.resource == 0:
            self.takeout_level = 0
        self.past_takeouts.append(self.takeout)
        self.past_takeout_levels.append(self.takeout_level)
        self.takeout_sum += self.takeout
        self.score += self.takeout
        model.resource -= self.takeout
        model.generated_resource += self.takeout
        #print("taking out: " + str(self.takeout))
        #print("resource after second tick: " + str(model.resource))

        
    def upstream_player_last_contribution(self, model):
        
        if self.unique_id == 0:
            upstream_player_last_contribution = [0]
        else:
            upstream_player_last_contribution = []
            for i in self.determine_upstream_players(model):
                upstream_player_last_contribution.append(i.last_contribution)
        #if self.unique_id == 4:
            #print("upstream players last contribution: " + str(upstream_player_last_contribution)) 
        return upstream_player_last_contribution
        
    def downstream_player_last_contribution(self, model):
        
        if self.unique_id == 4:
            downstream_player_last_contribution = [0]
        else:
            downstream_player_last_contribution = []
            for i in self.determine_downstream_players(model):
                downstream_player_last_contribution.append(i.last_contribution)
        return downstream_player_last_contribution
        
    def upstream_player_last_takeout(self, model):
        
        if self.unique_id == 0:
            upstream_player_last_takeout = [0]
        else:
            upstream_player_last_takeout = []
            for i in self.determine_upstream_players(model):
                upstream_player_last_takeout.append(i.last_takeout)
        #if self.unique_id == 4:
            #print("upstream players last takeout: " + str(upstream_player_last_takeout))
        return upstream_player_last_takeout
        
    def upstream_players_last_takeout_levels(self, model):
        
        if self.unique_id == 0:
            upstream_players_last_takeout_levels = [0]
        else:
            upstream_players_last_takeout_levels = []
            for i in self.determine_upstream_players(model):
                upstream_players_last_takeout_levels.append(i.last_takeout_level)
        return upstream_players_last_takeout_levels
    
    def upstream_players_last_takeout_levels_negative(self, model):
        
        if self.unique_id == 0:
            upstream_players_last_takeout_levels_negative = [1]
        else:
            upstream_players_last_takeout_levels_negative = []
            for i in self.determine_upstream_players(model):
                upstream_players_last_takeout_levels_negative.append(1 - (i.last_takeout_level))
        #print("UPSTREAM PLAYERS LAST TAKEOUT LEVELS NEG:   "+ str(upstream_players_last_takeout_levels_negative))
        return upstream_players_last_takeout_levels_negative

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

    def get_last_contribution(self, model):
        
        last_contribution = self.past_contributions[model.schedule.steps]
        return last_contribution
        
    def get_last_takeout(self, model):
        
        last_takeout = self.past_takeouts[model.schedule.steps]
        return last_takeout
    
    def get_last_takeout_level(self, model):
        
        last_takeout_level = self.past_takeout_levels[model.schedule.steps]
        return last_takeout_level

    def upstream_players_contribution_sum(self, model):
        
        contribution_sum = sum(self.upstream_player_last_contribution(model))
        return contribution_sum
                
    def downstream_players_contribution_sum(self, model):
        
        downstream_contribution_sum = sum(self.downstream_player_last_contribution(model))
        return downstream_contribution_sum
        
    def upstream_players_takeout_sum(self, model):
        
        takeout_sum = sum(self.upstream_player_last_takeout(model))
        return takeout_sum
    
    def upstream_players_takeout_level_combined(self, model):
        
        takeout_level_combined = functools.reduce(lambda x, y: x*y, self.upstream_players_last_takeout_levels_negative(model))
        #print("TAKEOUT LEVEL COMBINED " + str(takeout_level_combined))
        return takeout_level_combined
        
    
    def determine_available_resource(self, k, model):
        
        contributions = self.upstream_players_contribution_sum(model) + self.downstream_players_contribution_sum(model) + k
        resource = self.resource_transformation(contributions)
        available_resource = resource - self.upstream_players_takeout_sum(model)
        return available_resource
        
    def resource_transformation(self, contributions):
        
        if contributions >= 0 and contributions <= 10:
            produced_resource = 0
        elif contributions >10 and contributions <= 15:
            produced_resource = 5
        elif contributions >15 and contributions <= 20:
            produced_resource = 20
        elif contributions >20 and contributions <= 25:
            produced_resource = 40
        elif contributions >25 and contributions <= 30:
            produced_resource = 60
        elif contributions >30 and contributions <= 35:
            produced_resource = 75
        elif contributions >35 and contributions <= 40:
            produced_resource = 85
        elif contributions >40 and contributions <= 45:
            produced_resource = 95
        elif contributions >45 and contributions <= 50:
            produced_resource = 100
        return produced_resource

        
    def determine_wage(self, k, model):
        
        potential_takeout = self.determine_available_resource(k, model) * self.upstream_players_takeout_level_combined(model)
        wage_player = (self.endowment - k) + potential_takeout
        return wage_player
            
    def determine_other_wages(self, model):
        
        other_takeouts = []
        for i in model.schedule.agents:
            if i.unique_id != self.get_id():
                other_takeouts.append(i.last_takeout)
        other_contributions_net = []
        for j in model.schedule.agents:
            if j.unique_id != self.get_id():
                other_contributions_net.append(self.endowment - (j.last_contribution))
        other_wages = other_takeouts + other_contributions_net
        return other_wages
        
    def determine_average_wage(self, k, model):
        
        n = 0
        for i in model.schedule.agents:
            n += 1
        average_wage = (self.determine_wage(k, model) + sum(self.determine_other_wages(model))) / n
        return average_wage
    
    def determine_player_utility(self, k, model):
        
        if self.determine_wage(k, model) > self.determine_average_wage(k, model):
            utility = self.determine_wage(k, model) - (self.alpha * (self.determine_wage(k, model) - self.determine_average_wage(k, model)))
        elif self.determine_wage(k, model) == self.determine_average_wage(k, model):
            utility = self.determine_wage(k, model)
        else:
            utility = self.determine_wage(k, model) + (self.beta * (self.determine_average_wage(k, model) - self.determine_wage(k, model)))
        return utility

    def utility_transformation(self, k, model):
        
        transformed_utility = math.exp(self.eta * (self.determine_player_utility(k, model)))
        return transformed_utility

    def transformed_utility_sum(self, model):
    
        utility_sum = 0
        for i in range(0, 11, 1):
            calculation = self.utility_transformation(i, model)
            utility_sum += calculation
        return utility_sum
    
    def probability_of_k(self, k, model):
    
        utility_of_k = self.utility_transformation(k, model)
        probability = utility_of_k / self.transformed_utility_sum(model)
        return probability
    
    def k_probability_list_function(self, model):
    
        probability_list = []
        for i in range(0, 11, 1):
            probability_list.append(self.probability_of_k(i, model))
        return probability_list
    
    """
    Functions for share allocation in phase 2
    """
    
    def get_this_takeout(self, model):
        
        this_takeout = self.past_takeouts[model.schedule.steps + 1]
        return this_takeout

    def upstream_player_this_takeout(self, model):
        
        if self.unique_id == 0:
            upstream_player_this_takeout = [0]
        else:
            upstream_player_this_takeout = []
            for i in self.determine_upstream_players(model):
                upstream_player_this_takeout.append(i.this_takeout)
        return upstream_player_this_takeout
    
    def upstream_players_this_takeout_sum(self, model):
        
        this_takeout_sum = sum(self.upstream_player_this_takeout(model))
        return this_takeout_sum
        
    def phase_two_average_wage(self, model):
        
        n = 0
        for i in model.schedule.agents:
            n += 1
        two_resource = model.resource
        phase_two_average_wage = (two_resource + (n * self.endowment)) / n
        return phase_two_average_wage
    
        
    def phase_two_determine_wage(self, l, model):
        
        phase_two_wage = (self.endowment - self.contribution) + (l * model.resource)
        return phase_two_wage
        
    def determine_phase_two_utility(self, l, model):
        
        if self.phase_two_determine_wage(l, model) > self.phase_two_average_wage(model):
            utility = self.phase_two_determine_wage(l, model) - (self.alpha * (self.phase_two_determine_wage(l, model) - self.phase_two_average_wage(model)))
        elif self.phase_two_determine_wage(l, model) == self.phase_two_average_wage(model):
            utility = self.phase_two_determine_wage(l, model)
        else:
            utility = self.phase_two_determine_wage(l, model) + (self.beta * (self.phase_two_average_wage(model) - self.phase_two_determine_wage(l, model)))
        return utility

    def phase_two_utility_transformation(self, l, model):
        
        phase_two_transformed_utility = math.exp(self.eta * (self.determine_phase_two_utility(l, model)))
        return phase_two_transformed_utility

    def phase_two_transformed_utility_sum(self, model):
    
        phase_two_utility_sum = 0
        for i in range(0, 11, 1):
            calculation = self.phase_two_utility_transformation(i / 10, model)
            phase_two_utility_sum += calculation
        return phase_two_utility_sum
        
    def probability_of_l(self, l, model):
    
        utility_of_l = self.phase_two_utility_transformation(l, model)
        probability_of_l = utility_of_l / self.phase_two_transformed_utility_sum(model)
        return probability_of_l
        
    def l_probability_list_function(self, model):
    
        l_probability_list = []
        for i in range(0, 11, 1):
            l_probability_list.append(self.probability_of_l(i / 10, model))
        return l_probability_list

        
class Resource_Model(Model):
    
    
    def __init__(self, N, alpha, beta, eta):
        
        self.running = True
        self.num_agents = N
        self.schedule = BaseScheduler(self)
        self.create_agents()
        self.resource = 0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.contribution_sum = 0
        #self.resource_list = []
        ar = {"Contribution": lambda ab: ab.contribution, 
              "Takeout": lambda ac: ac.takeout,
              "TakeoutLevel": lambda ah: ah.takeout_level,
              "SumOfTakeouts": lambda af: af.takeout_sum, 
              "SumOfContributions": lambda aa: aa.contribution_sum,
              "ProbabilityOfZeroContribution": lambda ay: ay.k_probability_list[0]}
        self.dc = DataCollector(agent_reporters = ar)
        self.generated_resource = 0
        ad = {"Generated_Resource": lambda ad: ad.generated_resource}
        self.dci = DataCollector(model_reporters = ad)
        self.past_cprs = []
        self.kept_tokens = 0

        
    def create_agents(self):
        
        for i in range(self.num_agents):
            a = Resource_Agent(i, alpha, beta, eta)
            self.schedule.add(a)
          
    def produced_resource(self):
        
        if self.contribution_sum >= 0 and self.contribution_sum <= 10:
            produced_resource = 0
        elif self.contribution_sum >10 and self.contribution_sum <= 15:
            produced_resource = 5
        elif self.contribution_sum >15 and self.contribution_sum <= 20:
            produced_resource = 20
        elif self.contribution_sum >20 and self.contribution_sum <= 25:
            produced_resource = 40
        elif self.contribution_sum >25 and self.contribution_sum <= 30:
            produced_resource = 60
        elif self.contribution_sum >30 and self.contribution_sum <= 35:
            produced_resource = 75
        elif self.contribution_sum >35 and self.contribution_sum <= 40:
            produced_resource = 85
        elif self.contribution_sum >40 and self.contribution_sum <= 45:
            produced_resource = 95
        elif self.contribution_sum >45 and self.contribution_sum <= 50:
            produced_resource = 100
        return produced_resource
          
          
    def step(self):

        self.past_cprs.append(self.resource)
        self.dc.collect(self)
        self.schedule.step()
        for agent in self.schedule.agents:
            self.generated_resource += agent.contribution
        self.dci.collect(self)
        self.contribution_sum = 0
        #print("GENRES")
        #print(str(self.generated_resource))
        
    def determine_resource(self):
        
        return self.generated_resource
        
    def determine_total_payoff(self):
        
        total_payoff = self.kept_tokens + self.generated_resource
        return total_payoff
    
    def run_model(self, steps):
        
        for i in range(steps):
            self.step()
            if i % 100 == 0:
                print("")
                print("########################################")
                print("")
                print("STEP " + str(i) + " of " + str(steps))

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

        for agent in self.agents:
            agent.step_one(self.model)
        self.model.resource = self.model.produced_resource()
        for agent in self.agents:            
            agent.step_two(self.model)
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

"""
Model Parameters
"""

runs = 10
agents = 5
alpha = 0
beta = 0
eta = 0.5

"""
Run Command
"""
#test_model = Resource_Model(agents, alpha, beta, eta)
#test_model.run_model(runs)
"""
Contribution / Takeout Table
"""

"""data = test_model.dc.get_agent_vars_dataframe()
data.reset_index(inplace=True)

model_data = test_model.dci.get_model_vars_dataframe()

last_step_data = data[data.Step == (runs -1 )]

fig = plt.figure()
ax = fig.add_subplot(111)

N = agents
ind = np.arange(N)
width = 0.28
player_one = last_step_data[last_step_data.AgentID == 0]
highest_takeout = int(player_one.SumOfTakeouts)


#rects1 = ax.bar(ind, last_step_data.Score, width, color = "black")
rects3 = ax.bar(ind + width * 2, last_step_data.SumOfTakeouts, width, color = "red")
rects2 = ax.bar(ind + width, last_step_data.SumOfContributions, width, color = "blue")

ax.set_xlim(-width, len(ind) + width)
ax.set_ylim(0, highest_takeout * 1.1)
ax.set_ylabel("Points")
ax.set_title("Points by Agents")
xTickMarks = ["Agent" + str(i) for i in range(agents)]
ax.set_xticks(ind + width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation = 45, fontsize = 10)

ax.legend((rects2[0], rects3[0]), ("SumOfContributions", "SumOfTakeouts"))"""


"""
#Graph showing the development of contributions over time for a specific agent
"""
"""for x in range(0,5):
    agent_x_data = data[data.AgentID == x]
    agent_x_contributions = agent_x_data.Contribution
    agent_x_takeouts = agent_x_data.Takeout

    plt.plot(agent_x_contributions)
    #plt.plot(agent_x_takeouts)
    plt.ylabel("Contributions")
    plt.xlabel("Round")
    plt.title("Agent " + str(x) + " Contributions")
    plt.show()
       
for y in range(0,5):
    agent_y_data = data[data.AgentID == y]
    agent_y_contributions = agent_y_data.Contribution
    agent_y_takeout_levels = agent_y_data.TakeoutLevel

    plt.plot(agent_y_takeout_levels)
    #plt.plot(agent_x_takeouts)
    plt.ylabel("Takeout Levels")
    plt.xlabel("Round")
    plt.title("Agent " + str(y) + " Takeout Level")
    plt.show()

for g in range(0,5):
    
    agent_g_data = data[data.AgentID == g]
    agent_g_probabilityofzero = agent_g_data.ProbabilityOfZeroContribution
    plt.plot(agent_g_probabilityofzero)
    plt.ylabel("Probability of Zero Contribution")
    plt.xlabel("Round")
    plt.title("Agent " + str(g) + " Probability of Zero Contribution")
    plt.show()"""

"""
Batch Runner
"""


alpha_values = []
for i in range(7, 11):
    x = i/10
    alpha_values.append(x)

beta_values = []
for i in range(7, 11):
    x = i/10
    beta_values.append(x)




"""resource_data = test_model.dci.get_model_vars_dataframe()
resource_data.reset_index(inplace = True)
final_resource_data = resource_data[resource_data.index == (runs -1 )]
print("FINAL RESOURCE DATA")
print(final_resource_data)"""
#final_resource = int(final_resource_data.iloc[0]["Generated_Resource"])
#print("FINAL RESOURCE")
#print(final_resource)
def determine_final_resource(model):

    final_resource = model.determine_resource()
    #print("model.generated_resource: " + str(final_resource))
    return final_resource
    
    
def determine_total_payoff(model):
    
    total_payoff = model.determine_total_payoff()
    return total_payoff

iterations = 1

model = Resource_Model(agents, alpha, beta, eta)   
param_values = {"N": 5, "alpha": alpha_values, "beta": beta_values, "eta": 0.5}
model_reporter = {"Generated_Resource": determine_final_resource, "Total_Payoff": determine_total_payoff}
batch = BatchRunner(Resource_Model, param_values, iterations, runs, model_reporter)
batch.run_all()
out = batch.get_model_vars_dataframe()
print("OUT")
print(out)
"""plt.scatter(out.alpha, out.Generated_Resource)
plt.grid(True)
plt.xlabel("alpha")
plt.ylabel("Generated Resource")"""

"""plt.scatter(out.alpha, out.Generated_Resource)
plt.grid(True)
plt.xlabel("beta")
plt.ylabel("Generated Resource")
plt.show()"""

"""plt.scatter(out.beta, out.Generated_Resource)
plt.grid(True)
plt.xlabel("beta")
plt.ylabel("Total Payoff")
plt.show()"""

"""plt.scatter(out.alpha, out.Generated_Resource)
plt.grid(True)
plt.xlabel("alpha")
plt.ylabel("Generated Resource")
plt.show()"""

"""print("Generated Resource")
print(out.Generated_Resource)
print("alpha")
print(out.alpha)
print("beta")
print(out.beta)"""

newout = np.array(out)
#print("OUTARRAY")
#print(newout)
newout = sorted(newout, key=lambda a_entry: a_entry[0])
#sorted(newout, key = lambda z: z[0])
print("SORTEDOUT")
print(newout)

values = 16 * iterations

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
X = []
for j in range(0, values):
    X.append(newout[j][4])
#print("X")
#print(x)
Y = []
for j in range(0,values):
    y.append(newout[j][5])
#print("y")
#print(y)
Z = []
for j in range(0,values):
    Z.append(newout[j][0])
#print("Z")
#print(Z)

#X, Y = np.meshgrid(x, y)


def plot_3d(X, Y, Z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    plt.gca().invert_xaxis()
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    plt.show()

plot_3d(X, Y, Z)
