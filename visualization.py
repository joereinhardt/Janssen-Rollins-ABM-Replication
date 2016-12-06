# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 09:26:37 2016

@author: Joe
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Visu(object):
    
    def __init__(self, out, cultural_selection = 0):
        if cultural_selection == 0:
            out_one = out[out.alpha >= out.beta]
            out_two = out[out.beta > out.alpha]
            for j in out_two.index:
                out_two.set_value(j, "Generated_CPR", 0)
                out_two.set_value(j, "Gini_Contributions", 0)
                out_two.set_value(j, "Gini_Takeouts", 0)
            out = out_one.merge(out_two, how="outer")
            grouped_all = out.groupby(["alpha", "beta"]).mean()
            resource_values = grouped_all.Generated_CPR
            resource_list = np.array(resource_values)
            resource_list = ((resource_list) / 10)
            z_values = [resource_list[x:x+21] for x in range(0, len(resource_list), 21)]
            variables = []
            for j in range(-10, 11):
                variables.append(j/10)
                x = []
            for j in range(21):
                x.append(variables)      
            y = []
            for j in variables:
                sublist = []
                for s in range(21):
                    sublist.append(j)
                y.append(sublist)
            self.X = x
            self.Y = y
            self.Z = z_values
            contri_gini_values = grouped_all.Gini_Contributions
            contri_gini_list = np.array(contri_gini_values)
            a_values = [contri_gini_list[x:x+21] for x in range(0, len(contri_gini_list), 21)]
            self.A = a_values
            takeout_gini_values = grouped_all.Gini_Takeouts
            takeout_gini_list = np.array(takeout_gini_values)
            b_values = [takeout_gini_list[x:x+21] for x in range(0, len(takeout_gini_list), 21)]
            self.B = b_values
        else:
            self.out = pd.DataFrame(out)

        
    def plot_3d(self, X, Y, Z, zaxis_label, azim, elev, invert_xaxis=True, invert_yaxis=True, invert_labels=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        if invert_xaxis==True:
            plt.gca().invert_xaxis()
        if invert_yaxis==True:
            plt.gca().invert_yaxis()
        ax.plot_wireframe(X, Y, Z)
        if invert_labels==True:
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Beta')
        else:
            ax.set_xlabel('Beta')
            ax.set_ylabel('Alpha')
        ax.set_zlabel(str(zaxis_label))
        ax.set_title(str(zaxis_label))
        ax.view_init(azim, elev)
        plt.show()        
        
    def plot(self):
        self.plot_3d(self.X, self.Y, self.Z, "CPR level", 30, 60)
        self.plot_3d(self.Y, self.X, self.A, "Gini Contributions", 30, 60, invert_xaxis=False, invert_yaxis=False, invert_labels=True)
        self.plot_3d(self.X, self.Y, self.B, "Gini Takeouts", 30, 30, invert_xaxis=True, invert_yaxis=False)
        
    def cultural_selection_plot(self):
        plt.plot(self.out[1])
        plt.xlabel("Number of times an asymmetric game is played")
        plt.ylabel("Average CPR value")
        plt.xticks(range(11))
        plt.show()
        
        alpha = plt.plot(self.out[2], label = "alpha", marker = "d")
        beta = plt.plot(self.out[3], label = "beta", marker = "s")
        plt.legend()
        plt.xticks(range(11))
        plt.xlabel("Number of times an asymmetric game is played")
        plt.ylabel("Average values for alpha and beta")
        plt.ylim(-1, 1)
        plt.hlines(0, 0, 10)
        plt.show()