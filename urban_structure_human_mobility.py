#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats
import matplotlib.ticker as mticker
from matplotlib import rc, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
from matplotlib import legend_handler
from mycolorpy import colorlist as mcp
import func_timeout
import pomegranate
import networkx as nx
import pickle as p
import geopandas as gpd
from shapely.geometry import Polygon, Point
import rtree
import pygeos
import json


# In[13]:




def figure_monocentric_hypothesis(df_distribution_network, predictions, city):
    
    print('This is a figure corresponding to the main nucleus in ' + str(city) + '.')
    fig, ax = plt.subplots(figsize = (15,12))
    ax.tick_params(axis = 'both', which = 'both', width = 0, length = 0, color = 'k', labelsize = 30, pad=10)
    ax.text(0.5, -0.14, r'Distance between $S_1$ and $S_i$, $d_{1i}$', ha='center', rotation=0, size = 32, transform=ax.transAxes)
    ax.text(-0.1, 0.5, r'$E[L_i] \approx \hat{\mu}_i$', ha='center', va='center', rotation=90, size = 32, transform=ax.transAxes)
    ax.set_xticks(np.arange(0, max(df_distribution_network['network_distance_to_nucleus'])+1, 5.0))
    ax.set_ylim([-2,44])
    ax.set_yticks(np.arange(0, 43+1, 10.0))
    plt.scatter(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['average_length_journeys'], s=2*df_distribution_network.no_journeys/100, color='None', ec=(36/255,1/255,84/255,1), lw=0.9, alpha=1)
    plt.plot(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['network_distance_to_nucleus'], color='firebrick', lw=5, linestyle='--', dashes=(1, 0.51)) 
    plt.plot(df_distribution_network['network_distance_to_nucleus'], predictions['mean'], color=(36/255,1/255,84/255,1), lw=5)
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions['mean_ci_lower'], predictions['mean_ci_upper'], alpha=.5, color=(36/255,1/255,84/255,1))
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions['obs_ci_lower'], predictions['obs_ci_upper'], alpha=.1, color=(36/255,1/255,84/255,1))
    legend_elements = [Line2D([0], [0], lw=5, color='firebrick', linestyle='--', dashes=(1, 0.51), markeredgewidth=0.5, mfc=(36/255,1/255,84/255,0.6), mec='k', marker='o', markersize=0, label=r'$ E[L_i] = d_{1i}$'),
                      Line2D([0], [0], lw=5, color=(36/255,1/255,84/255,1), label= 'Regression line'),
                      Patch(facecolor=(36/255,1/255,84/255,0.5), edgecolor='None', label= 'Confidence interval'),
                      Patch(facecolor=(36/255,1/255,84/255,0.1), edgecolor='None', label= 'Prediction interval'),
                      Line2D([0], [0], lw=0, markeredgewidth=1, mfc=(1,1,1,0), mec=(36/255,1/255,84/255,1), marker='o', markersize=30, label= 'Observed'),
                      ]
    legend = ax.legend(handles=legend_elements, handlelength=2.47, fontsize = 22, shadow=False,
                       facecolor = 'lightgray', fancybox= None, loc=(0.03,0.52), ncol=1, columnspacing=1.2, borderpad=0.5)
    for t in legend.get_texts():
        t.set_ha('left')
    legend.get_frame().set_alpha(0.3)
    legend.get_frame().set_edgecolor('lightgray')
    legend.get_frame().set_linewidth(2)
    ax.scatter([0.15,0.15], [0.823,0.848], s=[int(2.2*256/100)+60, int(2.2*79504/100)], ec = 'k', facecolor='None', lw=2, transform=ax.transAxes)
    ax.text(0.22, 0.95, 'Number of journeys', transform=ax.transAxes, va='top', ha='center', rotation=0, color='k', size = 22, zorder=11)
    ax.text(0.23, 0.82, str(int(min(df_distribution_network.no_journeys))), ha='left', rotation=0, color='k', size = 22, zorder=11, transform=ax.transAxes)
    ax.text(0.23, 0.87, str(int(max(df_distribution_network.no_journeys))), ha='left', rotation=0, color='k', size = 22, zorder=11, transform=ax.transAxes)
    ax.plot([0.15, 0.21], [0.88,0.88], lw=2, c='k', transform=ax.transAxes)
    ax.plot([0.15, 0.21], [0.83,0.83], lw=2, c='k', transform=ax.transAxes)

    
    plt.show()


# In[14]:


def monocentric_hypothesis(df_distribution_network, i, city):

    df_distribution_network = df_distribution_network.sort_values(by='network_distance_to_nucleus').reset_index(drop=True)
    df_distribution_network = df_distribution_network[df_distribution_network['network_distance_to_nucleus']>0].reset_index(drop=True)
    model = smf.ols('average_length_journeys ~ network_distance_to_nucleus', df_distribution_network)
    results = model.fit()
    alpha = .05
    if i == 0:
        predictions = results.get_prediction(df_distribution_network).summary_frame(alpha)
        figure_monocentric_hypothesis(df_distribution_network, predictions, city)
    
    return list(results.params) + list(results.bse) + list(scipy.stats.pearsonr(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['average_length_journeys']))


# In[15]:


def fit_station_2mix(x):
# Input: all the observed lengths of journeys terminating at a given station  
# Output: the parameters of the 2-component Poisson mixture that best describes the data for an individual station
    
    model = pomegranate.GeneralMixtureModel.from_samples(pomegranate.PoissonDistribution, 2, x.reshape((len(x), 1)), weights=None)
    json_model = json.loads(model.to_json())
    mu1 = model.distributions[0].parameters[0]
    mu2 = model.distributions[1].parameters[0]
    proximal_w = json_model['weights'][0]
    distal_w = json_model['weights'][1]
    if mu1<mu2:
        proximal_x = mu1
        proximal_y = model.probability([mu1])[0]
        distal_x = mu2
        distal_y = model.probability([mu2])[0]
    elif mu2<mu1:
        proximal_x = mu2
        proximal_y = model.probability([mu2])[0]
        distal_x = mu1
        distal_y = model.probability([mu1])[0]
        
    return [proximal_x, proximal_y, proximal_w, distal_x, distal_y, distal_w], model


# In[16]:



def stations_2mix(df_distribution_network):
# Input: the data base for all the stations with the network distance from each station to the chosen nucleus
# Output: the same data base with extra columns for the 2-component mixture model fitted to each station, 
#         BIC for all the stations, Log-likelihood for whole model, no. of degrees of freedom and no. of parameters
    
    df_distribution_network['proximal_x_2mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['proximal_y_2mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['proximal_w_2mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['distal_x_2mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['distal_y_2mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['distal_w_2mix'] = np.zeros(len(df_distribution_network))
    ll_2mix = 0
    dof_2mix = 0
    n_params_2mix = 0
    for i in range(len(df_distribution_network)):
        frequencies = np.array([df_distribution_network.loc[i, str(j)] for j in range(1,51)])
        lengths = np.array([])
        for j in range(len(frequencies)):
            frequency = frequencies[j]
            for k in range(int(frequency)):
                lengths = np.append(lengths, j+1)
        station_2mix, model = func_timeout.func_timeout(20, fit_station_2mix, args=(lengths,))
        df_distribution_network.loc[i, 'proximal_x_2mix'] = station_2mix[0]
        df_distribution_network.loc[i, 'proximal_y_2mix'] = station_2mix[1]
        df_distribution_network.loc[i, 'proximal_w_2mix'] = station_2mix[2]
        df_distribution_network.loc[i, 'distal_x_2mix'] = station_2mix[3]
        df_distribution_network.loc[i, 'distal_y_2mix'] = station_2mix[4]
        df_distribution_network.loc[i, 'distal_w_2mix'] = station_2mix[5]
        ll_2mix += np.sum(model.log_probability(lengths))
        dof_2mix += len(lengths)
        n_params_2mix += 2*2 #2 components, with one weight and one parameter each
    bic_2mix = np.log(dof_2mix)*n_params_2mix - 2*ll_2mix

    return df_distribution_network, bic_2mix, ll_2mix, dof_2mix, n_params_2mix        

 


# In[19]:




def figure_model_2mix(df_distribution_network, predictions_proximal_2mix, predictions_distal_2mix, city):
    
    print('This is a figure corresponding to the main nucleus in ' + str(city) + '.')
    fig, ax = plt.subplots(figsize=(18,15))
    ax.tick_params(axis = 'both', which = 'both', width = 0, length = 0, color = 'k', labelsize = 32, pad=10)
    ax.text(0.483, -0.10,  r'Distance between $S_1$ and $S_i$, $d_{1i}$', ha='center', rotation=0, size = 32, transform=ax.transAxes)
    ax.text(-0.09, 0.5, 'Proximal and distal mean, $\hat{\mu}^p_i$ and $\hat{\mu}^d_i$', va='center', ha='center', rotation=90, size = 32, transform=ax.transAxes)
    ax.set_xticks(np.arange(0, max(df_distribution_network['network_distance_to_nucleus'])+1, 5.0))
    ax.set_ylim([-2,44])
    ax.set_yticks(np.arange(0, 43+1, 10.0))
    plt.scatter(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['proximal_x_2mix'], s=2*df_distribution_network.no_journeys/100, color='None', ec=(36/255,1/255,84/255), lw=1, alpha=1)
    plt.plot(df_distribution_network['network_distance_to_nucleus'], predictions_proximal_2mix['mean'], color=(36/255,1/255,84/255,1), lw=6)
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_proximal_2mix['mean_ci_lower'], predictions_proximal_2mix['mean_ci_upper'], alpha=.5, color=(36/255,1/255,84/255,1))
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_proximal_2mix['obs_ci_lower'], predictions_proximal_2mix['obs_ci_upper'], alpha=.1, color=(36/255,1/255,84/255,1))

    plt.scatter(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['distal_x_2mix'], s=2*df_distribution_network.no_journeys/100, color='None', ec=(180/255,222/255,44/255,1), lw=1.5, alpha=1)
    plt.plot(df_distribution_network['network_distance_to_nucleus'], predictions_distal_2mix['mean'], color=(180/255,222/255,44/255,1), lw=6)
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_distal_2mix['mean_ci_lower'], predictions_distal_2mix['mean_ci_upper'], alpha=.35, color=(180/255,222/255,44/255,1))
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_distal_2mix['obs_ci_lower'], predictions_distal_2mix['obs_ci_upper'], alpha=.15, color=(180/255,222/255,44/255,1))

    legend_elements = [Line2D([0], [0], lw=5, color=(36/255,1/255,84/255,1), label= 'Regre. line'),
                      Patch(facecolor=(36/255,1/255,84/255,0.5), edgecolor='None', label= 'Conf. int.'),
                      Patch(facecolor=(36/255,1/255,84/255,0.1), edgecolor='None', label= 'Pred. int.'),
                      Line2D([0], [0], lw=0, markeredgewidth=1, mfc=(1,1,1,0), mec=(36/255,1/255,84/255,1), marker='o', markersize=30, label= 'Observed'),
                      Line2D([0], [0], lw=5, color=(180/255,222/255,44/255,1), label= 'Regre. line'),
                      Patch(facecolor=(180/255,222/255,44/255,0.5), edgecolor='None', label= 'Conf. int.'),
                      Patch(facecolor=(180/255,222/255,44/255,0.1), edgecolor='None', label= 'Pred. int.'),
                      Line2D([0], [0], lw=0, markeredgewidth=1, mfc=(1,1,1,0), mec=(180/255,222/255,44/255,1), marker='o', markersize=30, label= 'Observed'),
                      ]
    legend = ax.legend(handles=legend_elements, handlelength=2.4, markerfirst = True, fontsize = 22, shadow=False,
                       facecolor = 'lightgray', fancybox= None, loc=(0.03,0.76), ncol=2, columnspacing=1.2, borderpad=0.7)
    for t in legend.get_texts():
        t.set_ha('left')
    legend.get_frame().set_alpha(0.3)
    legend.get_frame().set_edgecolor('lightgray')
    legend.get_frame().set_linewidth(2)
    ax.text(0.15, 0.95,  'Proximal', ha='center', rotation=0, fontweight= 'bold', size = 22, transform=ax.transAxes)
    ax.text(0.36, 0.95,  'Distal', ha='center', rotation=0, fontweight= 'bold', size = 22, transform=ax.transAxes)
    ax.scatter([0.784,0.784], [0.83,0.851], s=[int(2.2*256/100)+60, int(2.2*79504/100)], ec = 'k', facecolor='None', lw=2, transform=ax.transAxes)
    ax.text(0.85, 0.935, 'Number of journeys', transform=ax.transAxes, va='top', ha='center', rotation=0, color='k', size = 22, zorder=11)
    ax.text(0.85, 0.827, str(int(min(df_distribution_network.no_journeys))), ha='left', rotation=0, color='k', size = 22, zorder=11, transform=ax.transAxes)
    ax.text(0.85, 0.869, str(int(max(df_distribution_network.no_journeys))), ha='left', rotation=0, color='k', size = 22, zorder=11, transform=ax.transAxes)
    ax.plot([0.784, 0.84], [0.877,0.877], lw=2, c='k', transform=ax.transAxes)
    ax.plot([0.784, 0.84], [0.835,0.835], lw=2, c='k', transform=ax.transAxes)

    plt.show()
    
    


# In[20]:


def model_2mix(df_distribution_network, i, city):  
    
    df_distribution_network = df_distribution_network.sort_values(by='network_distance_to_nucleus').reset_index(drop=True)
    df_distribution_network = df_distribution_network[df_distribution_network['network_distance_to_nucleus']>0].reset_index(drop=True)
    df_distribution_network, bic_2mix, ll_2mix, dof_2mix, n_params_2mix = stations_2mix(df_distribution_network)
    model_proximal_2mix = smf.ols('proximal_x_2mix ~ network_distance_to_nucleus', df_distribution_network)
    results_proximal_2mix = model_proximal_2mix.fit()
    model_distal_2mix = smf.ols('distal_x_2mix ~ network_distance_to_nucleus', df_distribution_network)
    results_distal_2mix = model_distal_2mix.fit()
    alpha = .05
    if i == 0:
        predictions_proximal_2mix = results_proximal_2mix.get_prediction(df_distribution_network).summary_frame(alpha)
        predictions_distal_2mix = results_distal_2mix.get_prediction(df_distribution_network).summary_frame(alpha)
        figure_model_2mix(df_distribution_network, predictions_proximal_2mix, predictions_distal_2mix, city)
    results_model_2mix = list(results_proximal_2mix.params) + list(results_proximal_2mix.bse) + list(scipy.stats.pearsonr(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['proximal_x_2mix'])) + list(results_distal_2mix.params) + list(results_distal_2mix.bse) + list(scipy.stats.pearsonr(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['distal_x_2mix'])) + [ll_2mix, n_params_2mix, dof_2mix, bic_2mix]

    return results_model_2mix
    


# In[1]:



def fit_station_3mix(x):
# Input: all the observed lengths of journeys terminating at a given station  
# Output: the parameters of the 3-component Poisson mixture that best describes the data for an individual station
    
    model = pomegranate.GeneralMixtureModel.from_samples(pomegranate.PoissonDistribution, 3, x.reshape((len(x), 1)), weights=None)
    json_model = json.loads(model.to_json())
    mu1 = model.distributions[0].parameters[0]
    mu2 = model.distributions[1].parameters[0]
    mu3 = model.distributions[2].parameters[0]
    proximal_w = json_model['weights'][0]
    medial_w = json_model['weights'][1]
    distal_w = json_model['weights'][2]
    if mu1 == min(mu1, mu2, mu3):
        proximal_x = mu1
        proximal_y = model.probability([mu1])[0]
        if mu2 == max(mu2, mu3): 
            distal_x = mu2
            distal_y = model.probability([mu2])[0]
            medial_x = mu3
            medial_y = model.probability([mu3])[0]
        else:
            distal_x = mu3
            distal_y = model.probability([mu3])[0]
            medial_x = mu2
            medial_y = model.probability([mu2])[0]
    elif mu2 == min(mu1, mu2, mu3):
        proximal_x = mu2
        proximal_y = model.probability([mu2])[0]
        if mu1 == max(mu1, mu3): 
            distal_x = mu1
            distal_y = model.probability([mu1])[0]
            medial_x = mu3
            medial_y = model.probability([mu3])[0]
        else:
            distal_x = mu3
            distal_y = model.probability([mu3])[0]
            medial_x = mu1
            medial_y = model.probability([mu1])[0]
    elif mu3 == min(mu1, mu2, mu3):
        proximal_x = mu3
        proximal_y = model.probability([mu3])[0]
        if mu1 == max(mu1, mu2): 
            distal_x = mu1
            distal_y = model.probability([mu1])[0]
            medial_x = mu2
            medial_y = model.probability([mu2])[0]
        else:
            distal_x = mu2
            distal_y = model.probability([mu2])[0]
            medial_x = mu1
            medial_y = model.probability([mu1])[0]
        
    return [proximal_x, proximal_y, proximal_w, medial_x, medial_y, medial_w, distal_x, distal_y, distal_w], model


# In[ ]:



def stations_3mix(df_distribution_network):
# Input: the data base for all the stations with the network distance from each station to the chosen nucleus
# Output: the same data base with extra columns for the 3-component mixture model fitted to each station, 
#         BIC for all the stations, Log-likelihood for whole model, no. of degrees of freedom and no. of parameters
    
    df_distribution_network['proximal_x_3mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['proximal_y_3mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['proximal_w_3mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['medial_x_3mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['medial_y_3mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['medial_w_3mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['distal_x_3mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['distal_y_3mix'] = np.zeros(len(df_distribution_network))
    df_distribution_network['distal_w_3mix'] = np.zeros(len(df_distribution_network))
    ll_3mix = 0
    dof_3mix = 0
    n_params_3mix = 0
    for i in range(len(df_distribution_network)):
        frequencies = np.array([df_distribution_network.loc[i, str(j)] for j in range(1,51)])
        lengths = np.array([])
        for j in range(len(frequencies)):
            frequency = frequencies[j]
            for k in range(int(frequency)):
                lengths = np.append(lengths, j+1)
        station_3mix, model = func_timeout.func_timeout(20, fit_station_3mix, args=(lengths,))
        df_distribution_network.loc[i, 'proximal_x_3mix'] = station_3mix[0]
        df_distribution_network.loc[i, 'proximal_y_3mix'] = station_3mix[1]
        df_distribution_network.loc[i, 'proximal_w_3mix'] = station_3mix[2]
        df_distribution_network.loc[i, 'medial_x_3mix'] = station_3mix[3]
        df_distribution_network.loc[i, 'medial_y_3mix'] = station_3mix[4]
        df_distribution_network.loc[i, 'medial_w_3mix'] = station_3mix[5]
        df_distribution_network.loc[i, 'distal_x_3mix'] = station_3mix[6]
        df_distribution_network.loc[i, 'distal_y_3mix'] = station_3mix[7]
        df_distribution_network.loc[i, 'distal_w_3mix'] = station_3mix[8]
        ll_3mix += np.sum(model.log_probability(lengths))
        dof_3mix += len(lengths)
        n_params_3mix += 3*2 #3 components, with one weight and one parameter each
    bic_3mix = np.log(dof_3mix)*n_params_3mix - 2*ll_3mix

    return df_distribution_network, bic_3mix, ll_3mix, dof_3mix, n_params_3mix   


# In[ ]:


def figure_model_3mix(df_distribution_network, predictions_proximal_3mix, predictions_medial_3mix, predictions_distal_3mix, city):
    
    print('This is a figure corresponding to the main nucleus in ' + str(city) + '.')
    fig, ax = plt.subplots(figsize=(18,15))
    ax.tick_params(axis = 'both', which = 'both', width = 0, length = 0, color = 'k', labelsize = 32, pad=10)
    ax.text(0.483, -0.10,  r'Distance between $S_1$ and $S_i$, $d_{1i}$', ha='center', rotation=0, size = 32, transform=ax.transAxes)
    ax.text(-0.09, 0.5, 'Proximal and distal mean, $\hat{\mu}^p_i$ and $\hat{\mu}^d_i$', va='center', ha='center', rotation=90, size = 32, transform=ax.transAxes)
    ax.set_xticks(np.arange(0, max(df_distribution_network['network_distance_to_nucleus'])+1, 5.0))
    ax.set_ylim([-2,54])
    ax.set_yticks(np.arange(0, 54+1, 10.0))
    
    plt.scatter(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['proximal_x_3mix'], s=2*df_distribution_network.no_journeys/100, color='None', ec=(36/255,1/255,84/255), lw=1, alpha=1)
    plt.plot(df_distribution_network['network_distance_to_nucleus'], predictions_proximal_3mix['mean'], color=(36/255,1/255,84/255,1), lw=6)
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_proximal_3mix['mean_ci_lower'], predictions_proximal_3mix['mean_ci_upper'], alpha=.5, color=(36/255,1/255,84/255,1))
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_proximal_3mix['obs_ci_lower'], predictions_proximal_3mix['obs_ci_upper'], alpha=.1, color=(36/255,1/255,84/255,1))

    plt.scatter(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['medial_x_3mix'], s=2*df_distribution_network.no_journeys/100, color='None', ec=(38/255,130/255,142/255), lw=1, alpha=1)
    plt.plot(df_distribution_network['network_distance_to_nucleus'], predictions_medial_3mix['mean'], color=(38/255,130/255,142/255,1), lw=6)
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_medial_3mix['mean_ci_lower'], predictions_medial_3mix['mean_ci_upper'], alpha=.5, color=(38/255,130/255,142/255,1))
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_medial_3mix['obs_ci_lower'], predictions_medial_3mix['obs_ci_upper'], alpha=.1, color=(38/255,130/255,142/255,1))

    plt.scatter(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['distal_x_3mix'], s=2*df_distribution_network.no_journeys/100, color='None', ec=(180/255,222/255,44/255,1), lw=1.5, alpha=1)
    plt.plot(df_distribution_network['network_distance_to_nucleus'], predictions_distal_3mix['mean'], color=(180/255,222/255,44/255,1), lw=6)
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_distal_3mix['mean_ci_lower'], predictions_distal_3mix['mean_ci_upper'], alpha=.35, color=(180/255,222/255,44/255,1))
    plt.fill_between(df_distribution_network['network_distance_to_nucleus'], predictions_distal_3mix['obs_ci_lower'], predictions_distal_3mix['obs_ci_upper'], alpha=.15, color=(180/255,222/255,44/255,1))

    legend_elements = [Line2D([0], [0], lw=5, color=(36/255,1/255,84/255,1), label= 'Regre. line'),
                  Patch(facecolor=(36/255,1/255,84/255,0.5), edgecolor='None', label= 'Conf. int.'),
                  Patch(facecolor=(36/255,1/255,84/255,0.1), edgecolor='None', label= 'Pred. int.'),
                  Line2D([0], [0], lw=0, markeredgewidth=1, mfc=(1,1,1,0), mec=(36/255,1/255,84/255,1), marker='o', markersize=30, label= 'Observed'),
                  Line2D([0], [0], lw=5, color=(38/255,130/255,142/255,1), label= 'Regre. line'),
                  Patch(facecolor=(38/255,130/255,142/255,0.5), edgecolor='None', label= 'Conf. int.'),
                  Patch(facecolor=(38/255,130/255,142/255,0.1), edgecolor='None', label= 'Pred. int.'),
                  Line2D([0], [0], lw=0, markeredgewidth=1, mfc=(1,1,1,0), mec=(38/255,130/255,142/255,1), marker='o', markersize=30, label= 'Observed'),
                  Line2D([0], [0], lw=5, color=(180/255,222/255,44/255,1), label= 'Regre. line'),
                  Patch(facecolor=(180/255,222/255,44/255,0.5), edgecolor='None', label= 'Conf. int.'),
                  Patch(facecolor=(180/255,222/255,44/255,0.1), edgecolor='None', label= 'Pred. int.'),
                  Line2D([0], [0], lw=0, markeredgewidth=1, mfc=(1,1,1,0), mec=(180/255,222/255,44/255,1), marker='o', markersize=30, label= 'Observed'),
                  ]
    legend = ax.legend(handles=legend_elements, handlelength=2.4, markerfirst = True, fontsize = 22, shadow=False,
                       facecolor = 'lightgray', fancybox= None, loc=(0.03,0.76), ncol=3, columnspacing=1.2, borderpad=0.7)
    for t in legend.get_texts():
        t.set_ha('left')
    legend.get_frame().set_alpha(0.3)
    legend.get_frame().set_edgecolor('lightgray')
    legend.get_frame().set_linewidth(2)
    ax.text(0.15, 0.95,  'Proximal', ha='center', rotation=0, fontweight= 'bold', size = 22, transform=ax.transAxes)
    ax.text(0.36, 0.95,  'Medial', ha='center', rotation=0, fontweight= 'bold', size = 22, transform=ax.transAxes)
    ax.text(0.57, 0.95,  'Distal', ha='center', rotation=0, fontweight= 'bold', size = 22, transform=ax.transAxes)

    ax.scatter([0.784,0.784], [0.83,0.851], s=[int(2.2*256/100)+60, int(2.2*79504/100)], ec = 'k', facecolor='None', lw=2, transform=ax.transAxes)
    ax.text(0.85, 0.935, 'Number of journeys', transform=ax.transAxes, va='top', ha='center', rotation=0, color='k', size = 22, zorder=11)
    ax.text(0.85, 0.827, str(int(min(df_distribution_network.no_journeys))), ha='left', rotation=0, color='k', size = 22, zorder=11, transform=ax.transAxes)
    ax.text(0.85, 0.869, str(int(max(df_distribution_network.no_journeys))), ha='left', rotation=0, color='k', size = 22, zorder=11, transform=ax.transAxes)
    ax.plot([0.784, 0.84], [0.877,0.877], lw=2, c='k', transform=ax.transAxes)
    ax.plot([0.784, 0.84], [0.835,0.835], lw=2, c='k', transform=ax.transAxes)

    plt.show()


# In[ ]:




def model_3mix(df_distribution_network, i, city):  
    
    df_distribution_network = df_distribution_network.sort_values(by='network_distance_to_nucleus').reset_index(drop=True)
    df_distribution_network = df_distribution_network[df_distribution_network['network_distance_to_nucleus']>0].reset_index(drop=True)
    df_distribution_network, bic_3mix, ll_3mix, dof_3mix, n_params_3mix = stations_3mix(df_distribution_network)
    model_proximal_3mix = smf.ols('proximal_x_3mix ~ network_distance_to_nucleus', df_distribution_network)
    results_proximal_3mix = model_proximal_3mix.fit()
    model_medial_3mix = smf.ols('medial_x_3mix ~ network_distance_to_nucleus', df_distribution_network)
    results_medial_3mix = model_medial_3mix.fit()
    model_distal_3mix = smf.ols('distal_x_3mix ~ network_distance_to_nucleus', df_distribution_network)
    results_distal_3mix = model_distal_3mix.fit()
    alpha = .05
    if i == 0:
        predictions_proximal_3mix = results_proximal_3mix.get_prediction(df_distribution_network).summary_frame(alpha)
        predictions_medial_3mix = results_medial_3mix.get_prediction(df_distribution_network).summary_frame(alpha)
        predictions_distal_3mix = results_distal_3mix.get_prediction(df_distribution_network).summary_frame(alpha)
        figure_model_3mix(df_distribution_network, predictions_proximal_3mix, predictions_medial_3mix, predictions_distal_3mix, city)
    results_model_3mix = list(results_proximal_3mix.params) + list(results_proximal_3mix.bse) + list(scipy.stats.pearsonr(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['proximal_x_3mix'])) + list(results_medial_3mix.params) + list(results_medial_3mix.bse) + list(scipy.stats.pearsonr(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['medial_x_3mix'])) + list(results_distal_3mix.params) + list(results_distal_3mix.bse) + list(scipy.stats.pearsonr(df_distribution_network['network_distance_to_nucleus'], df_distribution_network['distal_x_3mix'])) + [ll_3mix, n_params_3mix, dof_3mix, bic_3mix]

    return results_model_3mix

    




