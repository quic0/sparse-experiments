# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:50:56 2017

@author: Philipp
"""

#%%

import pandas as pd
import matplotlib.pyplot as plt

#%%
def figsize(scale):
    fig_width_pt = 505.89                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*0.25              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size
#%%
# Read accuracy
knn_acc = pd.read_csv('results le2/LE2KNNacc.txt',header=None,sep=' ')
snc_acc = pd.read_csv('results le2/LE2SNCacc.txt',header=None,sep=' ')
svm_acc = pd.read_csv('results le2/LE2SVMacc.txt',header=None,sep=' ')
le2_den = pd.read_csv('results le2/LE2KNNden.txt',header=None,sep=' ')
#%%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize(1))

# Create accuracy plot
plt.subplot(1,2,1)
plt.plot(knn_acc[0],knn_acc[1], label='KNN',marker='o',linestyle='--',markersize=10, zorder=3, mfc='none')
plt.plot(snc_acc[0],knn_acc[1], label='SNC',marker='x',linestyle='--',markersize=10, zorder=2)
plt.plot(svm_acc[0],knn_acc[1], label='SVM',marker='+',linestyle='--',markersize=15, zorder=1)
plt.xlabel('Grid resolution')
plt.ylabel('Accuracy [\%]')
plt.legend()
ax = plt.gca()
ax.set_xlim([2,20])
ax.set_ylim([70,100])

# Create density plot
plt.subplot(1,2,2)
plt.plot(le2_den[0],le2_den[1], label='Density')
plt.xlabel('Grid resolution')
plt.ylabel('Density [\%]')
plt.legend()
ax = plt.gca()
ax.set_xlim([2,20])
ax.set_ylim([0,100])
plt.tight_layout()

plt.savefig('le2.pdf',bbox_inches="tight")
#%%