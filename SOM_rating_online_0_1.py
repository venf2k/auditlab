################################################################################
#
# M A I N
#
# ################################################################################

#cd  /media/user/_home1/apps/python/DL/Agri
#cd  D:/Apps/Python/DL/Agri
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines
from som_online import SOM

# reading data
input_data = pd.read_csv("data/DatasetNuovoConAAeBFiltrato.csv", delimiter = ";")
# visualize some data
input_data.iloc[:30,:]

# shuffling data
rating_data = input_data.iloc[np.random.permutation(len(input_data))]
trunc_data = rating_data[["ASSET_TURNOVER", "CFO/TOT debt", "Tot Debt/Tot Equity"]]
trunc_data.iloc[:20,:]

# normalizing data
trunc_data = (trunc_data - trunc_data.min() ) / ( trunc_data.max() - trunc_data.min() )
trunc_data.iloc[:10,:]

# som = SOM(x_size, y_size, num_features)
rating_som = SOM(3,3,3)
# Initial weights
init_fig = plt.figure()
rating_som.show_plot(init_fig, 1, 0)
plt.show()

rating_som.train(trunc_data.values,
          num_epochs=200,
          init_learning_rate=0.01
          )

#PREDICTION AND ANALYSIS
#Now, its time to predict. 
#Prediction means finding the Best Matching Unit(BMU)’s weight (i.e. RGB color) 
#and it’s index (i.e. co-ordinate) for each data points.
def predict(df):
    bmu, bmu_idx = rating_som.find_bmu(df.values)
    df['bmu'] = bmu
    df['bmu_idx'] = bmu_idx
    return df
clustered_df = trunc_data.apply(predict, axis=1)
clustered_df.iloc[0:20]

#Let’s use those labels now by joining original data with clustered_df.
joined_df = rating_data.join(clustered_df, rsuffix="_norm")
joined_df[0:20]

#Now, let’s visualize how the original data gets clustered in the SOM with the magic of matplotlib.
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines

fig = plt.figure()
# setup axes
ax = fig.add_subplot(111)
scale = 50
ax.set_xlim((0, rating_som.net.shape[0]*scale))
ax.set_ylim((0, rating_som.net.shape[1]*scale))
ax.set_title("AA and B rating Clustering by using SOM")

for x in range(0, rating_som.net.shape[0]):
    for y in range(0, rating_som.net.shape[1]):
        ax.add_patch(patches.Rectangle((x*scale, y*scale), scale, scale,
                     facecolor='white',
                     edgecolor='grey'))
legend_map = {}
        
for index, row in joined_df.iterrows():
    x_cor = row['bmu_idx'][0] * scale
    y_cor = row['bmu_idx'][1] * scale
    x_cor = np.random.randint(x_cor, x_cor + scale)
    y_cor = np.random.randint(y_cor, y_cor + scale)
    color = row['bmu'][0]
    marker = "$\\ " + row['Rating'][0]+"$"
    marker = marker.lower()
    ax.plot(x_cor, y_cor, color=color, marker=marker, markersize=10)
    label = row['Rating']
    if not label in legend_map:
        legend_map[label] =  mlines.Line2D([], [], color='black', marker=marker, linestyle='None',
                          markersize=10, label=label)
plt.legend(handles=list(legend_map.values()), bbox_to_anchor=(1, 1))
plt.show()
