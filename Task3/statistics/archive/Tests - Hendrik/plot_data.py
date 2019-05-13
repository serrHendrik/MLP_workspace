# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:40:57 2019

Although we live in an age present with dedicated statistical programming languages like R, 
here's a basic script to process the data generated from experiments on the first setting of the Harvest Game.
I ask forgiveness.

About the data:
Agents:
5 agents in the field

Environment:
50 initial apples
Apple growth rate: 0; 0.0008; 0; 0

For every alpha/beta combination (x), 20 games are played. The data per row here is thus a 20-game average.
"""

import numpy as np
import matplotlib.pyplot as plt

#Free Rider Data
fr_total = 444.4693878
fr_mean = 88.89387755
fr_std = 11.47294481
fr_gini = 0.071334719
fr_sustainability = 176.1579634
fr_timesteps = 424.9183673

#Inequity Averse Data
data = np.matrix([  [0.2,   267.7,	    53.54,	6.674959792,    0.060143286,    470.7778314,    999],
                    [0.25,  354.05,     70.81,  5.272127403,	0.039654107,	474.6674297,	999],
                    [0.33,  482.05,	    96.41,  4.447587327,	0.024832861,	484.4292932,	999],
                    [0.5,   853.7,      170.74, 4.345397861,	0.013445155,	508.189482,     999],
                    [0.66,  1042.3,     208.46,	4.517752455,	0.011591826,	503.6415208,	999],
                    [0.75,  1104.45,	220.89,	4.251939005,	0.010303655,	501.8338772,	999],
                    [0.8,   1143.1,	    228.62,	5.25171294,     0.01229639,     500.2102997,	999],
                    [0.83,  1043.35,	208.67,	5.819269404,	0.017086863,	449.5850953,	918.45],
                    [0.86,  1030.2,	    206.04,	7.227918252,	0.021008283,	436.6437434,	931.55],
                    [0.89,  1039.3,	    207.86,	7.561571988,	0.020184711,	437.7098762,	932.25],
                    [0.92,  874.75,	    174.95,	6.665504584,	0.023209826,	356.9909468,	768.3],
                    [0.93,  714.05,	    142.81,	9.132399869,	0.039395806,	297.1983237,	682.25],
                    [0.95,  602.7,      120.54,	7.724478215,	0.036019082,	249.7978624,	567.6]
                ])    
    
x = data[:,0]
total = data[:,1]
mean = data[:,2]
std = data[:,3]
gini = data[:,4]
sustainability = data[:,5]
timesteps = data[:,6]


# plot mean + std
x_list = np.transpose(x).tolist()[0]
mean_above = mean + std
mean_above = np.transpose(mean_above).tolist()[0]
mean_below = mean - std
mean_below = np.transpose(mean_below).tolist()[0]
where_array = [True for _ in range(len(x_list))]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(x,mean,'-',color="g")
ax.axhline(fr_mean,color="r")
ax.plot(x,mean_above,'--',color="g",alpha=0.5)
ax.plot(x,mean_below,'--',color="g",alpha=0.5)
ax.fill_between(x_list,mean_above,mean_below,where=where_array,alpha=0.2,facecolor='g')
ax.axhline(fr_mean + fr_std,linestyle='--',color="r",alpha=0.5)
ax.axhline(fr_mean - fr_std,linestyle='--',color="r",alpha=0.5)
ax.fill_between([0,1],fr_mean - fr_std,fr_mean + fr_std,alpha=0.2,color='r')
ax.set_xlabel("alpha / (alpha + beta)")
ax.set_ylabel("Mean consumption per agent (+std)")
ax.set_xlim(0.15,1)
ax.legend(["IA agents", "FR reference"])
plt.rcParams.update({'font.size': 16})
plt.grid()
plt.savefig("Images/HG1_IA_mean.png")
plt.show()

#Plot gini Inequality
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(x,gini,color="g")
ax.axhline(fr_gini,color="r")
ax.set_xlabel("alpha / (alpha + beta)")
ax.set_ylabel("Gini Index")
ax.set_xlim(0.15,1)
ax.legend(["IA agents", "FR reference"])
plt.rcParams.update({'font.size': 16})
plt.grid()
plt.savefig("Images/HG1_IA_gini.png")
plt.show()



#Plot timesteps
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(x,timesteps,color="g")
ax.axhline(fr_timesteps,color="r")
ax.set_xlabel("alpha / (alpha + beta)")
ax.set_ylabel("Timesteps")
ax.set_xlim(0.15,1)
ax.set_ylim(150,1100)
#ax.legend(["IA agents", "FR reference"])
#ax2 = ax.twinx()
ax.plot(x,sustainability,color="g",linestyle="--")
ax.axhline(fr_sustainability,color="r",linestyle="--")
#ax.set_xlabel("alpha / (alpha + beta)")
#ax2.set_ylabel("Sustainability")
#ax2.set_ylim(150,1100)
#ax2.legend(["IA agents", "FR reference"])
ax.legend(["IA game over", "FR game over ref.","IA sust.", "FR sust. ref."])
plt.rcParams.update({'font.size': 16})
plt.yticks(np.arange(100,1100,100))
plt.grid()
plt.savefig("Images/HG1_IA_timesteps.png")
plt.show()

