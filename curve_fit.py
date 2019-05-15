import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
# from pg_config import *
import numpy.polynomial.polynomial as poly
from wesanderson import wes_palettes

#read in data
data = pd.read_csv('posgen_data.csv')

#initialize some variables
subs = data.subject.unique()
#create the face space
facex = np.linspace(0,1,100)
phases = [1,2,3]
_coefs_ = ['a','b','c','peak']

#create output dfs
curves = pd.DataFrame([],columns=['scr_est'],index=pd.MultiIndex.from_product(
						[subs,phases,facex],
						names=['subject','phase','face']))

coefs = pd.DataFrame([],columns=['est'],index=pd.MultiIndex.from_product(
						[subs,phases,_coefs_],
						names=['subject','phase','coef']))

#fit curves
for sub in subs:
	for phase in phases:
		
		#isolate each subjects data and order it by the x variable, face
		subdat = data[data.subject == sub][data.phase == phase].copy()
		subdat = subdat.sort_values(by='face')
		
		#fit a 2nd order polynomial to the data
		p = poly.Polynomial.fit(subdat.face,subdat.scr,2,domain=[0,1])
		
		#collect the predicted y values
		curve = p.linspace()[1]
		
		#save
		curves.loc[(sub,phase),'scr_est'] = curve

		#find the max value
		maxima = np.where(curve == curve.max())[0][0]

		#save the max and the curve coefs
		coefs.loc[(sub,phase,'a')]    = p.coef[0]
		coefs.loc[(sub,phase,'b')]    = p.coef[1]
		coefs.loc[(sub,phase,'c')]    = p.coef[2]
		coefs.loc[(sub,phase,'peak')] = maxima

#reset indices for graphing
curves.reset_index(inplace=True)
#this needs to happen for seaborn
curves = curves.astype(float)
curves.phase = curves.phase.astype(float)
coefs.reset_index(inplace=True)

#some style settings
sns.set_context('notebook')
sns.set_style('whitegrid')
pos_pal = sns.color_palette('mako_r',3)
# pos_pal = sns.color_palette(wes_palettes['Zissou'],3)

#graph the results by phase showing all subjects
fig, ax = plt.subplots(1,3,sharey=True)
for i, phase in enumerate(phases):
	sns.lineplot(x='face',y='scr_est',estimator=None,hue='subject',
			units='subject',data=curves.query('phase == %s'%(phase)),ax=ax[i],
			palette='mako_r')
	ax[i].grid(b=False,axis='y')
	ax[i].set_title('phase %s'%(phase))
	ax[i].legend_.remove()

#phase averages
fig, ax2 = plt.subplots()
ax2 = sns.lineplot(x='face',y='scr_est',hue='phase',data=curves,
					palette=pos_pal)

#seperate out the peaks and coefs
Peak = coefs[coefs.coef == 'peak']
Peak.est /= 100
coefs_ = coefs[coefs.coef != 'peak']

#coefs
plt.subplots()
cplot = sns.pointplot(x='coef',y='est',hue='phase',data=coefs_,dodge=True,palette=pos_pal)
#peak
plt.subplots()
pplot = sns.pointplot(x='coef',y='est',hue='phase',data=Peak,dodge=True,palette=pos_pal)


#examples of curve fitting

# coef = poly.polyfit(x,y,2)
# ffit = poly.polyval(x,coef)
# plt.plot(x,ffit)

# p = poly.Polynomial.fit(x,y,2,domain=[0,1])
# plt.plot(*p.linspace())

# #the coef we want is p.coef[2] (should be negative)