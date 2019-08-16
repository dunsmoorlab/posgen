import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

pospal = sns.color_palette('mako_r',5)
fearpal = sns.color_palette('hot_r',5)
def boot_ef(diff,X,Y,data,palette):
    sns.set_context('talk');sns.set_style('ticks')
    ef, ef_dist = pg.compute_bootci(x=diff,func='mean',method='cper',
                            confidence=.95,n_boot=5000,decimals=4,seed=42,return_dist=True)
    
    fig, ax = plt.subplots(1,2,sharey=True) #graph
    # sns.pointplot(x=X,y=Y,data=data,join=False,ax=ax[0],
    #               color='black',capsize=.3,nboot=5000) #means
    sns.swarmplot(x=X,y=Y,data=data,ax=ax[0],
                  linewidth=2,edgecolor='black',size=8,
                  palette=[palette[2]]) #swarmplot
    sns.kdeplot(ef_dist,shade=True,color='grey',vertical=True,ax=ax[1]) #effect size
    
    xdraw = ax[1].get_xlim()[1] / 2 #how far to draw horizontal lines (for editing in illustrator)
    desat_r = sns.desaturate('red',.8)
    ax[1].vlines(0,ef[0],ef[1],color=desat_r,linewidth=2) #underline the 95% CI of effect in red
    y2 = ef_dist.mean(); ax[1].scatter(0,y2,s=16,color=desat_r) #draw line for mean effect
    x2 = ax[1].get_xlim(); ax[1].hlines(0,x2[0],x2[1],color='grey',linestyle='--') #draw line at 0 effect
    
    sns.despine(ax=ax[0]); sns.despine(ax=ax[1],left=True,right=True) #despine the plots for aesthetics

    ax[1].set_title('%s 95 CI = %s'%(Y,ef))

pfc = pd.read_csv('pg_fc.csv').groupby(['subject','face']).mean().reset_index()
pdiff = pd.DataFrame({'scr':pfc.scr[pfc.face == .55].values - pfc.scr[pfc.face == .11].values,
                      'rt':pfc.rt[pfc.face == .55].values - pfc.rt[pfc.face == .11].values,
                      'face':'diff'})
for val in ['scr','rt']: boot_ef(pdiff[val],'face',val,pdiff,pospal)

ffc = pd.read_csv('fg_fc.csv').groupby(['subject','face']).mean().reset_index()
fdiff = pd.DataFrame({'scr':ffc.scr[ffc.face == .55].values - ffc.scr[ffc.face == .11].values,
                      'rt':ffc.rt[ffc.face == .55].values - ffc.rt[ffc.face == .11].values,
                      'face':'diff'})
for val in ['scr','rt']: boot_ef(fdiff[val],'face',val,fdiff,fearpal)

def two_samp_comp(T,C,X,Y,data,palette):
    sns.set_context('talk');sns.set_style('ticks')
    ef, ef_dist = pg.compute_bootci(x=T,y=C,func='mean',method='cper',
                            confidence=.95,n_boot=5000,decimals=4,seed=42,return_dist=True)
    
    ybias = T.mean() - ef_dist.mean() #calculate how much to shift up the effect for graphing
    plot_ef, plot_dist = ef + ybias, ef_dist + ybias

    fig, ax = plt.subplots(1,2,sharey=False) #graph
    sns.pointplot(x=X,y=Y,data=data,join=False,ax=ax[0],
                  color='black',capsize=.3,nboot=5000) #means
    sns.swarmplot(x=X,y=Y,data=data,ax=ax[0],
                  linewidth=2,edgecolor='black',size=8,
                  palette=[palette[0],palette[2]]) #swarmplot
    sns.kdeplot(plot_dist,shade=True,color='grey',vertical=True,ax=ax[1]) #effect size
    
    xdraw = ax[1].get_xlim()[1] / 2 #how far to draw horizontal lines (for editing in illustrator)
    # for ci in plot_ef: ax[1].hlines(ci,0,xdraw,linestyle=':')
    
    ax[1].vlines(0,plot_ef[0],plot_ef[1],color='red',linewidth=2) #underline the 95% CI of effect in red
    y2 = plot_dist.mean(); ax[1].scatter(0,y2,s=16,color='red') #draw line for mean effect
    x2 = ax[1].get_xlim(); ax[1].hlines(0+ybias,x2[0],x2[1],color='grey',linestyle='--') #draw line at 0 effect
    
    sns.despine(ax=ax[0]); sns.despine(ax=ax[1],right=False) #despine the plots for aesthetics

    #this is all to shift the effect size y-limit to be centered on the mean of the test group
    ax[1].set_ylim(ax[0].get_ylim())
    newticks = np.round(ax[1].get_yticks() - ybias,2)
    ax[1].set_yticklabels(newticks)
    ax[1].yaxis.tick_right()

    ax[1].set_title('%s 95 CI = %s'%(Y,ef))




###########Explicit Selection#############
select = pd.read_csv('express_selection.csv')
select['n_subs'] = select.percentage * 19
pS = select[select.exp == 1]
fS = select[select.exp == 2]

fig, ax = plt.subplots()
sns.barplot(x='face',y='n_subs',data=pS,ax=ax,palette=pospal)
sns.despine(ax=ax)

fig, ax = plt.subplots()
sns.barplot(x='face',y='n_subs',data=fS,ax=ax,palette=fearpal)
sns.despine(ax=ax)

###########Subjective Face Ratings########
def emo_ident(data,palette,fc=True):
    sns.set_context('talk');sns.set_style('ticks')

    if fc:
        N_trials = 15
        palette = [palette[0],palette[2]]
    else: N_trials = 9

    data['coded_response'] = data.response.astype(float) - 1 
    data = data.groupby(['subject','face']).coded_response.sum() / N_trials
    data = data.reset_index()
    data = data.groupby(['face']).mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='face',y='coded_response',data=data,palette=palette)
    ax.set_ylim([0,1])
    sns.despine(ax=ax)

pgen = pd.read_csv('posgen_data.csv')
pg_fc = pd.read_csv('pg_fc.csv')
fgen = pd.read_csv('feargen_data.csv')
fg_fc = pd.read_csv('fg_fc.csv')
emo_ident(pgen,pospal,fc=False)
emo_ident(pg_fc,pospal,fc=True)
emo_ident(fgen,fearpal,fc=False)
emo_ident(fg_fc,fearpal,fc=True)
##########Generalization####################

def gengraph(data,palette):
    sns.set_context('talk');sns.set_style('ticks')
    
    face = data.groupby(['subject','face']).mean().reset_index()
    for val in ['scr','rt']:
        fig, ax = plt.subplots()
        sns.pointplot(x='face',y=val,data=face,join=False,capsize=.3,nboot=5000,color='black')
        sns.swarmplot(x='face',y=val,data=face,palette=palette,ax=ax,linewidth=2,edgecolor='black',size=8)
        sns.despine(ax=ax)

    phase = data.groupby(['subject','face','phase']).mean().reset_index()
    for val in ['scr','rt']:
        fig, ax = plt.subplots(1,3,sharey=True)
        for i, p in enumerate([1,2,3]):
            sns.pointplot(x='face',y=val,data=phase.query('phase == @p'),join=False,capsize=.3,nboot=5000,color='black',ax=ax[i])
            sns.stripplot(x='face',y=val,data=phase.query('phase == @p'),palette=palette,linewidth=2,edgecolor='black',size=8,ax=ax[i])
            sns.despine(ax=ax[i])

gengraph(pgen,pospal)
gengraph(fgen,fearpal)

##########curve fitting##################
sns.set_context('talk');sns.set_style('ticks')
from curve_fit import *
cpal = [pospal[2],fearpal[2]]
p = pos_curve(exp='posgen')
f = pos_curve(exp='feargen')
curves = pd.concat([p.curves,f.curves])

#curves
fig, ax = plt.subplots(1,3)
for i, phase in enumerate([1,2,3]):
    sns.lineplot(x='face',y='scr_est',hue='exp',data=curves.query('phase == @phase'),
                    ax=ax[i],n_boot=5000,palette=cpal,hue_order=['posgen','feargen'])
    sns.despine(ax=ax[i]);ax[i].legend_.remove()
    ax[i].set_xlim([0,1])
    ax[i].set_ylim([0,1])

#coefs & peak
coefs = pd.concat([p.coefs,f.coefs])
#seperate out the peaks and coefs
Peak = coefs[coefs.coef == 'peak'].copy()
Peak.est /= 100
Peak.est = Peak.est.astype(float)
coefs_ = coefs[coefs.coef != 'peak'].copy()

#coefs
fig, cax = plt.subplots(1,3)
for i, phase in enumerate([1,2,3]):
    sns.barplot(x='coef',y='est',hue='exp',data=coefs_.query('phase == @phase'),n_boot=5000,ax=cax[i],
                capsize=0,palette=cpal)#,join=False,dodge=True,edgecolor='black')
    # sns.stripplot(x='coef',y='est',hue='exp',data=coefs_.query('phase == @phase'),palette=cpal,
    #                 ax=cax[i],linewidth=2,edgecolor='black',size=8,dodge=True)
    sns.despine(ax=cax[i]);cax[i].legend_.remove()


#peak
fig, pax = plt.subplots(1,3)
for i, phase in enumerate([1,2,3]):
    sns.pointplot(x='coef',y='est',hue='exp',data=Peak.query('phase == @phase'),dodge=True,palette=cpal,n_boot=5000,
                    ax=pax[i],join=False,capsize=.3)
    sns.stripplot(x='coef',y='est',hue='exp',data=Peak.query('phase == @phase'),palette=cpal,
                    ax=pax[i],linewidth=2,edgecolor='black',size=8,dodge=True)
    sns.despine(ax=pax[i]);pax[i].legend_.remove()

