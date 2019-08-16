import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
# from pg_config import *
import numpy.polynomial.polynomial as poly
from wesanderson import wes_palettes
from scipy.optimize import curve_fit
import pingouin as pg
#read in data
def binomial(x,a,b,c):
    return a*x**2 + b*x + c

class pos_curve():
    def __init__(self,exp='posgen'):

        self.data = pd.read_csv('%s_data.csv'%(exp))

        #initialize some variables
        self.subs = self.data.subject.unique()
        #create the face space
        self.facex = np.linspace(0,1,100)
        self.phases = [1,2,3]
        self._coefs_ = ['a','b','c','peak']

        self.create_output()
        self.collect_curves(exp=exp)

    def create_output(self):#create output dfs
        self.curves = pd.DataFrame([],columns=['scr_est'],index=pd.MultiIndex.from_product(
                        [self.subs,self.phases,self.facex],
                        names=['subject','phase','face']))

        self.coefs = pd.DataFrame([],columns=['est'],index=pd.MultiIndex.from_product(
                        [self.subs,self.phases,self._coefs_],
                        names=['subject','phase','coef']))

    def collect_curves(self,exp):#fit self.curves
        self.data.set_index(['subject','phase'],inplace=True)
        self.data.sort_index(inplace=True)

        for sub in self.subs:
            for phase in self.phases:
                #isolate each subjects data and order it by the x variable, face
                subdat = self.data.loc[(sub,phase)].copy()
                subdat = subdat.sort_values(by='face')
                
                if len(np.where(subdat.scr!=0)[0]) < 3:
                    a = np.nan
                    b = np.nan
                    c = np.nan
                    curve = np.repeat(np.nan,100)
                    maxima = np.nan

                else:   
                    #fit a 2nd order polynomial to the data
                    # p = poly.Polynomial.fit(subdat.face,subdat.scr,2,domain=[0,1])
                    p = curve_fit(binomial,subdat.face,subdat.scr,bounds=(
                        (-np.inf,-np.inf,-np.inf),(0,np.inf,np.inf)) )

                    #collect coef
                    a = p[0][0]
                    b = p[0][1]
                    c = p[0][2]

                    #collect the predicted y values
                    curve = binomial(self.facex,a,b,c)
                    # curve = p.linspace()[1]
                    
                    #find the max value 
                    maxima = np.where(curve == curve.max())[0][0]
                    # maxima = -1*(b/(2*a)) # this vertex
                
                #save curve
                self.curves.loc[(sub,phase),'scr_est'] = curve
                #save the max and the curve coefs
                self.coefs.loc[(sub,phase,'a')]    = a
                self.coefs.loc[(sub,phase,'b')]    = b
                self.coefs.loc[(sub,phase,'c')]    = c
                self.coefs.loc[(sub,phase,'peak')] = maxima             
                # coefs.loc[(sub,phase,'a')]    = p.coef[0]
                # coefs.loc[(sub,phase,'b')]    = p.coef[1]
                # coefs.loc[(sub,phase,'c')]    = p.coef[2]
                
        #reset indices for graphing #this needs to happen for seaborn
        self.curves.reset_index(inplace=True)
        self.curves = self.curves.astype(float)
        self.curves.phase = self.curves.phase.astype(float)
        #drop nan
        self.curves = self.curves.drop(index=np.where(np.isnan(self.curves.scr_est) == True)[0])
        self.curves['exp'] = exp

        self.coefs.reset_index(inplace=True)
        self.coefs.est = self.coefs.est.astype(float)
        self.coefs = self.coefs.drop(index=np.where(np.isnan(self.coefs.est) == True)[0])
        self.coefs['exp'] = exp

        self.data.reset_index(inplace=True)


    def vis_curve(self):
        #some style settings
        sns.set_context('talk');sns.set_style('ticks')
        # pos_pal = sns.color_palette('mako_r',3)
        pos_pal = ['tab:pink','darkgrey','tab:blue']
        # pos_pal = list((wes_palettes['Zissou'][0],wes_palettes['Zissou'][2],wes_palettes['Zissou'][-1]))
        #graph the results by phase showing all subjects
        # fig, ax = plt.subplots(1,3,sharey=True)
        # for i, phase in enumerate(self.phases):
        #     sns.lineplot(x='face',y='scr_est',estimator=None,hue='subject',
        #             units='subject',data=self.curves.query('phase == %s'%(phase)),ax=ax[i],
        #             palette=sns.color_palette(wes_palettes['Zissou'],n_colors=len(self.subs)))
        #     ax[i].grid(b=False,axis='y');ax[i].set_title('phase %s'%(phase));ax[i].legend_.remove();
            
        #     sns.despine(ax=ax[i]);ax[i].set_ylim(0,1.5);ax[i].set_xlim(0,1.05)
        
        # phase averages
        avg_lines = self.curves.groupby(['phase','face']).mean().reset_index()
        err_lines = self.curves.groupby(['phase','face']).sem().reset_index()
        
        fig, ax2 = plt.subplots(1,3)
        for i, phase in enumerate([1,2,3]):
            sns.lineplot(x='face',y='scr_est',hue='phase',data=self.curves.query('phase == @phase'),
                            ax=ax2[i],n_boot=5000)
        # for i, phase in enumerate(self.phases):
        #     y    = avg_lines.query('phase == @phase').scr_est
        #     err  = err_lines.query('phase == @phase').scr_est
        #     # ax2.fill_between(self.facex,y-err,y+err,color=pos_pal[i],alpha=.5)
        # ax2.set_xlim(0,1);ax2.set_ylim(0);sns.despine(ax=ax2)
        # ax2.legend_.remove()

        #seperate out the peaks and coefs
        Peak = self.coefs[self.coefs.coef == 'peak'].copy()
        Peak.est /= 100
        Peak.est = Peak.est.astype(float)
        coefs_ = self.coefs[self.coefs.coef != 'peak'].copy()

        #coefs
        avg_coefs = coefs_.groupby(['phase','coef']).mean().reset_index()
        err_coefs = coefs_.groupby(['phase','coef']).sem().reset_index()

        fig, cplot = plt.subplots()
        sns.pointplot(x='coef',y='est',hue='phase',data=coefs_,dodge=.5,n_boot=5000,ax=cplot,join=False,capsize=.3,color='black')
        sns.swarmplot(x='coef',y='est',hue='phase',data=coefs_,palette=pos_pal,
                        ax=cplot,linewidth=1,edgecolor='black',dodge=True)
        # for i, coef in enumerate(err_coefs.coef.unique()):
        #     x = cplot.get_xticks()[i]
        #     x = [x-.25, x, x+.25]
        #     cplot.errorbar(x,avg_coefs.query('coef == @coef').est,err_coefs.query('coef == @coef').est,
        #                     ls='none',clip_on=True,color='black')       
        sns.despine(ax=cplot);cplot.legend_.remove()
        

        avg_peak = Peak.groupby(['phase','coef']).mean().reset_index()
        err_peak = Peak.groupby(['phase','coef']).sem().reset_index()
        #peak
        fig, pplot = plt.subplots()
        sns.pointplot(x='coef',y='est',hue='phase',data=Peak,dodge=True,palette=pos_pal,n_boot=5000,ax=pplot,join=False,capsize=.3)
        sns.swarmplot(x='coef',y='est',hue='phase',data=Peak,palette=pos_pal,
                        ax=pplot,linewidth=1,edgecolor='black',dodge=True)
        x = pplot.get_xticks()[0]
        x = [x-.25, x, x+.25]
        pplot.errorbar(x,avg_peak.est,err_peak.est,
                            ls='none',clip_on=True,color='black')       
        sns.despine(ax=pplot);pplot.legend_.remove()
        pplot.set_ylim(0,1)
        # print(pg.rm_anova(data=Peak,dv='est',within='phase',subject='subject'),pg.pairwise_ttests(data=Peak,dv='est',within='phase',subject='subject'))




#examples of curve fitting

# coef = poly.polyfit(x,y,2)
# ffit = poly.polyval(x,coef)
# plt.plot(x,ffit)

# p = poly.Polynomial.fit(x,y,2,domain=[0,1])
# plt.plot(*p.linspace())

# #the coef we want is p.coef[2] (should be negative)