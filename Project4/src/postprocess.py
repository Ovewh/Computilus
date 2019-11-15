import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from settings import *
from parser import post_parser
import analytic as a

def plot_relative_error(datafile,T = 1.0,L=2,window=1,ordered=0):
    df = pd.read_csv(datafile)
    cv = a.cv(T)/L**2
    E = a.expected_energy(T)/L**2
    Mabs = a.expected_magnetization(T)/L**2
    suscept = a.susceptibility(T)/L**2


    sel = df.loc[(df['T'] == T) & (df['ordered']==ordered)]
    sel['relcv'] = np.abs(sel['cv'] - cv)/np.abs(cv)
    sel['relE'] = np.abs(sel['E'] - E)/np.abs(E)
    sel['relMabs'] = np.abs(sel['Mabs'] - Mabs)/np.abs(Mabs)
    sel['relsuscept'] = np.abs(sel['suscept'] - suscept)/np.abs(suscept)

    fig, ax = plt.subplots(1)
    ax.tick_params(left=True,right=True,which='both')
    ax.tick_params(labelleft =True, labelright=True)
    cols = ['relE','relcv','relMabs','relsuscept']

    for col in cols:
        sel.rolling(window).mean().plot('cycles',col,linestyle='-',logx=True,logy=True, ax=ax)
    ax.set_ylabel('Relative error')
    ax.set_xlabel('Monte Carlo cycles')
    plt.grid()
    plt.title('ordered: {} T: {}'.format(ordered, T))
    plt.savefig(FIGDIR + 'relative_error_order{}_T{}.png'.format(ordered,int(T)))


def plot_equilibrium(datafile):
    df = pd.read_csv(datafile)

    # Plot equilibrium tests
    df = pd.read_csv(equi)
    df['E'] = np.abs(df['E'])
    df['Mabs'] = np.abs(df['Mabs'])
    ticks = np.arange(-5,5,0.1)
    for col in ['E','Mabs']:
        fig, ax=plt.subplots(2, sharex=False)
        for T in [1.0,2.4]:
            lb = 'T={}'.format(T)
            # Plot for spins up and random
            for axis, ordered in zip(ax, [1,0]):
                sel = df.loc[(df['T']==T) & (df['ordered']==ordered)]
                sel = sel.loc[df['cycles'].between(0,50000)]
                sel.plot('cycles',col, ax=axis,linestyle='-', label=lb)#,logx=True)
                axis.set_ylabel(r'$\vert${}$\vert$ ord={}'.format(col,ordered))
                axis.yaxis.set_major_locator(MultipleLocator(0.1))
                axis.grid(linestyle='--',axis='y')
        plt.savefig(FIGDIR + 'equilibrium_{}.png'.format(col))
        plt.clf()

def plot_distribution(distfile, datafile):
    dist = pd.read_csv(distfile,index_col=0)
    # dist = dist/400.
    data = pd.read_csv(datafile)
    data['T'] = data['T'].round(2)
    data.index = data['T']
    columns = dist.columns[0::3]
    num_bins = int(1+3.3*np.log(len(dist)))
    # num_bins = int((dist.max() - dist.min()).max()/(0.04))
    print(num_bins)
    for T in columns:
        label = 'T={} $\sigma_E^2$={:.2f}'.format(T,data.loc[float(T),'cv']*float(T)**2)
        dist[T].hist(density=True,bins=64,label=label, alpha=0.8)
    plt.legend()
    plt.grid(False)
    plt.ylabel('Probability density')
    plt.xlabel('Energy')
    plt.savefig(FIGDIR + 'distribution.png')
    plt.clf()

def plot_phase(datafile):
    df = pd.read_csv(datafile)
    for col,ylabel in zip(['E','Mabs','cv','suscept'],['Energy','Magnetization','Heat capacity','Susceptibility']):
        fig, ax = plt.subplots(1)
        for L in [40,60,80,100]:
            spins = L**2
            label = 'L={}'.format(L)
            df.loc[df['spins']==spins].plot('T',col, ax=ax,linestyle=' ',markersize=1.5,marker='s',label=label)
        ax.legend()
        ax.set_ylabel(ylabel)
        plt.grid()
        plt.savefig(FIGDIR + 'phase_{}.png'.format(col))
        plt.clf()

def plot_accepted(datafile):
    df = pd.read_csv(datafile)
    df['ratio'] = df['accepted']/(df['spins']*df['cycles'])
    fig, ax = plt.subplots(1)
    # df = df.loc[df['cycles'].between(0,100000)]
    df.loc[(df['T']==1.0) & (df['ordered']==0)].plot('cycles','ratio',ax=ax,label='T=1.0')
    df.loc[(df['T']==2.4) & (df['ordered']==0)].plot('cycles','ratio',ax=ax,label='T=2.4')
    plt.xscale('log')
    # plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.yscale('log')
    plt.ylabel('Accepted / (spins $\cdot$ cycles)')
    plt.savefig(FIGDIR + 'accepted.png')
    plt.clf()


if __name__ == '__main__':
    args = post_parser().parse_args()
    equi = args.equi
    phase = args.phase
    distfile = args.distfile
    datafile = args.distdata

    file = '../data/equi_L2_large_7.csv'
    win = 5
    plot_relative_error(file,T=1.0,window=win)
    plot_relative_error(file,T=1.0,window=win,ordered=1)
    plot_relative_error(file,T=2.4,window=win)
    plot_relative_error(file,T=2.4,window=win,ordered=1)


    plot_equilibrium(equi)
    plot_accepted(equi)
    plot_distribution(distfile, datafile)
    plot_phase(phase)
