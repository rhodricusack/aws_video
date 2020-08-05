import io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from scipy import stats

filepath = '/imaging/rcusack/Dropbox/sharedwithme/Rhodri_Brea/data_14Nov2017_30.txt'

'''
Reads in manual ratings of frame features and produces summary graphs.
Then relates these to automatic performance.

CODES
Size of iris    Zone14
Colour of iris  Zone15
Eye shape       Zone16
Face in frame   Zone17
Amount          slider
Type            Zone1
Where Horiz     Zone9
Where Vert      Zone 10
Resolution      Zone 20
Other           
'''
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def get_key(fname):
    fname = fname[:-17]
    # strip off extra experiment name at start, as these are not in the autocoding files
    pos = fname.find('_')
    return fname[pos + 1:]

def get_fnum(fname):
    return fname[-10:-4]

if __name__=='__main__':
    # For figures and summaries
    outpth = '/imaging/rcusack/Dropbox/python/aws_video/figures'
    loadpth='/imaging/rcusack/Dropbox/python/aws_video/figures/allmldf.pickle'

#    loadpth='/imaging/rcusack/Dropbox/python/aws_video/figures_preflooking/allmldf.pickle'
#    outpth = '/imaging/rcusack/Dropbox/python/aws_video/figures_preflooking'

    # Close all existing figures
    plt.close('all')


    toshow=['behavauto'] # behavdist, behavreplicability, behavauto

    # Read in autocoding results
    with open(loadpth,'rb') as f:
        mldf=pickle.load(f)


    # Read in the data file as a dict
    tab = {}
    with io.open(filepath, 'r', encoding='utf-16') as f:
        cols = f.readline().rstrip('\n\r').split('\t')
        for col in cols:
            tab[col] = []
        for ind, lne in enumerate(f):
            flds = lne.rstrip('\n\r').split('\t')
            for ind, col in enumerate(cols):
                tab[col].append(flds[ind])

    #print(tab)

    # Get frame names
    frames = set([x for x in tab['FrameName'] if not len(x) == 0])
    #print(frames)

    # Decode chosen columns and drop into dictionary
    results = {}
    rawresults = {}

    ratings = [['light', 'slider', 'float','Amount of light',11],
               ['horz', 'Zone9','float','Horizontal clockface',12],
               ['vert', 'Zone10','float','Vertical light',4],
               ['resolution','Zone20','float','Resolution',10],
               ['irissize', 'Zone14', 'float', 'Iris size',4],
               ['iriscolour', 'Zone15', 'float', 'Iris colour',4],
               ['eyeshape', 'Zone16', 'float', 'Eye shape',5],
               ['faceinframe', 'Zone17', 'float', 'Face in frame',4],
               ]

    fig_av, axarr=plt.subplots(3,3)
    axarr=axarr.flatten()

    fig_av2, axarr2=plt.subplots(3,3)
    axarr2=axarr2.flatten()
    fig_av2.subplots_adjust(top=0.9)
#    fig_av2.suptitle('Prop one face')

    fig_av3, axarr3=plt.subplots(3,3)
    axarr3=axarr3.flatten()
    fig_av3.subplots_adjust(top=0.9)
#    fig_av3.suptitle('D-prime')

    allbehav=mldf[['dprime','proponeface']]

    with open(os.path.join(outpth,'man_to_auto_pearsonr.txt'),'w') as fout:
        for ind,rating in enumerate(ratings):
            results[rating[0]] = {}
            rawresults[rating[0]] = {}
            rows = [x[0] for x in enumerate(tab['Zone Name']) if x[1] == rating[1]]

            # Dictionary by subject
            for frame in frames:
                k=get_key(frame)
                results[rating[0]][k] = {}
                rawresults[rating[0]][k] = {}
            # Within this dictionary by frame
            for row in rows:
                fname = tab['FrameName'][row]
                fnum=get_fnum(fname)
                fname=get_key(fname)

                if rating[2] == 'float':
                    fld=tab['Response'][row]
                    try:
                        results[rating[0]][fname][fnum] = float(fld)
                        rawresults[rating[0]][fname][fnum] = "%02d"%float(fld)
                    except:
                        rawresults[rating[0]][fname][fnum] = fld

            if "behavreplicability" in toshow:
                # Plot consistency
                r=results[rating[0]]
                dat=[r[key].values() for key in r if len(r[key].values())==2]
                dat=np.asarray(dat)
                n,m=dat.shape
                dat=dat+np.random.randn(n,m)/4
                df=pd.DataFrame()
                dat=zip(*dat) # transpose
                df['Measure 1']=dat[0]
                df['Measure 2']=dat[1]
                g=sns.jointplot('Measure 1','Measure 2', data=df,kind='reg',size=10)
                #g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
                plt.subplots_adjust(top=0.9)
                g.fig.suptitle('Consistency: '+ rating[3])

            if "behavdist" in toshow:
                # All measures from all subjects
                rr=rawresults[rating[0]]
                dat=[rr[key].values() for key in rr]
                dat=[item for sublist in dat for item in sublist]
                dat=np.asarray(dat)
                dato=sorted(set(dat))
                g=sns.countplot(dat,ax=axarr[ind],order=dato)
                axarr[ind].set_title(rating[3])

            if "behavauto" in toshow:
                # Compare autocoding to manual video quality measures
                behav=pd.DataFrame()

                df = pd.DataFrame()
                r=results[rating[0]]
                mldf['stridx']=mldf.index
                for key in r:
                    key2=key
                    if key2[-6:-1]=='child':
                        key2=key2[:-6] + '_' + key2[-6:]
                    mldf_match=mldf['stridx'].str.contains(key2)
                    if mldf_match.sum()==1:
                    #   print('Got %s' % key2)
                        for frame,score in r[key].items():
                            mldfrow=mldf[mldf_match]
                            mldfrow[rating[0]]=pd.Series(score,index=mldfrow.index)
                            behav=behav.append(mldfrow)

                # Average across manual ratings (i.e., one value per subject)
                behav=behav.groupby(level=0).mean()

                # Value specific recoding
                if rating[0]=='irissize' or rating[0]=='iriscolour':
                    behav=behav[behav[rating[0]]!=0]
                if rating[0]=='horz':
                    behav['horz']=behav['horz'].apply(lambda x: 180-30*abs(x-6))

                allbehav[rating[0]]=behav[rating[0]]

                ax=sns.regplot(rating[0],'proponeface',  data=behav,ax=axarr2[ind])

                r_value,p_value=stats.pearsonr(behav[rating[0]], behav['proponeface'])
#                slope, intercept, r_value, p_value, std_err = stats.linregress(behav[rating[0]], behav['proponeface'])
                print('%s and %s, pearson r=%f p<%f'%(rating[3],'proponeface',r_value,p_value),file=fout)
        #        ax.set(xlabel=rating[3])
                ax=sns.regplot(rating[0],'dprime',  data=behav,ax=axarr3[ind])
                r_value, p_value = stats.pearsonr(behav[rating[0]], behav['dprime'])
#                slope, intercept, r_value, p_value, std_err = stats.linregress(behav[rating[0]], behav['dprime'])
                print('%s and %s, pearson r=%f p<%f'%(rating[3], 'dprime', r_value, p_value),file=fout)
        #        ax.set(xlabel=rating[3])

                if rating[0]=='amount':
                    # Compare automatic light as prediction of proponeface
                    fig=plt.figure()
                    sns.regplot(x=behav['QualityBrightness'],y=behav['proponeface'])
                    r_value, p_value = stats.pearsonr(behav['QualityBrightness'], behav['proponeface'])
#                    slope, intercept, r_value, p_value, std_err = stats.linregress(behav['QualityBrightness'], behav['proponeface'])
                    print('Automatic light rating and proponeface pearson r=%f p<%f' % (r_value, p_value),file=fout)
                    fig.savefig(os.path.join(outpth,'auto_brightness_and_proponeface.pdf'),format='pdf')

                    # Compare automatic and manual light rating
                    fig=plt.figure()
                    sns.regplot(x=behav['amount'],y=behav['QualityBrightness'])
                    r_value, p_value = stats.pearsonr(behav['amount'], behav['QualityBrightness'])
                    #slope, intercept, r_value, p_value, std_err = stats.linregress(behav['amount'], behav['QualityBrightness'])
                    print('Manual and automatic lighting rating  pearson r=%f p<%f' % (r_value, p_value),file=fout)
                    fig.savefig(os.path.join(outpth,'man_and_auto_brightness.pdf'),format='pdf')
    # Regression models for all manually-rated quality measures on dprime and proponeface
    #   2018-04-09: Give up because too many missing values
    # X = allbehav[[r[0] for r in ratings]]
    # indvars=['dprime','proponeface']
    # mask = np.isfinite(X).all(axis=1)
    # for indvar in indvars:
    #     y=allbehav[indvar]
    #     X2=X[mask]
    #     y2=y[mask]
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(X[mask], y[mask])
    #     print('Regression for %s  r=%f p<%f' % (indvar,r_value, p_value), file=fout)

    fig_av2.tight_layout()
    fig_av3.tight_layout()

    fig_av2.delaxes(axarr2[8])
    fig_av3.delaxes(axarr3[8])
    plt.show()

    fig_av2.savefig(os.path.join(outpth,'man_with_proponeface.pdf'),format='pdf')
    fig_av3.savefig(os.path.join(outpth,'man_with_dprime.pdf'),format='pdf')


