# Loads manually annotated looking times into dictionary in self.tmes with subject name as key
# Rhodri Cusack 2017-09-28

import os.path
import glob

# Inputs are path and list of filenames in this path, which should be text files formatted like LookingTime1.txt
def load(ltpth,lookingtimefnlist):
    subjflv={}
    tmes={}
    for lookingtimefn in lookingtimefnlist:
        with open(os.path.join(ltpth,lookingtimefn + '.txt'),'r') as fobj:
            for lne in fobj:
                flds=lne.split()
                if len(flds)==1:
                    subj=flds[0].strip()
                    g=glob.glob(os.path.join(ltpth,subj + '*.flv'))
                    if not g:
                        g = glob.glob(os.path.join(ltpth, subj + '*.mp4'))
                    print("New subject "+ subj + " with %d files"%len(g))
                    tmes[subj]=[]
                    if len(g)>0:
                        subjflv[subj]=g[0]
                elif len(flds)>0:
                    if not flds[0]=='Start':
                        tmes[subj].append([float(flds[0]),float(flds[1])])
    return tmes,subjflv





