import glob
import os
import numpy as np

for file in glob.glob("/media/ashvin/data2/s3doodad/ashvin/awacp/finetune/run*/id*/*.txt"):
    if "script_name" in file:
        continue
    print(file)
    x = np.loadtxt(open(file, "rb"))
    print(len(x))

    fname = file[:-file[::-1].index("/")] + "progress.csv"
    np.savetxt(fname,
       x,
       fmt=['%d', '%.1f'],
       delimiter=",", header="expl/num steps total,eval/Average Returns")
