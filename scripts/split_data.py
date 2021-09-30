import os
os.chdir("..")

########################################################################


from utils import read_file_and_split
(x1,y1),(x2,y2),(x3,y3) = read_file_and_split("data/training.csv")

with open("data_split/training.csv",'w') as f:
        for x,y in zip(x1,y1):
                print(f'"{x*4}","{y}"',file=f)
with open("data_split/dev.csv",'w') as f:
        print(*x2,sep="\n",file=f)
with open("data_split/dev.gold.csv",'w') as f:
        print(*[y*4 for y in y2],sep="\n",file=f)
with open("data_split/test.csv",'w') as f:
        print(*x3,sep="\n",file=f)
with open("data_split/test.gold.csv",'w') as f:
        print(*[y*4 for y in y3],sep="\n",file=f)
