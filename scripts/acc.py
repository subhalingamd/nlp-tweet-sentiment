import sys

f1 = sys.argv[1]
f2 = sys.argv[2]

with open(f1, "r") as f:
	y1 = f.read().split("\n")[:-1]

with open(f2, "r") as f:
	y2 = f.read().split("\n")[:-1]

c = 0
for i,j in zip(y1,y2):
	if i == j:
		c += 1
print(c/len(y1))