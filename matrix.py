import pandas as pd

l = [45,17,37,36,19,16,14,38,26,26,32,22,39,44,16,32,14,24,24,29,16,29,40,28,32,39,25,30,35,42,12,34,15,16,31,24,43,33,14,27,21,19,23,19,13,33,13,22,34,18,30,17,20,24,29,13,26,16,23,24,23,22,8,20,27,18,28,25,15,22,20,23,22,25,29,29,21,24,33,22,10,16,19,15,19,18,26,14,20,10,18,29,7,2,14,9,18,4,6,9]

c = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
f = ["Ap","Ba","Ch","Av","Kw","Ma","Or", "Pi","St","Wa"]
p = pd.DataFrame(index=c,columns=f)

k=0
for i in range(0,10):
    for j in range(0,10):
        p.iat[i,j] = l[k]
        k+=1

print(p)