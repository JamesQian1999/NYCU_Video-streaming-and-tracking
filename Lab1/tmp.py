import pandas as pd

val = pd.read_csv('HW1/HW1_311551096.csv')
pre = pd.read_csv('HW1_311551096.csv')

print(val)
print(pre)

corr = 0.0
for i in range(len(pre)):
    if val.iloc[val.index[val['names']==pre.iloc[i][0]].item()][1] == pre.iloc[i][1] :
        corr += 1
    
print(f'Acc. {corr/len(pre)*100:.5f}%')