import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import dataframe_image as dfi

df = pd.read_csv("po1_data.txt", delimiter=",",
                 names=['Subject Identifier', 'Jitter %', 'Jitter ms', 'Jitter rap',
                        'Jitter ppq5', 'Jitter ddp', 'Shimmer %', 'Shimmer dB', 'Shimmer apq3', 'Shimmer apq5',
                        'Shimmer apq11', 'Shimmer dda', 'Harmonicity Corelation', 'NHR', 'HNR',
                        'Median Pitch', 'Mean Pitch', 'Std Dev of Pitch', 'Min Pitch', 'Max Pitch',
                        'Number of Pulses', 'Number of Periods', 'Mean Period', 'Std Dev of Period',
                        'Fraction of Unvoiced Frames', 'Number of Voice Breaks', 'Degree of Voice Breaks',
                        'UPDRS', 'PD_Indicator'])
df.fillna(df.mean(numeric_only = True).round(1), inplace=True)

df1_pd = df[df["PD_Indicator"] == 1]
df2_npd = df[df["PD_Indicator"] == 0]

Null_Hypo =[]
P_Values = []
T_Stat =[]

for column in df.columns[1:-1]: 
    sample_pd = df1_pd[column].to_numpy()
    sample_npd = df2_npd[column].to_numpy()
 
    mean_pd = st.tmean(sample_pd)
    std_pd = st.tstd(sample_pd)
    n_pd = len(sample_pd)

    mean_npd = st.tmean(sample_npd)
    std_npd = st.tstd(sample_npd)
    n_npd = len(sample_npd)

    t,p = st.ttest_ind_from_stats(mean_pd,std_pd,n_pd, 
                                  mean_npd,std_npd, 
                                  n_npd, equal_var= True, 
                                  alternative = "two-sided")
    P_Values.append(p)
    T_Stat.append(t)
    
    print(f"Feature: {column}")
    print(f"\t p-value: {p:.4f}")
    print(f"\t t-value: {t:.4f}")

    if p < 0.05:    
        print("\t Null Hypothesis Rejected with P value")
        Null_Hypo.append('reject')
    else:
        print("\t Null Hypothesis Accepted with P value")
        Null_Hypo.append('accept')
    print("\n")

print(P_Values)

dict = {'Variables':df.columns[1:-1],
        'T-Stat':T_Stat,
        'P-Value':P_Values,
        'Null_Hypothesis': Null_Hypo}

df_final = pd.DataFrame(dict)
print(df_final)
dfi.export(df_final, 'dataframe.png')

x = np.arange(len(df.columns[1:-1]))
width = 0.20

fig,ax = plt.subplots()
ax.set_ylabel('p values')
ax.set_xlabel('Measuremnt Categories')
ax.set_ylim(ymin=.05)
ax.set_title('P value testing')
reacts = ax.bar(x,P_Values,width)
ax.set_xticks(x)
ax.set_xticklabels(df.columns[1:-1],rotation=90)
ax.bar_label(reacts,padding=5,rotation=90)
plt.tight_layout()
plt.show()