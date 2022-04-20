import pandas as pd 

#load data
df = pd.read_csv("RawData_sample.csv")
df = pd.DataFrame(df)


import matplotlib.pyplot as plt
 
'''
The throughput draws curve shape plots and becomes saturated when JIT1 is greater than 25 on certain WAC_time, Process_Time. 
The height of S shape is decreasing when higher WAC_time, Process_Time is used. 
However, in all four plots, throughput is saturated after JIT1 becomes 25. 
This means that the cluster tool scheduler can effectively hide the transport time of robot arms when JIT1 is at least 25 seconds. 
JIT1 value is the look-ahead time means determines when the wafer start to load from the port to the PM.
'''
a_df = df.query('Process_Time ==120 and WAC_Time ==90') 
a_Throughput = a_df['Throughput']
JIT1 = a_df['JIT1']
 
b_df = df.query('Process_Time ==150 and WAC_Time ==90')
b_Throughput = b_df['Throughput']
 
c_df = df.query('Process_Time ==180 and WAC_Time ==90')
c_Throughput = c_df['Throughput']
 
fig = plt.figure(figsize=(9,9)) ## create canvas
fig.set_facecolor('white') ## canvas color
ax = fig.add_subplot() ## create canvas frame
 
ax.plot(JIT1,a_Throughput,marker='o',label='ProcessTime=120') ##draws line graph which has certain process time
ax.plot(JIT1,b_Throughput,marker='o',label='ProcessTime=150') ##draws line graph which has certain process time
ax.plot(JIT1,c_Throughput,marker='o',label='ProcessTime=180') ##draws line graph which has certain process time
 
ax.legend() ## setting of legend 
 
plt.title('Throughput against JIT1 with different ProcessTime',fontsize=15) ## setting of title
plt.xlabel("JIT1",fontsize=10)
plt.ylabel("Throughput",fontsize=10)
plt.show()

