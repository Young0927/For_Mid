import pandas as pd 

#load data
df = pd.read_csv("RawData_sample.csv")
df = pd.DataFrame(df)


import matplotlib.pyplot as plt
 
## 3개 데이터 분리
a_df = df.query('Process_Time ==120 and WAC_Time ==90') 
a_Throughput = a_df['Throughput']
JIT1 = a_df['JIT1']
 
b_df = df.query('Process_Time ==150 and WAC_Time ==90')
b_Throughput = b_df['Throughput']
 
c_df = df.query('Process_Time ==180 and WAC_Time ==90')
c_Throughput = c_df['Throughput']
 
fig = plt.figure(figsize=(9,9)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성
 
ax.plot(JIT1,a_Throughput,marker='o',label='ProcessTime=120') ## 선그래프 생성
ax.plot(JIT1,b_Throughput,marker='o',label='ProcessTime=150') 
ax.plot(JIT1,c_Throughput,marker='o',label='ProcessTime=180') 
 
ax.legend() ## 범례
 
plt.title('Throughput against JIT1 with different ProcessTime',fontsize=15) ## 타이틀 설정
plt.xlabel("JIT1",fontsize=10)
plt.ylabel("Throughput",fontsize=10)
plt.show()

