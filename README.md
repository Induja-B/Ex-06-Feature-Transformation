# Ex-06-Feature-Transformation
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()
~~~

outputs
![image](https://user-images.githubusercontent.com/86044259/197975475-17334759-22f1-48cc-af7b-c20280e4f8d7.png)
![image](https://user-images.githubusercontent.com/86044259/197975630-3360825a-641a-44c4-8abf-94a44a96b2d2.png)
![image](https://user-images.githubusercontent.com/86044259/197975905-30b7a10e-2b01-48c5-8bc2-5d5b0ce26454.png)
![image](https://user-images.githubusercontent.com/86044259/197977483-eb6781d8-9729-4662-8c0b-ec8a0d1082db.png)
![image](https://user-images.githubusercontent.com/86044259/197977603-bde6180b-10a8-42b5-bc69-3a7327f33c6a.png)
![image](https://user-images.githubusercontent.com/86044259/197977678-e7ecf95e-d7a0-470a-988f-66e330232cbb.png)
![image](https://user-images.githubusercontent.com/86044259/197977703-8c7439b7-60b8-4f81-8a3d-cafbf382a6de.png)
![image](https://user-images.githubusercontent.com/86044259/197977810-a9925980-2377-47e6-88f9-07caf14243b9.png)
![image](https://user-images.githubusercontent.com/86044259/197977858-c0bda3bb-7e46-4769-8e1b-2f79b653c291.png)
![image](https://user-images.githubusercontent.com/86044259/197977904-cf6d8dea-7f21-4671-943b-a836025e2a6f.png)
![image](https://user-images.githubusercontent.com/86044259/197977971-3f7aae74-2d48-485e-b344-4982dc1cde72.png)
![image](https://user-images.githubusercontent.com/86044259/197978019-92111c40-6305-4856-9d0d-09e0b4f03251.png)
![image](https://user-images.githubusercontent.com/86044259/197978090-e82f83c7-6568-496e-b3fd-937d67548e51.png)
![image](https://user-images.githubusercontent.com/86044259/197978115-c89bdd16-7a5f-4521-bb05-7bd30e2fa6b6.png)






