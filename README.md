# Ex-06-Feature-Transformation

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

outputs
![image](https://user-images.githubusercontent.com/86044259/197975475-17334759-22f1-48cc-af7b-c20280e4f8d7.png)
![image](https://user-images.githubusercontent.com/86044259/197975630-3360825a-641a-44c4-8abf-94a44a96b2d2.png)
![image](https://user-images.githubusercontent.com/86044259/197975905-30b7a10e-2b01-48c5-8bc2-5d5b0ce26454.png)
![image](https://user-images.githubusercontent.com/86044259/197976033-1d24f7db-9f7b-4637-ac5b-ddf786932https://user-images.githubusercontent.com/103166779/197210302-f7e8af9a-8885-48aa-a652-5bfc2cbbfa9c.png821.png)
https://user-images.githubusercontent.com/103166779/197211906-71b64a7a-cdc7-4c23-8530-f2bb0387c34b.png
https://user-images.githubusercontent.com/103166779/197210503-89a66735-6965-471a-94a2-aa4f7b59f132.png
https://user-images.githubusercontent.com/103166779/197211065-baf2f24c-bf69-46df-867d-460ffb6139a3.png
https://user-images.githubusercontent.com/103166779/197211906-71b64a7a-cdc7-4c23-8530-f2bb0387c34b.png
https://user-images.githubusercontent.com/103166779/197212170-a670cee1-025e-4649-9584-88a7e4e28af5.png
https://user-images.githubusercontent.com/103166779/197212326-63f76916-efc5-4371-abce-91af068bbc48.png
![image](https://user-images.githubusercontent.com/86044259/197976399-8944e986-47e2-4adb-9f56-6564f1dce3f0.png)
https://user-images.githubusercontent.com/103166779/197210302-f7e8af9a-8885-48aa-a652-5bfc2cbbfa9c.png
![image](https://user-images.githubusercontent.com/86044259/197976531-53f1c307-c5a9-4be1-8d80-a388bdd29fd1.png)
![image](https://user-images.githubusercontent.com/86044259/197976661-9031d1b3-e816-45d6-972f-1411c1baa5e3.png)

