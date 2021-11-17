import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv('real_dataset.csv')

tear_rate = dataset['tear_rate_name'][2::]
lens = dataset['lens_name'][2::]

stacked = np.hstack((tear_rate, lens))
stacked = stacked.reshape(2, 7)

tear_type = np.unique(tear_rate)
lens_type = np.unique(lens)

#stacked = pd.DataFrame(stacked)


print(dataset)
print(dataset['tear_rate_name'][2::].unique())
#print(dataset.groupby('tear_rate_name').size())


#sns.countplot(fruits['fruit_name'],label="Count")
#plt.show()