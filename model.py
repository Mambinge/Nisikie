import pandas as pd
import numpy as np
import hashlib
import seaborn as sb
import matplotlib.pyplot as plt

#Load datasets
directory = 'dataset/'  
train =pd.read_csv(directory+'train.csv') 
test =pd.read_csv(directory+'train.csv')
train.head()

#Description of data
train.info()

#Frequency Distribution
train[ "DEPRESSED"].value_counts() 

train_set = train.replace('Yes', 1).replace('No', 0).replace('Not married',0).replace('Married', 1)

#Explore Gender - Depressed  relationship
gender_pivot = train_set.pivot_table(index="GENDER",values="DEPRESSED")
gender_pivot.plot.bar()

#Explore Education - Depression relationship
education_pivot = train_set.pivot_table(index="EDU_LEVEL",values="DEPRESSED")
education_pivot.plot.bar()

#Explore  numerical Age - Depressed relationship
#compare ages of those who are depressed and those who are not
depressed = train_set[train_set["DEPRESSED"] == 1]
not_depressed     = train_set[train_set["DEPRESSED"] == 0]
depressed["AGE"].plot.hist(alpha=0.5,color='blue',bins=50)
not_depressed["AGE"].plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Depressed','Not_depressed'])
plt.show()

# convert age to a categorical variable
#use pandas.cut() for creating bins
cut_points = [-1,0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior'] 

def process_age(df,cut_points,label_names,attribute):
    df[attribute] = df[attribute].fillna(-0.5)
    df["AGE_categories"] = pd.cut(df[attribute],cut_points,labels=label_names)
    return df

train_set = process_age(train_set,cut_points,label_names,'AGE')
test_set = process_age(test_set,cut_points,label_names,'AGE')

age_cat_pivot = train_set.pivot_table(index="AGE_categories",values="DEPRESSED")
age_cat_pivot.plot.bar()
plt.show()


#Checking on how much each attribute correlates with depression
corr_matrix = train_set.corr()
corr_matrix["DEPRESSED"].sort_values(ascending=False)


test_sets = test_set.replace('Male', 0).replace('Female', 1).replace('Student', 1).replace('Employeed', 2).replace('Unemployeed', 3).replace('Self-employeed', 4)


train_data =train_sets.fillna(0)


from sklearn.model_selection import train_test_split

columns = ['GENDER', 'SAVINGS', 'DEBT', 'SALARY_SUSTAIN','ENJOY_JOB','BORROW_MONEY','FAMILY_DEPRESSED',
         'MARITAL_STATUS', 'AGE', 'OCCUPATION', 'CHILDREN', 'HOUSEHOLD_SIZE', 'NO_CARE',
          'BE_ALONE','FAILURE_FEELING', 'FAMILY_DEPRESSED']



all_X = train_data[columns]
all_y = train_data['DEPRESSED']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.2,random_state=0)
train_X.shape