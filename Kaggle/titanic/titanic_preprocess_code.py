import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df_og = pd.read_csv(r'test.csv')
titanic_df = titanic_df_og.copy()

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')


titanic_df = titanic_df.drop(['PassengerId','Cabin','Name'],axis=1)

le = LabelEncoder()
titanic_df['Sex'] = le.fit_transform(titanic_df['Sex'])

onehot = OneHotEncoder()

encode_array = onehot.fit_transform(titanic_df[['Embarked']]).toarray()

Embarked_df = pd.DataFrame(data = encode_array,columns = onehot.get_feature_names_out(['Embarked']))

titanic_df = pd.concat([titanic_df,Embarked_df],axis=1)
titanic_df = titanic_df.drop('Embarked',axis=1)

titanic_df['Ticket_numeric_1'] = pd.to_numeric(titanic_df['Ticket'],errors='coerce')
titanic_df['Ticket_2_str'] = titanic_df['Ticket'].where(titanic_df['Ticket_numeric_1'].isna())
titanic_df = titanic_df.drop('Ticket',axis=1) # droping Ticket column
titanic_df['ticket_numeric >=300000'] = titanic_df['Ticket_numeric_1'].where(titanic_df['Ticket_numeric_1']>=300000)
titanic_df['ticket_numeric < 300000'] = titanic_df['Ticket_numeric_1'].where(titanic_df['Ticket_numeric_1']<300000)
titanic_df=titanic_df.drop('Ticket_numeric_1',axis=1)

titanic_df['Ticket_str_new_1'] = titanic_df['Ticket_2_str'].where(titanic_df['Ticket_2_str'].map(titanic_df['Ticket_2_str'].value_counts())==1)
titanic_df['Ticket_str_new_2'] = titanic_df['Ticket_2_str'].where(titanic_df['Ticket_2_str'].map(titanic_df['Ticket_2_str'].value_counts())==2)
titanic_df['Ticket_str_new_3'] = titanic_df['Ticket_2_str'].where(titanic_df['Ticket_2_str'].map(titanic_df['Ticket_2_str'].value_counts())>2)

titanic_df = titanic_df.drop('Ticket_2_str',axis=1)

titanic_df['ticket_numeric >=300000']= titanic_df['ticket_numeric >=300000'].replace(titanic_df['ticket_numeric >=300000'].value_counts().index,1)
titanic_df['ticket_numeric < 300000']= titanic_df['ticket_numeric < 300000'].replace(titanic_df['ticket_numeric < 300000'].value_counts().index,1)
titanic_df['Ticket_str_new_1']= titanic_df['Ticket_str_new_1'].replace(titanic_df['Ticket_str_new_1'].value_counts().index,1)
titanic_df['Ticket_str_new_2']= titanic_df['Ticket_str_new_2'].replace(titanic_df['Ticket_str_new_2'].value_counts().index,1)
titanic_df['Ticket_str_new_3']= titanic_df['Ticket_str_new_3'].replace(titanic_df['Ticket_str_new_3'].value_counts().index,1)


titanic_df = titanic_df.fillna(0)


encoder2 = OneHotEncoder()
encoded = encoder2.fit_transform(titanic_df[['Pclass']]).toarray()
Pclass_df = pd.DataFrame(encoded,columns=encoder2.get_feature_names_out())

titanic_df = pd.concat([titanic_df,Pclass_df],axis=1)
titanic_df = titanic_df.drop('Pclass',axis=1)

titanic_df.to_csv('test_preprocessed.csv')
