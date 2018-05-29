
# coding: utf-8

# In[3]:


# Functions needed for the pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import RandomForestClassifier
import graphviz 
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab as pl


# ### 1. Read data

# In[6]:


### 1 READ THE DATA

def create_df (file):
    '''Creates a pandas data frame using as input 
    a .csv file
    Inputs:
    file (string): Path to file
    Returns:
    Dataframe
    '''
    return pd.read_csv(file)


# ### 2. Explore the data

# In[7]:


def scatterplot(df, var_y,var_x):
    '''
    Creates a scatterplot using a dataframe df,
    and the features var_y and var_x
    Inputs:
    df-> Dataframe
    var_y (string)-> y variable
    var_x (string) -> x variable
    '''
    plt.scatter(df[var_y], df[var_x])
    plt.title(var_y +' '+ 'vs'+ ' '+ var_x)
    plt.xlabel(var_x) 
    plt.ylabel(var_y)
    plt.show()


# In[8]:


def plot_hist(df, variable, num_bins):
    '''
    Creates a histogram plot using a dataframe df for
    a variable of interest
    
    Inputs:
    df-> Dataframe
    variable -> variable of interest
    num_bins -> number of bins
    '''
    plt.hist(df[variable].dropna(), bins=num_bins)
    plt.title('Distribution of'+ ' '+  variable)
    plt.ylabel(variable)
    plt.xlabel('Frequency')
    plt.show()


# In[9]:


def corr_pair(df, variable):
    '''
    Creates a dataframe with the correlation values of
    the variable and the rest of the variables in a dataframe
    
    Inputs:
    df-> dataframe
    variable: the variable of interest which the correlation coefficients
    are calculated with
    
    Returns: 
    Dataframe
    '''
    dic = {}
    for element in list(set(list(df)) - set(list(variable))):
        dic[element] = df[variable].corr(data[element])
    a = pd.DataFrame(list(dic.items()), columns=['variable', variable])
    a.set_index('variable', inplace= True)
    return a


# In[10]:


def count_nulls(df):
    '''
    Counts the number of missing values per variable in a dataframe
    Returns: Dataframe
    '''
    dic = {}
    for element in list(df):
        dic[element] = sum(pd.isnull(data[element]))
    a = pd.DataFrame(list(dic.items()), columns=['variable', 'Missings'])
    a.set_index('variable', inplace= True)
    return a


# ### Pre-Processing

# In[11]:


#Taken from DataGotham
#To treat outliers
def cap_values(x, cap):
    '''
    if a value exceeds the threshold cap, it returns
    the value cap, otherwise the same value'''
    if x > cap:
        return cap
    else:
        return x


# In[12]:

'''
#Imputing nulls. Modify this code
df=df
is_test = np.random.uniform(0, 1, len(df)) > 0.75
train = data[is_test==False]
test = data[is_test==True]
from sklearn.neighbors import KNeighborsRegressor

income_imputer = KNeighborsRegressor(n_neighbors=3)

#split our data into 2 groups; data containing nulls and data 
# not containing nulls we'll train on the latter and make
# 'predictions' on the null data to impute monthly_income
train_w_monthly_income = train[train.MonthlyIncome.isnull()==False]
train_w_null_monthly_income = train[train.MonthlyIncome.isnull()==True]
cols = ['col1', 'col2']
income_imputer.fit(train_w_monthly_income[cols], train_w_monthly_income.MonthlyIncome)
new_values = income_imputer.predict(train_w_null_monthly_income[cols])
train_w_null_monthly_income['monthly_income'] = new_values
#combine the data back together
train = train_w_monthly_income.append(train_w_null_monthly_income)
train.head()

# I NEED TO REVIEW HOW TO IMPLEMENT THE FINAL IMPUTATION HERE
data_with_null = data[data.MonthlyIncome.isnull() == True]
data_no_null = data[data.MonthlyIncome.isnull() == False]
cols = ['NumberRealEstateLoansOrLines', 'NumberOfOpenCreditLinesAndLoans']
new_values = income_imputer.predict(data_with_null[cols])
data_with_null['monthly_income'] = new_values

dfs = [data_no_null , data_with_null]
data_imputed = pd.concat(dfs)
'''

# In[ ]:


def fill_na_mean(df, variable):
    '''Replaces missing values with the
    mean value of the variable
    Inputs:
    df-> Dataframe
    variable-> variable where NaN are to be replaced
    Returns:
    Panda Series
    '''
   
    return df[variable].fillna(df[variable].mean(), inplace=True)



# In[ ]:


def cap_outliers(x):
    '''Assigns the median value of an upward outlier
    Inputs: Dataframe
    Returns: Dataframe
    
    i.e. data['DebtRatio_wo_ol'] = cap_outliers(data['DebtRatio'])
    '''
       
    mask = (x >= x.quantile(.95))
    x[mask] = x.median()
    return x


# In[4]:


def impute(df, variable, statistic):
    '''Imputes values to a specified variable
    Inputs:
        df: dataframe
        variable: variable which will be imputed
        statistic: valid values: mean, median or zero
    Returns:
        dataframe without nulls
    '''
    if statistic == 'mean':
        df[variable].fillna(df[variable].mean(), inplace=True)
    elif statistic == 'median':
        df[variable].fillna(df[variable].median(), inplace=True)  
    elif statistic == 0:
        df[variable] = np.where(df[variable].isnull(), statistic,
                                  df[variable])
    
    


# ### 4 Generate and Select Features

# In[ ]:


def cut_buckets(df, variable, buckets):
    '''
    Creates a df which with the variable cut into buckets
    Inputs:
        df-> Dataframe where the new variable is added
        variable -> variable to be cut into bukets
        buckets-> List of values in which the variable will be cut into
    Returns:
        Dataframe
    '''
    df = pd.cut(df[variable], buckets, labels = (list(range(len(buckets))))[1:])
    return df 


# In[ ]:


def cut_in_pct_bins(df, variable, quantiles):
    '''
    Creates bins for the variable and dataframe specified
    Inputs:
        df -> Data Frame
        variable (String)-> Variable of interest
        quantiles (list) -> List of cutoffs for quantiles
    Returns:
        Series Dataframe
    '''
    bins = []
    for q in quantiles:
        bins.append(df[variable].quantile(q))
    labels = []
    for element in (list(range(len(quantiles))))[1:]:
        label = element
        labels.append(label)
        
    return pd.cut(df[variable], bins=bins, include_lowest = True, labels = labels)



# In[ ]:


def create_dummy(df, variable, threshold, relation):
    ''' 
    Creates a series dataframe with dummy variables equals to one
    if the variable is equal, less or greater than a threshold.
    Inputs:
        df: dataframe
        variable (string): variable to create the dummy variable
        threshold (int): value that makes the relation true in order to create the dummy
        relation (string): can take three values: 'greater', 'equal', or 'less'.
    Returns:
        Series Dataframe
    '''   
    if relation == 'greater':
        return df[variable].apply(lambda x: 1 if x > threshold else 0)
    elif relation == 'equal':
        return df[variable].apply(lambda x: 1 if x == threshold else 0)
    elif relation == 'less':
        return df[variable].apply(lambda x: 1 if x < threshold else 0)


# In[ ]:


def generate_best_features(df,features, dep_var):
    '''
    Creates a barplot to show the better predictors
    Inputs:
        df: dataframe
        feautures (list): features that will be evaluated
        dep_var (string): variable of interest
    '''
    features = np.array(features)
    clf = RandomForestClassifier()
    clf.fit(df[features], df[dep_var])
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")
    pl.show()


# ### 5 Build and Evaluating Classifier

# In[ ]:


def generate_train_and_test(df, predictors, var_to_predict, test_size, random_state):
    '''Creates train and test dataframes to run a model
    Inputs:
        df: Dataframe
        predictors(list): list of predictor variables(string)
        test size (float): share of the data to construct the training and test dfs
    Returns:
        Tuple with 'x_train', 'x_test', 'y_train', and 'y_test'
        '''
    X = df.filter(items = predictors)
    Y = df[var_to_predict]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = random_state)
    return x_train, x_test, y_train, y_test


# In[ ]:


def evaluate_tree(x_train, x_test, y_train, y_test, depths, criterion, 
                  max_features, splitter, min_sample_l, min_sample_spl, max_leaf_n):
    '''
    Creates a dataframe with the different models' accuracy predictions, and returns
    a dataframe with all these models, sorted by degree of accuracy in the test data
    Inputs:
        x_train: dataframe with the features that predict the outcome of interest. Used for 
            training the model
        x_test: dataframe with the features that predict the outcome of interest. Used for testing
            the model
        y_train: variable to predict. Used to train the model
        y_test: variable to predict. Used to test the model
        depths (list of integers): maximum depth of the tree
        criterion (list): Measure of purity of subgroups. 'Entropy' or 'Gini'
        max_features (list): maximum number of features to include in the model
        splitter (list): Method to split samples. 'Best' or 'Random'
        min_sample_l(list): min number of samples required to be at a leaf node.
        min_samples_split: minimum number of samples required to split an internal node 
        max_leaf_nodes (list): max number of leaf nodes a tree can have
    '''
    for d in depths:
        for criteria in criterion:
            for feature in max_features:
                for split in splitter:
                    for min_leave in min_sample_l:
                        for min_split in min_sample_spl:
                            for max_leaf in max_leaf_n:
                                dec_tree = DecisionTreeClassifier(max_depth= d, criterion = criteria, splitter = split,
                                max_features = feature, min_samples_leaf = min_leave,  min_samples_split = min_split )
                                dec_tree.fit(x_train, y_train)
                                train_pred = dec_tree.predict(x_train)
                                test_pred = dec_tree.predict(x_test)
                                # evaluate accuracy
                                train_acc = accuracy(train_pred, y_train)
                                test_acc = accuracy(test_pred, y_test)
                                result = d, criteria, feature, split, min_leave, min_split, max_leaf, train_acc, test_acc
                                results.append(result)        
    evaluation_df = pd.DataFrame(results,columns = ['Max_depth', 'Criteria', 'Number_Features', 'Split',                                                 'Min_Sample_leave','Min_Sample_split','Max_leaf_nodes', 'Train_Acc', 'Test_Acc'])
    return evaluation_df.sort_values(by=['Test_Acc'], ascending = False)


# EVALUATION USING CONFUSION MATRIX

# FUNCTION TO VISUALIZE THE MODEL
