#%% [markdown]
# # Introduction
#%% [markdown]
# This notebook will look into long-term unemployment through a small analysis. First the data is found on Statistics Denmark, accessed by an API, cleaned and structured and afterwards analyzed briefly.
# First we import the packages that will be used in this notebook:

#%%
# import relevant packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pandas as pd
import pydst
dst = pydst.Dst(lang = 'en')

#%% [markdown]
# # Find data on Statistics Denmark
#%% [markdown]
# Statistics Denmark contains a lot of tables. Therefore it was easier to go to [statisikbanken.dk (bank of statistics)](http://statistikbanken.dk/statbank5a/default.asp?w=1366) and surf around to find some interesting tables. We ended up with the table **AULK10** that contains unemployment after duration and age. In this notebook we will focus on long-term unemployment which is specied later in the code.
# 
# Now we have considered what table to look at, we need some lines of code to decide what variables we are interested in.

#%%
# see the name of the variables
aulk10 = dst.get_variables(table_id = 'AULK10')
print(aulk10)

# explore the values for 'type of benefits' further to pick only one of them
aulk10.loc[1]['values']

#%% [markdown]
# The table consists of 5 variables: duration, unit, type of benefits, age and time. We choose to explore duration (grouped), age (grouped) and time. Because we don't want to explore different types of benefits we looked into the values for that variable and will in the next lines of code choose id=1: **persons (number).**

#%%
# get the data from dst via an API and save it as 'unemp' 
unemp = dst.get_data(table_id = 'AULK10', variables = {'KMDR': ['*'], 'ENHED': ['1'], 'ALDER': ['*'], 'TID': ['*']})

# see a snip of the data
print(unemp.head(10))

# see the shape of the data
print(f'unemp has shape {unemp.shape}')

#%% [markdown]
# # Cleaning and structuring of data
#%% [markdown]
# The dataset needs to be cleaned and restructured before we can do analysis on it. First we dopt the columns that are the same for all observations. After each cleaning step we look at a snip and the shape of the dataset to see what happened.

#%%
# drop the columns 'enhed' and 'ydelsestype' as they are the same for all observations
unemp.drop(['ENHED','YDELSESTYPE'], axis = 1, inplace = True)

# status after this
print(unemp.head(10))
print(f'unemp has shape {unemp.shape}')

#%% [markdown]
# This step left the number of rows unchanged but decreased the number of columns from 6 to 4. Now we want to change the names of the variables (columns).

#%%
# change the name of the columns by a dictionary
columns_dict = {}
columns_dict['KMDR'] = 'duration' # in weeks
columns_dict['ALDER'] = 'age' # age groups and total
columns_dict['TID'] = 'month'
columns_dict['INDHOLD'] = 'n_persons' # number of persons

unemp.rename(columns = columns_dict, inplace = True)
print(unemp.head(10))

#%% [markdown]
# The shape is unchanged and therefore not shown. Next up is to delete missings and change the datatypes. As the code below shows all of the variables are loaded as objects (strings) and we need to make them more appropriate to do the analyses and the last cleaning.
# In this table Statistics Denmark named some observations '..' which is missings or too insecure to be shown. These are deleted.

#%%
# a. drop rows where n_persons='..' (missings)
unemp = unemp[unemp.n_persons != '..'] 

# b. see the initial dataypes
print(unemp.dtypes)

# c. change type of n_persons and duration to integers 
unemp.n_persons = unemp.n_persons.astype(str).str[:-2] # isolate to the whole number of persons (drop ',0')
unemp.n_persons = unemp.n_persons.astype(int) # convert n_persons to integer
unemp.duration = unemp.duration.astype(str).str[:-6] # isolate the actual duration (drop ' weeks') 
unemp.duration = unemp.duration.astype(int) # convert to an integer

# d. create date from month
unemp['d'] = (unemp.month.astype(str).str[:4] # grab the year (first four digits)
          + '/' # add /
          + unemp.month.astype(str).str[-2:]) # grab the month (last two digits)
unemp['date'] = pd.to_datetime(unemp.d, format = '%Y/%m')
unemp.drop(['month', 'd'], axis = 1, inplace = True) # drop month and intermediate variable, d 

# e. check new types and shape
print(unemp.dtypes)
print(unemp.head(5))
print(f'unemp has shape {unemp.shape}')

#%% [markdown]
# The cleaning of missing values reduces the number of rows. In the documentation of the table Statistic Denmark defines long-term unemployment as a duration of 52 weeks or more. We will only consider the long-term unemployment.

#%%
# restrict the dataset to duration >= 52
unemp = unemp.loc[unemp['duration'] >= 52]

# see a snippet and the shape of the data
print(unemp.head(5))
print(f'unemp has shape {unemp.shape}')

#%% [markdown]
# # Analysis
#%% [markdown]
# ## Descriptive statics: Table and static figure with mean
#%% [markdown]
# Now the data is structured and cleaned and we can begin our brief analysis. First we group the data both after age and duration. The descriptive statistic is then the **variation over time in the number of persons for each age-group and duration-group.**
# 
# The count is the same for each duration which shows that we have the same number of month-observations for each duration across age-groups. Furthermore we see that each duration have different number of month-observations - this will affect one of the coming graphs.
# 
# When looking at the mean we see that the group of 30-49 old is the largest group of long-term unemployed persons and the 16-29 old are the smallest group for all durations. Notice that 'Age, total' is a sum of the 3 age-groups for each duration.
# 
# The difference in level make it hard to compare the different age groups. A solution to that problem is to calculate index and look at the development over the months.

#%%
# descriptive statics grouped by the age-groups and duration describes the data across time
desc = unemp.groupby(['age', 'duration']).describe()
pd.options.display.float_format = '{:.0f}'.format # shows only one decimal
print(desc)

#%% [markdown]
# But before we will compare the age-groups further we will dive more into the **mean number of persons of the 'Age, total'-group divded in different durations in Figure 1 below.**
# 
# Not surprising the higher duration the lower number of people are unemployed. From the graph (and the describtive table above) we see that the number of persons (on average across months) drops by more than 10,000 when looking at a duration of 78 weeks (1.5 year) relative to 52 weeks (1 year).

#%%
# a. figure with mean of each duration (across months) for total age
mean = unemp.groupby(['age', 'duration']).mean()
mean = mean.loc['Age, total'] # only look at total age

# b. plot for total age
def static_figure():
    
    # i. docstring
    """ Makes a static figure of the mean number of persons of the 'Age, total'-group.
    
    Args: None  
            
    """
    
    # ii. make the plot
    ax = mean['n_persons'].plot(legend = True)
    
    # iii. set thousands separator on y-axis
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # iv. set x-axis to the number of weeks according to duration
    ax.set_xticks([52, 78, 104, 130, 156])
    
    # v. set labels
    ax.set_ylabel('number of persons')
    ax.set_xlabel('duration in weeks')
    
    # vi. set title
    ax.set_title('Figure 1')

# c. call the function
static_figure()

#%% [markdown]
# ## Descriptive statics: Interactive figure with indices
#%% [markdown]
# Now we dive more into the comparison between the age-groups. This is done with indices which is calculated by 
# 
# \\[
# \begin{eqnarray*}
# index & = & \frac{value_t}{value_0}*100
# \end{eqnarray*}\\]
# 
# where 0 defines the basis-month. Here we will use the first available month as basis.

#%%
# a. make copy
unemp2 = unemp.copy()

# b. sort
unemp2.sort_values(by = ['date', 'duration'], inplace = True)
unemp2.reset_index(inplace = True)
unemp2.drop(['index'], axis = 1, inplace = True) #delete the old index

# c. select the first element in a series
def first(x): 
    return x.iloc[0]

# d. group the data and calcualte the index
grouped = unemp2.groupby(['duration', 'age'])
unemp2['cal_index'] = grouped['n_persons'].transform(lambda x: x/first(x)*100)

# e. set index to the figure (run only once!)
unemp2.set_index('date', inplace = True)

# f. check the dataset
print(unemp2.head(10))

#%% [markdown]
# In the next lines of code we make an **interactive figure with the calculated indices over time where you can choose which duration you want to look at.**

#%%
# a. plot that is used to the interactive figure
def interactive_figure(x = 52):
    
    # i. docstring
    """ Makes an interactive figure of the calcualtes indices over time.
    
    Args: x : int
        Duration.
            
    """
    
    # ii. make the plot
    I = unemp2.duration == x # plots only if duration = x (argument)
    unemp2[I].groupby(['age'])['cal_index'].plot(legend = True) 
    
    # iii. set labels
    plt.xlabel('Year')
    start_index = unemp2.loc[(unemp2['cal_index'] == 100.0) & (unemp2['duration'] == x)].index # find the basis-month
    start_index2 = min(start_index) # as there is a month for each age-group we just select one of them (they are the same)
    plt.ylabel(f'Index (100 = {start_index2.date()})') # an interactive y-label that shows the basis-month
    
    # iv. set title
    plt.title('Figure 2')

# b. print values for duration to use in the dropdown
unemp2.duration.unique()

# c. dropdown with durations
widgets.interact(interactive_figure,
    x = widgets.Dropdown(description = '$duration$', options = [52, 78, 104, 130, 156])
)

#%% [markdown]
# Figure 2 shows the development in long-term unemployment across age-groups and for different durations. It is seen e.g. for a duration of 104 weeks that the unemployment rises a lot from 2010-2012. From 2013 it starts to decrease and the group of 50+ years decreases fastest. Despite this the group of 16-29 year olds ends at the lowest level, even lower than the level in the basis-month. Notice that the basis-month is showed on the y-axis for each duration.

