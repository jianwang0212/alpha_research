import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from IPython.display import display_html
from itertools import chain,cycle
from scipy.stats import kstest, norm
import numpy as np
import statsmodels.api as sm
import pylab as py
import matplotlib.pyplot as plt 

def display_side_by_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)

    
def get_info_df(df):
    df_types = pd.concat([pd.DataFrame(df.dtypes), df.count()], axis=1)
    df_types.columns = ['Dtype', 'non-null count']

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_numerics = df.select_dtypes(include=numerics)
    
    df_numerics_table = df_numerics.describe()
    display_side_by_side(df_types,df_numerics_table, titles=['DataTypes & Missing','Quantiles'])

def get_missing_details(df):
    df_null = df.isnull().sum().to_frame().reset_index()
    df_null.columns = ["columns", "count_of_null"]
    df_null_total = df_null.append(df_null.sum(numeric_only=True).rename('Total'))
    df_null_detail = df[df.isnull().any(axis=1)]
    display_side_by_side(df_null_total,df_null_detail, titles=['Missing Data Summary','Missing Data Details'])

def get_dup_details(df):
    df_dup = df[df.duplicated(keep=False)]
    display_side_by_side(df_dup, titles=['Dup Data Details'])


def get_norm_stats(df):
    p_values = []
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    for (column_name, column) in df.transpose().iterrows():
        p_value = kstest(column, 'norm')[1]
        p_values.append(p_value)
        # print (f"p_value is {p_value}")
    result = pd.DataFrame(data = [p_values], columns = df.columns)
    display_side_by_side(result, titles=['KS normal stats (p-value <5% then not a normal dist)'])
    
def plot_hist(data_frame):
    fig, ax = plt.subplots(figsize = (6,4))

    # Plot
        # Plot histogram
    data_frame.plot(kind = "hist", density = True, alpha = 0.65, bins = 15) # change density to true, because KDE uses density
        # Plot KDE
    data_frame.plot(kind = "kde")
        # Quantile lines
    quant_5, quant_25, quant_50, quant_75, quant_95 = data_frame.quantile(0.05), data_frame.quantile(0.25), data_frame.quantile(0.5), data_frame.quantile(0.75), data_frame.quantile(0.95)
    quants = [[quant_5, 0.6, 0.16], [quant_25, 0.8, 0.26], [quant_50, 1, 0.36],  [quant_75, 0.8, 0.46], [quant_95, 0.6, 0.56]]
    for i in quants:
        ax.axvline(i[0], alpha = i[1], ymax = i[2], linestyle = ":", color = "red")
    plt.title(data_frame.name)
    plt.show()

def plot_qq(data_frame):
    sm.qqplot(data_frame, line ='45')
    plt.title(data_frame.name)
    py.show()




def data_show(input_data):
    input_data.boxplot()
    input_data.hist()
    input_data.plot(subplots=True)
    print(input_data.describe())

def create_lag_df_y(df, lags_value):
    data_input = df.copy()
    df_y = data_input["y"].to_frame()
    
    lags = range(1,lags_value)
    df_y= df_y.assign(**{
        f'{col}_{lag}': df_y[col].shift(lag)
        for lag in lags
        for col in df_y
    })
    df_y["y_sum"] = df_y.sum(axis=1,skipna=False)
    col_name = "y_sum_" + str(lags_value)
    data_input[col_name] =df_y["y_sum"]
    data_input["Y"] = data_input[col_name]
    data_input["X"] = data_input["x"].shift(lags_value)
    data_input = data_input[lags_value:]
    return data_input


def create_lag_df_x(df, lags_value):
    data_input = df.copy()
    df_x = data_input["x"].to_frame()
    
    lags = range(1,lags_value)
    df_x= df_x.assign(**{
        f'{col}_{lag}': df_x[col].shift(lag)
        for lag in lags
        for col in df_x
    })
    df_x["x_sum"] = df_x.sum(axis=1,skipna=False)
    col_name = "x_sum_" + str(lags_value)
    data_input[col_name] =df_x["x_sum"]
    data_input["X"] = data_input[col_name]
    data_input["X"] = data_input["X"].shift()
    data_input["Y"] = data_input["y"]
    data_input = data_input[lags_value:]
    return data_input

def predict_result_in_sample_lag_determine(df_train):
    y = np.array(df_train['Y'])
    X = np.array(df_train['X'])
    est = sm.OLS(y, X)
    est2 = est.fit()
    Rsquared = est2.rsquared
    Coefficients = est2.params
    t_stats = est2.tvalues
    return [Rsquared,Coefficients,t_stats]
    
def predict_result_in_sample_frequency_determine(df_all,sample_frequency):
    df_all = df_all.resample(sample_frequency,label='right', closed='right').sum()
    df_all["x1"] = df_all["x"].shift()
    df_all = df_all[1:]
    y = np.array(df_all['y'])
    X = np.array(df_all['x1'])
    est = sm.OLS(y, X)
    est2 = est.fit()
    
    Rsquared = est2.rsquared
    Coefficients = est2.params
    t_stats = est2.tvalues
    ols_result=[Rsquared,Coefficients,t_stats]
    return ols_result

def predict_result_in_sample(df_train, threshold, close_position = True):
    y = np.array(df_train['Y'])
    X = np.array(df_train['X'])
    est = sm.OLS(y, X)
    est2 = est.fit()
    
    df_train["predicted_y"] = est2.predict(X)
    threshold_quantile = df_train['X'].quantile(q=threshold)
    print(f"threshold is {threshold_quantile}")
    print(est2.summary())


    df_train["predicted_y_trancated"] = df_train.apply(lambda x: 0 if np.absolute(x["X"]) < np.absolute(threshold_quantile) else x["predicted_y"], axis = 1)

    df_train['result_direction'] = df_train.apply(lambda x: 1 if x["Y"]*x["predicted_y_trancated"]>0 else 0, axis = 1)
    df_train['result_direction'] = df_train.apply(lambda x: -1 if x["Y"]*x["predicted_y_trancated"]<0 else x["result_direction"], axis = 1)
    
    df_train["direction"] = df_train['predicted_y_trancated'].apply(lambda x: 1 if x>0 else -1)
    df_train["direction"] = df_train.apply(lambda x: 0 if x['result_direction'].round() == 0 else x['direction'], axis = 1)
    
    print(f"hit ratio is: {df_train[['result_direction']].value_counts(normalize= True)}")
    if close_position:
        df_train["bal"] = df_train["direction"] #close position when that second end

        
        df_train["result_abs"] = df_train.apply(lambda x: x["bal"] * x["Y"], axis = 1 )        
        df_train["result_abs_cum_sum"] = df_train["result_abs"].cumsum()
        df_train["result_abs_cum_sum"].plot()
    else:

        df_train["bal"] = df_train["direction"].cumsum() # not close position
        
        df_train["result_abs"] = df_train.apply(lambda x: x["bal"] * x["Y"], axis = 1 )        
        df_train["result_abs_cum_sum"] = df_train["result_abs"].cumsum()
        df_train["result_abs_cum_sum"].plot()
    return df_train


    
def predict_result_out_sample(df_train, df_test):
    y = np.array(df_train['close_change'])
    X = np.array(df_train['signed_volume_lag_1'])
    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())
    
    X_out_of_sample = np.array(df_test['signed_volume_lag_1'])
    df_test["predicted_y"] = est2.predict(X_out_of_sample)

    df_test['result_direction'] = df_test.apply(lambda x: 1 if x["close_change"]*x["predicted_y"]>0 else 0, axis = 1)
    df_test['result_direction'] = df_test.apply(lambda x: -1 if x["close_change"]*x["predicted_y"]<0 else x["result_direction"], axis = 1)

    df_test[['result_direction']].value_counts(normalize= True)

    df_test["result_abs"] = df_test.apply(lambda x: np.absolute(x["close_change"]) if x["result_direction"]>0 else -np.absolute(x["close_change"]), axis = 1 )

    df_test["result_abs_cum_sum"] = df_test["result_abs"].cumsum()
    return df_test


def DL_model(df_all, lags_value):
    df_x = df_all["x"].to_frame()

    lags = range(1,lags_value+1)
    df_lags= df_x.assign(**{
        f'{col}_{lag}': df_x[col].shift(lag)
        for lag in lags
        for col in df_x
    })
    df_lags = df_lags.drop(['x'], axis=1)
    df_lags = df_lags[lags_value:]
    print(f'X matrix (not include the current x):')
    print(df_lags)
    X = np.array(df_lags)
    y = np.array(df_all['y'][lags_value:])
    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())



def ARDL_model(df_all, AR_lags_value, DL_lags_value, interactive = False):
    df_x = df_all["x"].to_frame()
    df_y = df_all["y"].to_frame()

    DL_lags = range(1,DL_lags_value+1)
    df_x_lags= df_x.assign(**{
        f'{col}_{lag}': df_x[col].shift(lag)
        for lag in DL_lags
        for col in df_x
    })
    df_x_lags = df_x_lags.drop(['x'], axis=1)

    AR_lags = range(1,AR_lags_value+1)
    df_y_lags= df_y.assign(**{
        f'{col}_{lag}': df_y[col].shift(lag)
        for lag in AR_lags
        for col in df_y
    })
    df_y_lags = df_y_lags.drop(['y'], axis=1)

    if interactive:
        df_lags = pd.concat([df_x_lags, df_y_lags], axis=1)
        df_lags["interactive_term_x1_y1"] = df_lags["x_1"] * df_lags["y_1"]

    else:
        df_lags = pd.concat([df_x_lags, df_y_lags], axis=1)

    df_lags = df_lags[DL_lags_value:]
    print(f'X matrix (not include the current x):')
    print(df_lags)
    X = np.array(df_lags)
    y = np.array(df_all['y'][DL_lags_value:])
    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())

def plot_fill(df_od,ax):
    for i,row in df_od.iterrows():
        color = "r" if row["bid_x"].startswith("f") else "g"
        style = color + 'o'
        ax.plot(row["_time_y"],float(row['price_y']),style,markersize = 10)
        ax.hlines(y=float(row["price_x"]),xmin=row["_time_x"],xmax=row["_time_y"],colors = color,linewidth = 3)
        ax.hlines(y=float(row["fairPrice_x"]),xmin=row["_time_x"],xmax=row["_time_y"],colors = 'yellow',linewidth = 3)
        
def plot_can(df_od,ax):
    for i,row in df_od.iterrows():
        color = "r" if row["bid_x"].startswith("f") else "g"
        style = color + 'X'
        ax.plot(row["_time_y"],float(row['price_y']),style,markersize = 10)
        ax.hlines(y=float(row["price_x"]),xmin=row["_time_x"],xmax=row["_time_y"],colors = color,linewidth = 3)
        ax.hlines(y=float(row["fairPrice_x"]),xmin=row["_time_x"],xmax=row["_time_y"],colors = 'yellow',linewidth = 3)
def plot_book(df,ax):
    cols = ["fair_px","bidL1px","askL1px"]
    colors = ["black","green","red"]
    df.plot(ax=ax,y=cols,drawstyle="steps-post",legend=False,linewidth=1,color=colors)

def plot_sig(df,sig,ax):

    df.plot(ax =ax,y =sig,drawstyle="steps-post",legend=False,linewidth=0.5,color="blue",grid=True)