from scipy.stats import pearsonr
import ppscore as pps
import matplotlib.pyplot as plt
import seaborn as sns
from cslib import fetch_ts, engineer_features
import os
import warnings
warnings.filterwarnings("ignore")

#show all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
def EDA_df_info(df,country,eda_dir):
    print("... Country ... :",country)
    df['date'] = df['date'].values.astype('datetime64[D]')
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month.map("{:02}".format)
    df['Day'] = df['date'].dt.day.map("{:02}".format)
    df['Weekday'] = df['date'].dt.weekday
    df['Day_Name'] = df['date'].dt.weekday_name
    df['CalWeek'] = df['date'].dt.week

    print(df.shape)
    print(df.dtypes)
    print(df.columns)
    return (df)

def EDA_feature_structure(df,country,eda_dir):
    plt.figure(figsize=(20, 6))
    plt.subplot(1,3,1)
    df['Day_Name'].value_counts(ascending=True).plot(kind='barh')
    plt.subplot(1,3,2)
    plt.title('Data insight '+str(country), fontsize=14)
    df['Month'].value_counts(ascending=True).plot(kind='barh')
    plt.subplot(1,3,3)
    df['Year'].value_counts(ascending=True).plot(kind='barh')
#    plt.show()
    plt.savefig(str(country) + " feature structure.png", format="png", dpi=300)


def EDA_timeplot(df,country,eda_dir):
    fig, ax = plt.subplots(figsize=(15,4))
    plt.plot( 'date', 'y', data=df, color='red', linewidth=2)
    plt.title('Timeplot '+str(country), fontsize=14)
#    plt.show()
    plt.savefig(str(country) + " time series.png", format="png", dpi=300)


def EDA_moving_averages(df,country,eda_dir):
    """
    show impact of moving averages
    """
    df = df.sort_values(by=['date'],ascending=True)
    df['y_MA'] = df['y'].transform(lambda x: 0.2*x+0.2*x.shift(1)+0.2*x.shift(-1)+0.2*x.shift(2)+0.2*x.shift(-2)+0.2*x.shift(3)+0.2*x.shift(-3))
    df['total_views_MA'] = df['total_views'].transform(lambda x: 0.2*x+0.2*x.shift(1)+0.2*x.shift(-1)+0.2*x.shift(2)+0.2*x.shift(-2)+0.2*x.shift(3)+0.2*x.shift(-3))
    fig, ax = plt.subplots(figsize=(15,4))
    plt.plot( 'date', 'y_MA', data=df, color='red', linewidth=2)
    plt.plot( 'date', 'total_views_MA', data=df, color='blue', linewidth=2)
    plt.title('Moving averages over time  '+str(country), fontsize=12)
#    plt.show()
    plt.savefig(str(country) + " moving averages.png", format="png", dpi=300)


def EDA_lags(df,country,eda_dir):
    """
    Lag between features
    """
    fig, ax = plt.subplots(figsize=(15,4))
    plt.plot( 'date', 'y_MA', data=df, color='red', linewidth=2)
    plt.plot( 'date', 'total_views_MA', data=df, color='blue', linewidth=2)
    plt.title('Line  '+str(country), fontsize=12)
#    plt.show()
    plt.savefig(str(country) + " lag analysis", format="png", dpi=300)


def EDA_PPS_CORR(df,country,eda_dir):
    """
    correlations (CORR)
    predictive power score (PPS)
    """
    corr = df.corr()
    plt.figure(figsize=(10, 10))                      
    ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0,square=True, annot=True)    #cmap=sns.diverging_palette(20, 220, n=200),
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,horizontalalignment='right');
    plt.title('CORR matrix'+str(country), fontsize=12)
#    plt.show()
    plt.savefig(str(country) + " correlations.png", format="png", dpi=300)

    ppsm=pps.matrix(df)
    plt.figure(figsize=(10, 10))                       
    ax = sns.heatmap(ppsm,vmin=-1, vmax=1, center=0,square=True, annot=True)  
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
    plt.title('PPS matrix '+str(country), fontsize=12)
#    plt.show()
    plt.savefig(str(country) + " prediction performance score.png", format="png", dpi=300)

def EDA_start():
    data_dir = os.path.join(os.getcwd(), "data", "cs-train")
    print("data_dir", data_dir)
    eda_dir = os.path.join(os.getcwd(), "data_insight")
    print("... EDA start .... ")
    ## load the data
    ts_data = fetch_ts(data_dir)
    ## first info
    for country, df in ts_data.items():
        print("... Country: ... ", country)
        print("... EDA_DIR", eda_dir)
        os.chdir(eda_dir)
        EDA_df_info(df, country, eda_dir)
        EDA_feature_structure(df, country, eda_dir)
        EDA_timeplot(df, country, eda_dir)
        #        EDA_moving_averages(df,country,eda_dir)
        EDA_PPS_CORR(df, country, eda_dir)

if __name__ == "__main__":
    """
    basic EDA for training data
    """
    data_dir = os.path.join(os.getcwd(),"data","cs-train")
    print("data_dir",data_dir)
    eda_dir = os.path.join(os.getcwd(),"data_insight")
    print("... EDA start .... ")
    ## load the data
    ts_data = fetch_ts(data_dir)
    ## first info
    for country, df in ts_data.items():
        print("... Country: ... ", country)
        print("... EDA_DIR",eda_dir)
        os.chdir(eda_dir)
        EDA_df_info(df,country,eda_dir)
        EDA_feature_structure(df,country,eda_dir)
        EDA_timeplot(df,country,eda_dir)
#        EDA_moving_averages(df,country,eda_dir)
        EDA_PPS_CORR(df,country,eda_dir)