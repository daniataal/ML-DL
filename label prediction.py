'''
author:Dani Atalla
Date:09/09/2021
time:14:21
label prediction
'''
# Import packages needed

# Data Manipulation
from importlib._common import _

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

#Stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import PowerTransformer

#Data Processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, OneHotEncoder, PowerTransformer, LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipe
from sklearn.metrics import f1_score, recall_score, precision_score

#Config
pd.pandas.set_option('display.max_columns', None)


df = pd.read_csv(r'C:\Users\97254\Downloads\archive\BankChurners.csv')
df.head()

df = df.drop(columns= ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])


o_shape = df.shape
print(f'Data shape: {o_shape} \n')
print(f'Information of data columns:{df.info()} \n')

df, df_test = train_test_split(df, test_size=0.2, random_state=42)

pd.concat([df.nunique(), df.dtypes], axis= 1).rename(columns={0:'Unique_Values',1:'Data_Type'}).sort_values('Unique_Values')


df  = df.drop(columns=['CLIENTNUM'])

df.describe().apply(lambda s: s.apply('{0:.2f}'.format))

df_numeric = df.select_dtypes (include='number')


def variable_remove_underline(name):
    """Remove the underline for a string

    Parameters
    ----------
    name : str
        The string which contains the underline to be removed

    Returns
    -------
    str
        a string with none underline
    """
    return name.replace ('_', ' ')


def create_plotly_graph_object(dataframe, fig, type_char, params, dataframe2):
    """Create each trace for a the 2 grid plotly graph according to the type of char for all the columns in a dataframe

    Parameters
    ----------
    dataframe : pd.Dataframe
        The dataframe
    fig : plotly.graph_objs._figure.Figure
        A figure with the grid
    type_char : plotly.graph_objs._
        A graph object from plotly it could be go.Violin, go.Histogram

    Returns
    -------
    None
    """
    if dataframe2 is None:
        for num, col in enumerate (dataframe):
            params['name'] = variable_remove_underline (col)
            params['y'] = dataframe[col]
            fig.add_trace (type_char (params),
                           row=(num // 2) + 1, col=(num % 2) + 1)
    else:
        for num, col in enumerate (dataframe):
            fig.append_trace (type_char (y=dataframe[col],
                                         name=col + '_Att',
                                         marker=dict (color='red')),
                              row=(num // 2) + 1, col=(num % 2) + 1)
            fig.append_trace (type_char (y=dataframe2[col],
                                         name=col + '_Non-Att',
                                         marker=dict (color='blue')),
                              row=(num // 2) + 1, col=(num % 2) + 1)


def make_2_col_grid(dataframe, type_char, params={}, dataframe2=None):
    """Create a 2 grid plotly graph according to the type of char for all the columns in a dataframe

    Parameters
    ----------
    dataframe : pd.Dataframe
        The dataframe
    type_char : plotly.graph_objs._
        A graph object from plotly it could be go.Violin, go.Histogram

    Returns
    -------
    None
    """
    num_rows = dataframe.shape[1]
    total_rows = num_rows // 2 + num_rows % 2
    fig = make_subplots (rows=total_rows, cols=2)
    fig.update_layout (
        autosize=False,
        width=1000,
        height=400 * total_rows,
        margin=dict (
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        yaxis=dict (tickformat=",.2f"),
        xaxis=dict (tickformat=",.2f"),
        paper_bgcolor="LightSteelBlue", )
    create_plotly_graph_object (dataframe, fig, type_char, params, dataframe2)
    fig.show ()

params_violin = {'box_visible': True,
                     'meanline_visible': True,
                     'opacity': 0.6,
                     'hovertemplate': 'y:%{y:20,.2f}'}

make_2_col_grid (df_numeric, go.Violin, params_violin)

for col in df.select_dtypes(include='O'):
    print(df[col].value_counts(normalize=True).apply('{0:.3f}'.format), '\n')

new_cat = {'Dependent_count':'object',
             'Total_Relationship_Count':'object',
             'Months_Inactive_12_mon':'object',
             'Contacts_Count_12_mon':'object'}

for key, value in new_cat.items():
    df[key] = df[key].astype(value)

df_cat = df.select_dtypes(include='O')
params_hist = {'histnorm':'percent',
              'legendgroup':True,}
make_2_col_grid(df_cat, go.Histogram, params_hist)

df['Attrition_Flag'].value_counts(normalize = True, dropna = False)
df['Attrition_Flag'].replace({'Existing Customer': 0, 'Attrited Customer': 1}, inplace=True)


def create_corr_heatmap(method_tittle, df):
    '''
    '''
    customc = [[0.0, '#FEC5BB'],
               [0.5, 'white'],
               [1.0, '#FD9235']]

    corr_val = [[corr.iloc[i][j] if i > j else None for j, row in enumerate (df)] for i, col in enumerate (df)]
    hovertext = [[f'corr({col}, {row})= {corr.iloc[i][j]:.3f}' if i > j else '' for j, row in enumerate (df)] for i, col
                 in enumerate (df)]

    heat = go.Heatmap (z=corr_val,
                       x=list (corr.index),
                       y=list (corr.columns),
                       xgap=1, ygap=1,
                       colorscale=customc,
                       colorbar_thickness=20,
                       colorbar_ticklen=3,
                       hovertext=hovertext,
                       hoverinfo='text',
                       zmin=-1,
                       zmax=1
                       )

    title = method_tittle + ' Correlation Matrix'

    layout = go.Layout (title_text=title, title_x=0.5,
                        width=600, height=600,
                        xaxis_showgrid=False,
                        yaxis_showgrid=False,
                        yaxis_autorange='reversed')

    fig = go.Figure (data=[heat], layout=layout)
    fig.show ()

df_numeric = df.select_dtypes(include='number')
corr = df.select_dtypes(include='number').corr()
create_corr_heatmap('Pearson', corr)

att = df_numeric[df_numeric['Attrition_Flag']==1].drop(columns=['Attrition_Flag'])
n_att = df_numeric[df_numeric['Attrition_Flag']==0].drop(columns=['Attrition_Flag'])
make_2_col_grid(att, go.Box, _, n_att)

list_dim = [dict (label=i, values=df[i]) for i in df_numeric]
textd = ['Attrited Customer' if at == 1 else 'Non-Attrited Customer' for at in df_numeric['Attrition_Flag']]

fig = go.Figure (data=go.Splom (
    dimensions=list_dim,
    diagonal=dict (visible=False),
    text=textd,
    showupperhalf=False,
    legendgroup=True,
    marker=dict (color=df_numeric['Attrition_Flag'],
                 size=2.5,
                 colorscale='Bluered',
                 line=dict (width=0.5,
                            color='rgb(230,230,230)'))))

title = "Scatterplot between all numeric variables"
fig.update_layout (title={'text': title,
                          'xanchor': 'left',
                          'yanchor': 'top',
                          'font_size': 30},
                   dragmode='select',
                   width=1500,
                   height=1500,
                   hovermode='closest',
                   font=dict (size=8))

fig.show ()


def create_cat_df_stack(dataframe, target):
    df_cat = dataframe.select_dtypes (include='O')
    df_cat = pd.concat ([df_cat, dataframe[target]], axis=1)
    return df_cat


def create_cat_stacked_bars(dataframe, fig, target, target_map):
    for num, col in enumerate (dataframe):
        if col == target:
            pass
        else:
            df_stack = df_cat.groupby ([target, col]).size ().reset_index ()
            df_stack.columns = [target, col, 'Counts']
            df_stack['Percentage'] = df.groupby ([target, col]).size ().groupby (level=0).apply (
                lambda x: x / float (x.sum ())).values
            if target_map is not None:
                df_stack[target].replace (target_map, inplace=True)

            trace = go.Figure (data=[go.Bar (x=label_df.Attrition_Flag,
                                             y=label_df.Percentage,
                                             name=label)
                                     for label, label_df in df_stack.groupby (col)])
            row = (num // 2) + 1
            col = (num % 2) + 1
            for i in trace['data']:
                fig.add_trace (i, row, col)


def create_stacked_grid(dataframe_complete, target, target_map=None):
    dataframe = create_cat_df_stack (dataframe_complete, target)
    num_rows = dataframe.shape[1] - 1
    total_rows = num_rows // 2 + num_rows % 2

    names_fig = list (dataframe.columns)
    names_fig = names_fig[:-1]

    fig = make_subplots (rows=total_rows, cols=2, subplot_titles=names_fig)
    fig.update_layout (autosize=False,
                       width=1000,
                       height=400 * total_rows,
                       margin=dict (l=50,
                                    r=50,
                                    b=100,
                                    t=100,
                                    pad=4),
                       paper_bgcolor="LightSteelBlue",
                       barmode="stack",
                       showlegend=False,
                       title=target + ' vs categoricals', )

    for row in range (total_rows):
        fig.update_yaxes (title_text="Percentage", tickformat=",.0%", row=row, col=1)
        fig.update_yaxes (title_text="Percentage", tickformat=",.0%", row=row, col=2)

    create_cat_stacked_bars (dataframe, fig, target, target_map)
    fig.show ()

target_map = {0:'Existing Customer', 1: 'Attrited Customer'}
create_stacked_grid(df, 'Attrition_Flag', target_map)

categories=[['Unknown','Uneducated', 'High School', 'College','Graduate','Post-Graduate', 'Doctorate'],
           ['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
           ['Blue','Silver','Gold','Platinum']]

categorical_ordinal_cat = ['Education_Level','Income_Category','Card_Category']
categorical_ordinal = ['Dependent_count','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon']

df_target = df['Attrition_Flag'].reset_index(drop=True)

df_cat_ord = df[categorical_ordinal]
enc_1 = OrdinalEncoder()
enc_1.fit(df[categorical_ordinal])
df_cat_ord = pd.DataFrame(columns= df[categorical_ordinal].columns,
                          data = enc_1.transform(df[categorical_ordinal]))

enc_2 = OrdinalEncoder(categories=categories)
enc_2.fit(df[categorical_ordinal_cat])
df_cat_enc = pd.DataFrame(columns= df[categorical_ordinal_cat].columns,
                          data = enc_2.transform(df[categorical_ordinal_cat]))


df_cat_enc = pd.concat([df_cat_enc, df_cat_ord], axis = 1)
df_cat_enc = pd.concat([df_target,df_cat_enc], axis = 1)

corr = df_cat_enc.corr(method='spearman')
create_corr_heatmap('Spearman', corr)

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(df_numeric.drop(columns = 'Attrition_Flag'))

test_cat_cols = ['Gender',
                'Total_Relationship_Count',
                'Months_Inactive_12_mon',
                'Contacts_Count_12_mon',
                'Card_Category',
                'Dependent_count']

df_cat_test = df[test_cat_cols]

label_dict = defaultdict(LabelEncoder)
df_cat_test = df_cat_test.apply(lambda x: label_dict[x.name].fit_transform(x))

chi_scores = chi2(df_cat_test, df['Attrition_Flag'])
print(chi_scores[1])
results = pd.DataFrame(data = [chi_scores[1],
            ['Accepted H0' if i >= 0.05 else 'Rejected H0' for i in chi_scores[1] ]],
             index = ['P-value','Result'],
            columns = df_cat_test.columns).transpose()


class OutlierRemover (BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5, only_max=False):
        self.factor = factor
        self.only_max = only_max
        self.saved_params = {}

    def outlier_removal_fit(self, X, y=None):
        X_val = X.values
        q1 = np.quantile (X_val, 0.25)
        q3 = np.quantile (X_val, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - (self.factor * iqr)
        upper_bound = q3 + (self.factor * iqr)
        self.saved_params[X.name] = {'lower_bound': lower_bound,
                                     'upper_bound': upper_bound}

    def outlier_removal_transf(self, X, y=None):
        X = pd.Series (X).copy ()
        upper_bound = self.saved_params[X.name]['upper_bound']
        lower_bound = self.saved_params[X.name]['lower_bound']
        if self.only_max:
            X.loc[((X > upper_bound))] = np.nan
        else:
            X.loc[((X < lower_bound) | (X > upper_bound))] = np.nan
        return pd.Series (X)

    def fit(self, X, y=None):
        if X.ndim == 1:
            self.outlier_removal_fit (X)
        else:
            X.apply (self.outlier_removal_fit)
        return self

    def transform(self, X, y=None):
        if X.ndim == 1:
            return self.outlier_removal_transf (X)
        else:
            return X.apply (self.outlier_removal_transf)


outlier_remover = OutlierRemover ()
ol = outlier_remover.fit (df_numeric.drop (columns='Attrition_Flag'))
df_without_ol = ol.transform (df_numeric.drop (columns='Attrition_Flag'))
df_without_ol['Attrition_Flag'] = df_numeric['Attrition_Flag']
df_without_ol = df_without_ol.dropna ()
num_drops = df_numeric.shape[0] - df_without_ol.shape[0]
print (f'Total drops = {num_drops}, {num_drops / df_numeric.shape[0]:.2%} loss')

yj = PowerTransformer()
yj.fit(df_without_ol.drop(columns = 'Attrition_Flag'))
df_gauss = pd.DataFrame(data = yj.transform(df_without_ol.drop(columns = 'Attrition_Flag')),
             columns = df_without_ol.columns[:-1])

make_2_col_grid(df_gauss, go.Violin, params_violin)


anova = f_classif(df_gauss,
                  df_without_ol['Attrition_Flag'])
anova = pd.DataFrame(anova,
                     columns=df_gauss.columns,
                     index = ['Predictive Power(F-score)','p-value'])\
                    .transpose().sort_values('Predictive Power(F-score)',
                                              ascending=False)
anova['H0_Rejected'] = anova['p-value']<0.05
anova


anova = f_classif(df_numeric.drop(columns='Attrition_Flag'),
                  df_numeric['Attrition_Flag'])
anova = pd.DataFrame(anova,
                     columns=df_gauss.columns,
                     index = ['Predictive Power(F-score)','p-value'])\
                    .transpose().sort_values('Predictive Power(F-score)',
                                              ascending=False)
anova['H0_Rejected'] = anova['p-value']<0.05
anova

pd.DataFrame(data = mutual_info_classif(df_gauss, df_without_ol['Attrition_Flag']),
             index= df_gauss.columns,
             columns = ['Mutual_Info']).sort_values('Mutual_Info', ascending = False)

def AddNewFeatures(X):
    X['Change_ticket_Q4_Q1'] = np.where(X['Total_Ct_Chng_Q4_Q1'] == 0, 0,
                                         X['Total_Amt_Chng_Q4_Q1']/X['Total_Ct_Chng_Q4_Q1'])
    X['Avg_Ticket'] = np.where(X['Total_Trans_Ct'] == 0, 0,
                                         X['Total_Trans_Amt']/X['Total_Trans_Ct'])

    X['Est_RevBal_Q4_Q1'] =  X['Total_Revolving_Bal']*X['Total_Amt_Chng_Q4_Q1']
    X['Credit_Limit_per_Month'] = X['Credit_Limit']/X['Months_on_book']
    X['Credit_Limit_per_Age'] = X['Credit_Limit']/X['Customer_Age']
    X['Contacts_per_product'] = X['Contacts_Count_12_mon']/X['Total_Relationship_Count']
    return X

df_exp = df.copy()
df_numeric = df_exp.select_dtypes(include = 'number').drop(columns= 'Attrition_Flag')
df_numeric = ol.transform(df_numeric)
df_exp = pd.concat([df_exp[df_exp.select_dtypes(exclude = 'number').columns.union(['Attrition_Flag'])],
           df_numeric], axis = 1)
df_exp = AddNewFeatures(df_exp)
df_exp = df_exp.dropna()

exp_cols = ['Attrition_Flag','Change_ticket_Q4_Q1', 'Avg_Ticket',
            'Est_RevBal_Q4_Q1','Credit_Limit_per_Month', 'Credit_Limit_per_Age',
            'Contacts_per_product']
make_2_col_grid(df_exp[exp_cols], go.Violin, params_violin)

att = df_exp[df_exp['Attrition_Flag']==1].drop(columns=['Attrition_Flag'])
n_att = df_exp[df_exp['Attrition_Flag']==0].drop(columns=['Attrition_Flag'])
make_2_col_grid(att[exp_cols[1:]], go.Box, _, n_att[exp_cols[1:]])

anova = f_classif(df_exp[exp_cols].drop(columns=['Attrition_Flag']),
                  df_exp['Attrition_Flag'])
anova = pd.DataFrame(anova,
                     columns=df_exp[exp_cols].columns[1:],
                     index = ['Predictive Power(F-score)','p-value'])\
                    .transpose().sort_values('Predictive Power(F-score)',
                                              ascending=False)
anova['H0_Rejected'] = anova['p-value']<0.05
anova

df = pd.read_csv(r'C:\Users\97254\Downloads\archive/BankChurners.csv')
X_train, X_test, y_train, y_test = preProcess(df)
X_train, y_train, ordinal_encoder_auto, ordinal_encoder_cat, ohe_encoder = encode(X_train, y_train)
X_test, y_test = encode(X_test, y_test, ordinal_encoder_auto, ordinal_encoder_cat, ohe_encoder, False)
preprocessor = outlierRemoveDf(X_train)
X_train, X_test = OutlierTransform(X_train, X_test, preprocessor)
X_train, y_train = dropNa(X_train, y_train)
X_test, y_test = dropNa(X_test, y_test)
X_train = AddNewFeaturesRefined(X_train)
X_test = AddNewFeaturesRefined(X_test)


def modelGrid(X, Y, vainilla=False):
    numeric_cols = ['Customer_Age', 'Months_on_book', 'Credit_Limit',
                    'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                    'Avg_Utilization_Ratio', 'Change_ticket_Q4_Q1',
                    'Est_RevBal_Q4_Q1', 'Contacts_per_product']
    if vainilla:
        numeric_cols = numeric_cols[:-3]

    passthrough_cols = ['Education_Level', 'Income_Category', 'Card_Category', 'Gender_F',
                        'Gender_M', 'Dependent_count', 'Total_Relationship_Count',
                        'Months_Inactive_12_mon', 'Contacts_Count_12_mon']

    # Random forest model
    rf_params = {'classifier__min_samples_split': [50, 75, 100, 250],
                 'classifier__max_depth': [*range (3, 11)]}
    rf_model = RandomForestClassifier ()

    oversample = SMOTE (random_state=0)

    numeric_transformer = Pipeline (steps=[('scaler', StandardScaler ())])
    scaler = ColumnTransformer (transformers=[('num', numeric_transformer, numeric_cols),
                                              ('pass', 'passthrough', passthrough_cols)])

    clf = ImbPipe (steps=[('scaler', scaler),
                          ('oversampling', SMOTE ()),
                          ('classifier', rf_model)])

    if vainilla:
        clf = ImbPipe (steps=[('scaler', scaler),
                              ('classifier', rf_model)])

    n_folds = 5
    scoring = ['f1', 'recall', 'precision']
    gscv = GridSearchCV (clf, param_grid=rf_params, scoring=scoring, n_jobs=-1, cv=n_folds, refit='f1')
    gscv.fit (X, Y)

    return gscv

rf_model = modelGrid(X_train, y_train)
results = pd.DataFrame(rf_model.cv_results_)
print(results.shape)
best_results = results[(results['rank_test_f1'] <= 5) |
                       (results['rank_test_recall'] <= 5) |
                       (results['rank_test_precision'] <= 5)]
best_results


def confMatrix(model, X, y):
    '''
    Visualize a confusion matrix with the best parameters and CV score for the model

    Parameters
    ----------
    model : Sklearn trained model
        The model already fitted

    Returns
    -------
    None

    '''
    from sklearn.metrics import confusion_matrix
    print ("Best parameter (CV score=%0.3f):" % model.best_score_)
    print (model.best_params_)
    # Generate predictions with the model using our X values
    y_pred = model.predict (X)
    cm = confusion_matrix (y, y_pred)
    # Get the confusion matrix
    sns.heatmap (cm / np.sum (cm), annot=True,
                 fmt='.2%', cmap='Blues')

confMatrix(rf_model, X_train, y_train)

confMatrix(rf_model, X_test, y_test)


y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

print('Train:')
print(f'F1Score: {f1_score(y_train, y_pred_train):.2%}')
print(f'Recall: {recall_score(y_train, y_pred_train):.2%}')
print(f'Precision: {precision_score(y_train, y_pred_train):.2%}')

print('\n Test:')
print(f'F1Score: {f1_score(y_test, y_pred_test):.2%}')
print(f'Recall: {recall_score(y_test, y_pred_test):.2%}')
print(f'Precision: {precision_score(y_test, y_pred_test):.2%}')

X_train, X_test, y_train, y_test = preProcess(df)
X_train, y_train, ordinal_encoder_auto, ordinal_encoder_cat, ohe_encoder = encode(X_train, y_train)
X_test, y_test = encode(X_test, y_test, ordinal_encoder_auto, ordinal_encoder_cat, ohe_encoder, False)

rf_model_vainilla = modelGrid(X_train, y_train, True)
results_vainilla = pd.DataFrame(rf_model_vainilla.cv_results_)
print(results_vainilla.shape)
best_results_vainilla = results_vainilla[(results_vainilla['rank_test_f1'] <= 5) |
                       (results_vainilla['rank_test_recall'] <= 5) |
                       (results_vainilla['rank_test_precision'] <= 5)]
best_results_vainilla

y_pred_train = rf_model_vainilla.predict(X_train)
y_pred_test = rf_model_vainilla.predict(X_test)

print('Train:')
print(f'F1Score: {f1_score(y_train, y_pred_train):.2%}')
print(f'Recall: {recall_score(y_train, y_pred_train):.2%}')
print(f'Precision: {precision_score(y_train, y_pred_train):.2%}')

print('\n Test:')
print(f'F1Score: {f1_score(y_test, y_pred_test):.2%}')
print(f'Recall: {recall_score(y_test, y_pred_test):.2%}')
print(f'Precision: {precision_score(y_test, y_pred_test):.2%}')

