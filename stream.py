import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from matplotlib import pyplot as plt
import statsmodels.api as sm
plt.style.use('seaborn')

@st.cache
def load_data():

    dados = pd.read_csv('desafio1.csv', index_col = 'RowNumber')

    return dados

@st.cache
def get_aux():

    dados = load_data()

    # Change genero to numeric
    #dados = dados.join(pd.get_dummies(dados['genero'], drop_first = True))
    #dados.drop('genero', axis = 1, inplace = True)

    # Numeric and categorical features
    num = dados.select_dtypes(['int64', 'float64', 'uint8'])
    cat = dados.select_dtypes(['object'])

    # Calculate State
    aux = pd.DataFrame(
    {
        
        'type': dados.dtypes,
        'missing values': dados.isnull().sum(),
        'mean': num.mean(),
        'median': num.median(),
        'max': num.max(),
        'min': num.min(),
        '# Unique Values': dados.nunique(),
        'STD': num.std(),
        'Skew': num.skew(),
        'Kurtosis': num.kurtosis()
           
    },
    index = dados.columns)

    return aux

@st.cache
def get_corr(data):
    num = data.select_dtypes(['int64', 'float64', 'uint8'])
    corr = num.corr()

    return corr, num.columns

def main():

    st.title('Aceleradev-Data-Science-Week3')

    # Load data
    data = load_data()

    # Raw data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    # Aux stats
    aux = get_aux()

    st.subheader('Statistics')

    st.write(aux.fillna(''))

    st.header('Univariate analysis')

    # Histogram
    st.subheader('Numeric Features')
    var_hist = st.selectbox('Select Variable for histogram:',
     ['saldo_conta', 'pontuacao_credito', 'idade', 'nivel_estabilidade'])

    cat_hist = st.selectbox('Select color variable:',
     ['genero', 'estado_residencia', 'nivel_estabilidade', 'numero_produtos', 'possui_cartao_de_credito', 'membro_ativo'])
    n_bins = st.slider('Number of bins:', 5, 30, 15)

    fig = px.histogram(data, x = var_hist, nbins = n_bins, color = cat_hist)
    st.plotly_chart(fig, use_container_width=True)

    # Bar plot
    st.subheader('Categorical Features')

    cat_bar = st.selectbox('Select Feature:',
     ['genero', 'estado_residencia', 'nivel_estabilidade', 'numero_produtos', 'possui_cartao_de_credito', 'membro_ativo'])
    st.bar_chart(data.astype('object')[cat_bar].value_counts())

    st.header('Multivariate analysis')
    # Scatter plot
    st.subheader('Numeric x Numeric')
    x_scat = st.selectbox('Select X:',
     ['saldo_conta', 'pontuacao_credito', 'idade'])
    y_scat = st.selectbox('Select Y:',
     ['pontuacao_credito', 'saldo_conta', 'idade'])
    col_scat = st.selectbox('Select Color:',
     ['genero', 'estado_residencia', 'nivel_estabilidade', 'numero_produtos', 'possui_cartao_de_credito', 'membro_ativo', 'pontuacao_credito', 'idade'])
    n_sample = st.slider('Sample Size:', 0, data.shape[0],2000)

    f_scat = px.scatter(
        data.sample(n_sample), 
        x = x_scat,
        y = y_scat,
        color = col_scat
        )

    st.plotly_chart(f_scat)

    # Boxplot
    st.subheader('Numeric x Categoric')

    y_box = st.selectbox('Select Y:',
     ['saldo_conta', 'pontuacao_credito', 'idade', 'nivel_estabilidade'])
    x_box = st.selectbox('Select X:',
     ['genero', 'estado_residencia', 'numero_produtos', 'possui_cartao_de_credito', 'membro_ativo'])
    color_box = st.selectbox('Select X:',
     [None, 'genero', 'estado_residencia', 'numero_produtos', 'possui_cartao_de_credito', 'membro_ativo'])
    

    f_boxplot = px.box(
        data,
        y = y_box,
        x = x_box,
        color = color_box
    )

    st.plotly_chart(f_boxplot)

    # Correlation
    st.subheader('Correlation')
    corr, labels_corr = get_corr(data)
    fig_corr = px.imshow(corr, x = labels_corr, y = labels_corr)
    st.plotly_chart(fig_corr)

    # Linear regression
    st.header('Linear Regression')
    target_lin = st.selectbox('Select Dependent Variable:',
     ['saldo_conta', 'pontuacao_credito', 'idade', 'nivel_estabilidade'])

    x = pd.get_dummies(data.iloc[:,2:].astype({'numero_produtos':'str'})).drop(target_lin, axis = 1)
    y = data[target_lin]
    model = sm.OLS(y, x).fit()

    st.text(model.summary())

    if st.checkbox('Plot'):
        fig_lin = px.scatter(data, x = model.predict(x), y = target_lin, labels = {'x': 'prediction', 'y': target_lin})
        st.plotly_chart(fig_lin)
    



if __name__ == '__main__':
    main()