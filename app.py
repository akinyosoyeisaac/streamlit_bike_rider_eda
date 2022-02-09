# loading streamlit library
import streamlit as st

# loading data manipulation library
import pandas as pd
import numpy as np

# loading visualization library
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache(persist=True)
def loader():
    '''
    :return: a pandas dataframe object
    '''
    df = pd.read_csv(r'.\Divvy_clean.csv')
    return df


def main():
    '''
    class: main()
    This function helps carryout an exploratory data analysis of bike rider
    with streamlit
    '''

    st.title('Bike Rider EDA')
    st.text("In this app users will have the free will to independently explore the bike rider dataset")


    df = loader()     # bike rider dataset

    st.header('Exploratory Data Analysis')
    if st.checkbox('Show dataset'):
        st.text('Data preview')
        choose = st.selectbox('choose you preview type', ['head', 'tail'])
        if choose == 'head':
            st.dataframe(df.head())
        elif choose == 'tail':
            st.dataframe(df.tail())
        else:
            st.write('nothing to show')

    if st.checkbox('Data shape'):
        if st.button('row'):
            st.write(df.shape[0])
        if st.button('column'):
            st.write(df.shape[1])

    if st.checkbox('Descriptive statistics'):
        view = st.selectbox('type', ['categorical', 'numerical'])
        if view == 'categorical':
            st.write(df.describe(include='object', datetime_is_numeric=True).T)
        if view == 'numerical':
            st.dataframe(df.describe(exclude='object', datetime_is_numeric=True).T)

    st.text('Display column names')
    if st.button('columns name'):
        st.write(df.columns)

    if st.checkbox('Chose column(s) to preview'):
        st.text('Select column(s) to preview')
        column = st.multiselect('column(s)', df.columns)
        st.write(df[column])

    cat_cols = df.select_dtypes(include='object').columns
    if st.checkbox('Unique values for categorical columns'):
        col = st.selectbox('select column', cat_cols)
        st.write(df[col].unique())
        st.text('Length of the selected column')
        st.write(len(df[col].unique()))

    st.header('Visualization')
    if st.checkbox('Target count'):
        fig = plt.figure()
        sns.countplot(data=df, x='member_casual')
        st.pyplot(fig)

    num_cols = df.select_dtypes(include=np.number).columns

    st.text('Display numerical column name')
    st.selectbox('column name', num_cols)

    if st.checkbox('Histogram of numerical columns'):
        cols = st.multiselect('select columns', num_cols, default = ['distance(km)', 'velocity'])
        st.write(f'Number of columns selected: {len(cols)}')
        st.text('Do well to check the info below')
        st.info('For this run effectively select more than one column')
        fig, axes = plt.subplots(nrows=len(cols), figsize=(30,30))
        for ax, col in zip(axes, cols):
            sns.histplot(data=df, x=col, ax=ax)
        st.pyplot(fig)

    if st.checkbox('Correlation plot'):
        cols = st.multiselect('select columns', num_cols, default = ['distance(km)', 'velocity'])
        d = df[cols]
        corr = d.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig = plt.figure()
        sns.heatmap(corr, mask=mask, annot=True)
        st.pyplot(fig)



def sidebar():
    if st.sidebar.button('About'):
        st.sidebar.text('This app is built by Engr Carnot using streamlit to a simple EDA using bike rider dataset')

    if st.sidebar.button('References'):
        st.sidebar.markdown('> References')
        st.sidebar.markdown('>> 1. [youtube video by **Jcharis**](https://www.youtube.com/watch?v=LmZcEMFUIc&t=2881s)')
        st.sidebar.markdown('>> 2. [streamlit documentation](https://docs.streamlit.io/library/api-reference/widgets/st.multiselect)')
        st.sidebar.markdown('>> 3. [pandas documentation](https://pandas.pydata.org/docs/reference/)')

    if st.sidebar.button('Library'):
        st.sidebar.markdown('**Streamlit**')
        st.sidebar.markdown('**Pandas**')
        st.sidebar.markdown('**Numpy**')
        st.sidebar.markdown('**Matplotlib**')
        st.sidebar.markdown('**Seaborn**')

if __name__ == '__main__':
    main()
    sidebar()