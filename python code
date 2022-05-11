import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA Player Stats Explorer')

st.markdown("""
This app performs simple webscraping of NBA player stats data!
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2022))))

@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)

sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]
st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
test = df_selected_team.astype(str)
st.dataframe(test)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()

if st.button('Boxplot analysis'):
    st.header('Box Plot')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')
    sns.set_style("whitegrid")
    sns.boxplot(x = 'Age', y = 'PTS', data = df)
    st.pyplot()

if st.button('Scatterplot analysis'):
    st.header('Scatter Plot')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')
    sns.set_style("whitegrid")
    sns.scatterplot(x = 'Age', y = 'PTS', data = df)
    st.pyplot()

if st.button('Lineplot analysis'):
    st.header('Line Plot')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')
    sns.set_style("whitegrid")
    sns.lineplot(x = 'Age', y = 'PTS', data = df)
    st.pyplot()

if st.button('Barplot analysis'):
    st.header('Bar Plot')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')
    sns.set_style("whitegrid")
    sns.barplot(x = 'Age', y = 'PTS', data = df)
    st.pyplot()

if st.button('Violinplot analysis'):
    st.header('Violin Plot')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')
    sns.set_style("whitegrid")
    sns.barplot(x = 'Age', y = 'PTS', data = df)
    st.pyplot()

if st.button('Histogram analysis'):
    st.header('Histogram')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')
    sns.set_style("whitegrid")
    sns.histplot(x = '3P%', data = df, kde = True)
    st.pyplot()
