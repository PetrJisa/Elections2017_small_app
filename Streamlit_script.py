import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st

# Csv to DataFrame, decorated by streamlit.cache (to make the app faster)
@st.cache
def primary_table():
    rough_df = pd.read_csv('Districts.csv', encoding='windows-1250').fillna(0).set_index('Kraj')
    basic_df = (rough_df
        .drop(columns=['Vydané obálky', 'Odevzdané obálky'])
        .apply(lambda x, y: x if x.name == 'Platné hlasy' else x/y*100, y=rough_df['Platné hlasy'])
        .drop(['Platné hlasy', 'Unnamed: 4'], axis=1)
        .round(1)
        )

    means = basic_df.apply(np.mean, axis=0).sort_values(ascending=False)
    return basic_df.loc[:, means.index]

def parties(table, district: str):
    intermediate = table.loc[district].sort_values(ascending=False).to_frame().reset_index()
    aggregated = (intermediate
                .assign(Strana=np.where(intermediate[district] >= 1, intermediate['index'], 'Other parties'))
                .drop('index', axis=1)
                .groupby('Strana').sum()
                  )

    without_minors = aggregated[aggregated.index != 'Other parties'].sort_values(by=district, ascending=False)
    only_minors = aggregated[aggregated.index == 'Other parties']
    return without_minors.append(only_minors)

def dist_comp_table(df, selection:list):
    table = df[selection]
    means = table.apply(np.mean, axis=0).sort_values(ascending=False)
    table = table.loc[:, means.index].assign(sum=table.apply(np.sum, axis=1))
    table['Other parties'] = table['sum'].apply(lambda x: 100 - x)
    table = table.drop('sum', 1)
    return table

def plot_one_party(table, party: str, color: str):
    plt.rcParams.update({'font.size': 12})

    filtered = table[party].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(filtered.index, filtered, color=color, edgecolor='black')
    ax.set_xticklabels(filtered.index, rotation=90, fontsize=12)
    ax.set_ylabel('Votes (%)', fontsize=12)
    ax.set_title(party, fontsize=14)
    ax.grid(axis='y', linewidth=1.2)
    return fig


def plot_pie(table, district: str):
    plt.rcParams.update({'font.size': 14})
    parties_df = parties(table, district)
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    wedges, labels, autopct = ax.pie(parties_df[district], labels=parties_df.index, rotatelabels=False, autopct=lambda x: f'{x:.1f} %')
    plt.setp(labels, fontsize=20)
    ax.set_title(f'Results for {district}', fontsize=28)
    return fig

def minors_show(table, district):
    plt.rcParams.update({'font.size': 14})
    im = table.loc[district].to_frame()
    minors_lst = im[(im[district] < 1) & (im[district] > 0)].sort_values(ascending=False, by=district).index.values.tolist()
    fin_str = '{0:^40}\n\n'.format('Other parties, gaining less than 1 % of votes')

    for party in minors_lst:
        fin_str += f'{party}\n'

    fig, ax = plt.subplots(1, 1, figsize=(14, 2))
    ax.text(0.3, 0.6, fin_str, fontsize=16)
    ax.axis('off')
    return fig

def dist_comp_plot(table):
    parties = table.columns.tolist()
    colors = ['purple', 'brown', 'blue', 'orange', 'yellow', 'red', 'lightblue']
    districts = table.index.tolist()
    bottoms = [len(districts) * [0]]
    for party in parties[:-1]:
        bottoms.append(list((map(lambda x, y: x + y, bottoms[-1], table[party].to_list()))))

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for i in range(len(parties) - 1):
        ax.bar(districts,
               table[parties[i]],
               bottom=bottoms[i],
               label=parties[i],
               width=0.45,
               color=colors[i%7],
               edgecolor='black')

    others = parties[-1]
    ax.bar(districts,
           table[others],
           bottom=bottoms[-1],
           label=others,
           width=0.45,
           color='black',
           edgecolor='black')

    ax.set_xticklabels(districts, rotation=90, fontsize=14)
    ax.legend(bbox_to_anchor=(0.1, 1), fontsize=12)
    ax.set_ylabel('Votes (%)', fontsize=14)
    ax.grid(axis='y', linewidth=1.2)

    return fig

def regline(ax, x, y, linestyle='-', color='black', linewidth=2.5):
    a, b = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    if b >= 0:
        eq = 'y = {:.3f}*x + {:.3f}, corr_coef = {:.2f}'.format(a, b, corr)
    else:
        eq = 'y = {:.3f}*x + {:.3f}, corr_coef = {:.2f}'.format(a, b, corr)
    reg_x = np.linspace(x.min(), x.max(), 3)
    reg_y = a*reg_x + np.array(3*[b])

    ax.plot(reg_x, reg_y, label=eq, linestyle=linestyle, color=color, linewidth=linewidth)

def correlation_plot(table, party1: str, party2: str):
    plt.rcParams.update({'font.size': 14})
    x = table[party1]
    y = table[party2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.scatter(x,
               y,
               s=70,
               c='orange',
               marker='o',
               edgecolor='black',
               label='Votes')

    regline(ax, x, y, linewidth=1.5, linestyle='--')
    ax.set_xlabel(f'{party1} - votes (%)', fontsize=14)
    ax.set_ylabel(f'{party2} - votes (%)', fontsize=14)
    ax.grid(linewidth=1.2)
    ax.legend(bbox_to_anchor=(0.6, 1.14), fontsize=12)

    return fig

def corr_comment(table, party1: str, party2: str):
    x = table[party1]
    y = table[party2]
    cc = np.corrcoef(x, y)[0, 1]

    if cc >= 0.7:
        result = f'''A strong positive correlation was found between the votes for *{party1}* and *{party2}* in individual districts.\n
The districts with higher votes for *{party1}* showed  a significant tendency to bring higher votes for *{party2}*.'''
    elif cc >= 0.4:
        result = f'''A middle-strength positive correlation was found between the votes for *{party1}* and *{party2}* in individual districts.\n
The districts with higher votes for *{party1}* showed a slight tendency to bring higher votes for *{party2}*.'''
    elif cc >= -0.4:
        result = f'''No significant correlation was found between the votes for *{party1}* and *{party2}* in individual districts.\n
The votes for *{party1}* and votes for *{party2}* are independent.'''
    elif cc >= -0.7:
        result = f'''A middle-strength negative correlation was found between the votes for *{party1}* and *{party2}* in individual districts.\n
The districts with higher votes for *{party1}* showed a slight tendency to bring lower votes for *{party2}*.'''
    elif cc >= -1:
        result = f'''A strong negative correlation was found between the votes for *{party1}* and *{party2}* in individual districts.\n
The districts with higher votes for *{party1}* showed a significant tendency to bring lower votes for *{party2}*.'''

    return result

# Streamlit part begins here

# DataFrame for the streamlit app
basic_df = primary_table()

# Function for displaying results of elections according to districts
def introduction_res():
    st.write('''# Parliament Elections 2017
    Parliament elections in 2021 knock on the door.
    To avoid wrong decisions, we should always look into history.
    There is a small application for viewing the results of the last elections!
    Use the sidebar navigation to choose the application layer''')


def district_res():
    st.write('## Here you can look at the results for selected districts')
    districts = st.multiselect('Districts', basic_df.index)
    for district in districts:
        st.pyplot(plot_pie(basic_df, district))
        st.pyplot(minors_show(basic_df, district))

# Function for displaying results of elections according to parties
def parties_res():
    st.write('## Here you can look at the ordered results for selected parties')
    st.write('You can do a multiple selection if you want to see results for more parties')
    colors = ['purple', 'brown', 'blue', 'yellow', 'red']
    parties_selection = st.multiselect('Parties', basic_df.columns)
    for i in range(len(parties_selection)):
        st.pyplot(plot_one_party(basic_df, parties_selection[i], colors[i%5]))

def district_comp_res():
    st.write('## Here you can see comparison of results for the districts with each other')
    st.write('However, there are too many parties for one graph')
    st.write('Therefore we kindly ask you for selecting the parties, which will be distinguished')
    parties_selection = st.multiselect('Distinguished parties', basic_df.columns)
    st.pyplot(dist_comp_plot(dist_comp_table(basic_df, parties_selection)))

def relationships_res():
    st.write('## Here you can see relationships between votes for the selected parties in individual districts')
    st.write('We propose the correlations between results are important only for the most important parties')
    st.write('Therefore only 10 parties with highest votes are here to be selected')

    relevant_parties = basic_df.iloc[:, :10].columns.tolist()
    left_col, right_col = st.columns([1, 1])
    with left_col:
        party1 = st.selectbox('Choose first party', relevant_parties)
    with right_col:
        party2 = st.selectbox('Choose second party', relevant_parties)

    if party1 == party2:
        st.write('The same parties are selected to be compared. Come on, it would be so senseless! :-)')
        st.write('Please, select different ones...')
    else:
        st.write(corr_comment(basic_df, party1, party2))
        st.pyplot(correlation_plot(basic_df, party1, party2))


# Main part (drives the app)
layers = ['Introduction', 'Districts', 'Parties', 'Districts - comparative', 'Relationships']
mode = st.sidebar.radio('Application layer', layers)
if mode == 'Introduction':
    introduction_res()
elif mode == 'Districts':
    district_res()
elif mode == 'Parties':
    parties_res()
elif mode == 'Districts - comparative':
    district_comp_res()
elif mode == 'Relationships':
    relationships_res()