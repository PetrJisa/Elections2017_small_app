import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st

# Převod dat z .csv do DataFrame, nahrazení chybějících hodnot nulami a nastavení indexu 'Kraj'
# Tady původně problém s diakritikou, kterou nezvládá defaultní encoding v utf-8
# Použitý soubor .csv má ANSI kódování, proto je také nutné provést encoding v ANSI

@st.cache
def primary_table():
    rough_df = pd.read_csv('Districts.csv', encoding='ANSI').fillna(0).set_index('Kraj')
    basic_df = (rough_df
        .drop(columns=['Vydané obálky', 'Odevzdané obálky'])
        .apply(lambda x, y: x if x.name == 'Platné hlasy' else x/y*100, y=rough_df['Platné hlasy'])
        .drop(['Platné hlasy', 'Unnamed: 4'], axis=1)
        .round(1)
        )
    return basic_df

def max_len(*args: str):
    max = 0
    for arg in args:
        if len(arg) > max:
            max = len(arg)

    return max


def parties(table, district: str):
    intermediate = table.loc[district].sort_values(ascending=False).to_frame().reset_index()
    aggregated = (intermediate
                .assign(Strana=np.where(intermediate[district] >= 1, intermediate['index'], 'Ostatní'))
                .drop('index', axis=1)
                .groupby('Strana').sum()
                  )

    without_minors = aggregated[aggregated.index != 'Ostatní'].sort_values(by=district, ascending=False)
    only_minors = aggregated[aggregated.index == 'Ostatní']
    return without_minors.append(only_minors)

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

# Streamlit part begins here

# DataFrame for the streamlit app
basic_df = primary_table()

# Introduction
st.write('''# Parliament Elections 2017
Parliament elections in 2021 knock on the door.
To avoid wrong decisions, we should always look into history.
There is a small application for viewing the results of the last elections!''')

# Pie chart and minors
st.write('## Here you can look at the results for a given district')
district = st.selectbox('District', basic_df.index)
st.pyplot(plot_pie(basic_df, district))
st.pyplot(minors_show(basic_df, district))

# Bar charts
st.write('## Here you can look at the ordered results for selected parties')
st.write('You can do a multiple selection if you want to see results for more parties')
colors = ['blue', 'brown', 'green', 'red', 'orange']
parties_selection = st.multiselect('Parties', basic_df.columns)
for i in range(len(parties_selection)):
    st.pyplot(plot_one_party(basic_df, parties_selection[i], colors[i%5]))









