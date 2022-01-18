import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st

colors = {
        'Občanská demokratická strana': 'blue',
        'Řád národa - Vlastenecká unie': (1, 0.1, 1),
        'Cesta odpovědné společnosti': (0.77, 1, 0.3),
        'Česká strana sociálně demokratická': 'orange',
        'Volte Pravý Blok': (0, 0.67, 0.8),
        'Radostné Česko': (1, 0.4, 0.5),
        'Starostové a nezávislí': (0, 0.1, 0.02),
        'Komunistická strana Čech a Moravy': 'red',
        'Strana zelených': 'green',
       'Rozumní - stop migraci a diktátu EU': (0.7, 0, 0),
       'Společnost proti developerské výstavbě v Prokopském údolí': (0.27, 0.4, 0),
       'Strana svobodných občanů': (0, 1, 0.5),
       'Blok proti islamizaci - Obrana domova': (0.2, 0.1, 0),
       'Občanská demokratická aliance': (0, 0.3, 0.6),
       'Česká pirátská strana': 'brown',
       'OBČANÉ 2011-SPRAVEDLNOST PRO LIDI': (0.5, 1, 0.5),
       'Unie H.A.V.E.L.': (0, 0.8, 0.7),
       'Česká národní fronta': (0, 0.2, 0.2),
       'Referendum o EU': (0, 0.9, 0.9),
       'TOP 09': 'purple',
       'Ano 2011': (0.3, 0.42, 1),
       'Dobrá volba 2016': (0.6, 0, 0.3),
       'Sdružení pro republiku': (1, 0.7, 0.9),
       'Křesťanská a demokratická unie': 'yellow',
       'Česká strana národně sociální': (0.4, 0.13, 0),
       'Realisté': (0, 0.3, 0.15),
       'Sportovci': (1, 0.84, 0),
       'Dělnická strana sociální spravedlnosti': (0.5, 0, 0),
       'Svoboda a přímá demokracie': (0.3, 0.42, 1),
       'Strana Práv Občanů': (0.6, 0, 0.3),
       'Národ Sobě': (0.7, 1, 0.8)}


@st.cache
def create_basic():
    '''Imports the table from source .csv.
Capitalizes the names of the columns
Recalculates the amount votes to percentual votes ratios
Removes columns which are not related directly to parties'''

    imported_df = pd.read_csv('Districts.csv', encoding='windows-1250').fillna(0).set_index('Kraj')

    rename_lst = [party.capitalize() for party in imported_df.columns]

    basic_df = (imported_df
                .set_axis(rename_lst, axis=1)
                .drop(columns=['Vydané obálky', 'Odevzdané obálky', 'Unnamed: 4'])
                .apply(lambda x, y: x if x.name == 'Platné hlasy' else x / y * 100, y=imported_df['Platné hlasy'])
                .drop('Platné hlasy', axis=1)
                .rename(columns={'Komunistická strana čech a moravy': 'Komunistická strana Čech a Moravy',
                                 'Unie h.a.v.e.l.': 'Unie H.A.V.E.L.',
                                 'Referendum o eu referendum o evropské unii':'Referendum o EU',
                                 'Top 09':'TOP 09',
                                 'Rozumní - stop migraci a diktátu eu':'Rozumní - stop migraci a diktátu EU',
                                 'Blok proti islamizaci - obrana domova':'Blok proti islamizaci - Obrana domova'})
                .round(1)
                )
    return basic_df


def sort_table(sort_mode: str, table):
    '''Sorts the table
sort_mode can be "relevance" or "alphabetical order"
Sorting acc to alphabetical order is specific for data from 2017'''

    if sort_mode == 'relevance':
        means = table.apply(np.mean, axis=0).sort_values(ascending=False)
        return table.loc[:, means.index]
    elif sort_mode == 'alphabetical order': # Vypracovat univerzálně přes změnu capitals na bezdiakritické a pak znovu přejmenovat
        # Fucking diacritics! Not a nice thing...

        # Dict for rewriting names of parties which begin with diacritical letter
        # Designed to represent the names to take a right position during alphabetical sorting
        corr_dict = \
                {'Česká národní fronta': 'Czeská národní fronta',
                 'Česká pirátská strana': 'Czeská pirátská strana',
                 'Česká strana národně sociální': 'Czeská strana národně sociální',
                 'Česká strana sociálně demokratická': 'Czeská strana sociálně demokratická',
                 'Řád národa - vlastenecká unie': 'Rzád národa - vlastenecká unie'}

        # Dict for giving back the names of the parties from corr_dict, after sorting
        rev_corr_dict = {j : i for i, j in corr_dict.items()}

        temp_table = table.rename(columns=corr_dict)

        table = (temp_table
                .reindex(sorted(temp_table.columns), axis=1)
                .rename(columns = rev_corr_dict)
                 )

        return table

class DataHandler():
    '''Class creating instance, from which tables for specific purposes are generated'''

    def __init__(self, sort_mode: str):

        self.sort_mode = sort_mode
        self.basic_df = sort_table(sort_mode, create_basic())

    def one_party(self, party: str):
        '''Creates table containing results of party in districts.
Results are sorted descendant according to results for the party.'''
        return self.basic_df[[party]].sort_values(by = party, ascending=False)


    def more_parties(self, parties: list):
        '''Creates table containing results for selected multiple parties in districts, no sorting.'''
        return sort_table(self.sort_mode, self.basic_df[parties])


    def ranking_table(self, party: str):
        '''Creates table with ranking in districts for selected party.'''
        return self.basic_df.rank(axis=1, method='max', ascending=False).astype(int)[[party]]


    def piechart_table(self, district: str):
        '''Creates table contaning results of parties for given district'''
        intermediate = (self.basic_df
                        .loc[district]
                        .to_frame()
                        .reset_index()
                        )

        aggregated = (intermediate
                      .assign(Strana=np.where(intermediate[district] >= 1, intermediate['index'], 'Other parties'))
                      .drop('index', axis=1)
                      .groupby('Strana').sum()
                      .sort_values(by=district, ascending=False)
                      )

        without_minors = aggregated[aggregated.index != 'Other parties']
        only_minors = aggregated[aggregated.index == 'Other parties']
        return without_minors.append(only_minors)

    def minors_show(self, district):  # Implemented into DataHandler
        im = self.basic_df.loc[district].to_frame()
        minors_lst = im[(im[district] < 1) & (im[district] > 0)].sort_values(ascending=False,
                                                                             by=district).index.values.tolist()
        fin_str = '{0:^40}\n\n'.format('Other parties, gaining less than 1 % of votes')

        for party in minors_lst:
            fin_str += f'{party}\n'

        return fin_str


class PlotHandler():
    '''Class creating instance, which handles with creation of plots'''

    def district_main(self, src_data):
        '''Plots results for parties in selected districts as pie chart'''
        plt.rcParams.update({'font.size': 14})
        district_name = src_data.iloc[:, 0].name

        clrs = [colors[party] for party in src_data.index[:-1]] + ['white']
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        wedges, labels, autopct = ax.pie(src_data[district_name], labels=src_data.index, rotatelabels=False,
                                         autopct=lambda x: f'{x:.1f} %', colors=clrs)
        plt.setp(labels, fontsize=20)
        ax.set_title(f'Results for {district_name}', fontsize=28)
        plt.show()  # Potom odstranit
        return fig

    def district_minors(self, src_data):
        '''Creates the list of parties gaining less than 1 % votes'''
        fig, ax = plt.subplots(1, 1, figsize=(14, 2))
        ax.text(0.3, 0.6, src_data, fontsize=16)
        ax.axis('off')
        plt.show() # Potom odstranit
        return fig

    def party(self, src_data):
        '''Plots results for one party in all districts, sorted descendant'''
        plt.rcParams.update({'font.size': 14})
        party_name = src_data.iloc[:,0].name

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(src_data.index, src_data[party_name], color=colors[party_name], edgecolor='black')
        ax.set_xticklabels(src_data.index, rotation=90, fontsize=12)
        ax.set_ylabel('Votes (%)', fontsize=12)
        ax.set_title(f'Results for {party_name}', fontsize=14)
        ax.grid(axis='y', linewidth=1.2)
        plt.show() #Potom odstranit
        return fig

    def parties_comparative(self, src_data, P = 0.35):
        '''Plots results for selected parties in all districts as multiple bar plot
P is a diameter of the area which consists of aggregated columns, default 0.35'''

        n = src_data.shape[1]  # Amount of data sets
        w = 2 * P / n  # Width of one column
        shifts = np.linspace(-(n - 1) * w / 2, (n - 1) * w / 2, n)  # Posuny jednotlivých datových řad

        length = src_data.shape[0]  # Length of the data row
        x = np.array(range(1, length + 1))  # x values
        i = 0

        fig, ax = plt.subplots(figsize=(10, 7))

        for party in src_data.columns:
            ax.bar(x + shifts[i],
                   src_data[party],
                   w,
                   edgecolor='black',
                   color=colors[party],
                   label=party)

            ax.set_xticks(x)
            ax.set_xticklabels(src_data.index, rotation=90, fontsize=14)
            ax.set_ylabel('Votes (%)', fontsize=14)
            ax.grid(axis='y', linewidth=1.2)
            ax.legend(bbox_to_anchor=(0.5, 1.14), fontsize=12)
            i += 1

        plt.show() # Potom odstranit
        return fig


    def parties_cumulative(self, src_data):
        '''Plots results for selected parties in all districts as multiple bar plot'''
        plt.rcParams.update({'font.size': 14})

        parties = src_data.columns.tolist()
        districts = src_data.index.tolist()
        bottoms = [len(districts) * [0]]
        for party in parties[:-1]:
            bottoms.append(list((map(lambda x, y: x + y, bottoms[-1], src_data[party].to_list()))))

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        for i in range(len(parties)):
            ax.bar(districts,
                   src_data[parties[i]],
                   bottom=bottoms[i],
                   label=parties[i],
                   width=0.45,
                   color=colors[parties[i]],
                   edgecolor='black')

        ax.set_xticklabels(districts, rotation=90, fontsize=14)
        ax.legend(bbox_to_anchor=(0.5, 1.14), fontsize=12)
        ax.set_ylabel('Votes (%)', fontsize=14)
        ax.grid(axis='y', linewidth=1.2)

        plt.show() #Pak odstranit
        return fig

    def ranking(self, src_data):
        plt.rcParams.update({'font.size': 14})
        party = src_data.iloc[:, 0].name
        ds = src_data[party]

        # Array - like basic variables
        coef_array = np.max(ds) - ds + 1
        expon_array = pow(2, coef_array)
        x = np.array(range(1, np.shape(ds)[0] + 1))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(x,
               expon_array,
               edgecolor='black',
               color=colors[party])

        ax.set_title(f'{party} - ranking')
        ax.set_xticks(x)
        ax.set_xticklabels(ds.index, rotation=90, fontsize=14)
        ax.set_ylim(0, np.max(expon_array) + np.max(expon_array) / 10)
        ax.get_yaxis().set_visible(False)

        # Annotate
        for i in range(np.shape(ds)[0]):
            ax.annotate(ds[i], xy=(i + 0.90, expon_array[i] + np.max(expon_array) / 20), fontsize=14)

        plt.show() #Potom odstranit
        return fig

    def correlation_plot(self, src_data, no_Prague = False):

        def regline(ax, x, y, linestyle='-', color='black', linewidth=2.5):
            a, b = np.polyfit(x, y, 1)
            corr = np.corrcoef(x, y)[0, 1]
            if b >= 0:
                eq = 'y = {:.3f}*x + {:.3f}, corr_coef = {:.2f}'.format(a, b, corr)
            else:
                eq = 'y = {:.3f}*x + {:.3f}, corr_coef = {:.2f}'.format(a, b, corr)
            reg_x = np.linspace(x.min(), x.max(), 3)
            reg_y = a * reg_x + np.array(3 * [b])

            ax.plot(reg_x, reg_y, label=eq, linestyle=linestyle, color=color, linewidth=linewidth)

        if no_Prague == True:
            src_data = src_data.drop('Hlavní město Praha', 0)

        party1 = src_data.columns[0]
        party2 = src_data.columns[1]
        plt.rcParams.update({'font.size': 14})
        x = src_data[party1]
        y = src_data[party2]

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
        ax.legend(bbox_to_anchor=(0.5, 1.14), fontsize=12)

        plt.show() #Potom odstranit
        return fig

    def corr_comment(self, src_data, no_Prague = False):

        if no_Prague == True:
            src_data = src_data.drop('Hlavní město Praha', 0)

        party1 = src_data.columns[0]
        party2 = src_data.columns[1]
        x = src_data[party1]
        y = src_data[party2]
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

# STREAMLIT PART BEGINS HERE

# Function for displaying results of elections according to districts
def introduction_res():
    st.write('# Parliament elections 2017')
    st.write('')
    st.write('**To avoid wrong decisions, we should always look into history.**')
    st.write('Here is a small application for viewing the historical parliament election results.')
    st.write('')
    st.write('**Use the sidebar navigation on the left to choose the application layer.**')
    st.write('Although I recommend to use this application on the PC, it can be ran on the phone.')
    st.write('In that case, please **open the sidebar using the small arrow in the upper left!**')
    st.write('')
    st.write('**To display the generated objects in full screen mode, please use the double arrays.**')
    st.write('')
    st.write('**In case of any of troubles, please contact me via petr.jisa1406@gmail.com**')
    st.image('Flag.jpg')


def district_res(data_obj, plot_obj):
    st.write('## Here you can look at the results for selected districts')
    st.write('Please, select the district')
    district = st.selectbox('District', ['None'] + data_obj.basic_df.index.tolist())

    if district != 'None':
        st.pyplot(plot_obj.district_main(data_obj.piechart_table(district)))
        st.pyplot(plot_obj.district_minors(data_obj.minors_show(district)))
    else:
        st.write('Application is waiting for the selection of district')

# Function for displaying results of elections according to parties
def party_res(data_obj, plot_obj):
    st.write('## Here you can look at the ordered results for selected party')
    party = st.selectbox('Party', ['None'] + data_obj.basic_df.columns.tolist())
    if party != 'None':
        st.pyplot(plot_obj.party(data_obj.one_party(party)))
    else:
        st.write('Application is waiting for the selection of party')

def parties_comparative_res(data_obj, plot_obj):
    st.write('## Here you can look at the ordered results for multiple selected parties')
    st.write('Please, select the parties that shall be distinguished')

    parties = []
    left_col, right_col = st.columns([1, 1])

    with left_col:
        for party in data_obj.basic_df.iloc[:,:16].columns:
            if st.checkbox(party):
                parties.append(party)

    with right_col:
        for party in data_obj.basic_df.iloc[:,16:35].columns:
            if st.checkbox(party):
                parties.append(party)

    if len(parties) > 0:
        st.pyplot(plot_obj.parties_comparative(data_obj.more_parties(parties)))
    else:
        st.write('Application is waiting for the selection of parties')


def parties_cumulative_res(data_obj, plot_obj):
    st.write('## Here you can see comparison of results for the districts with each other')
    st.write('Please, select the parties that shall be distinguished')

    parties = []
    left_col, right_col = st.columns([1, 1])

    with left_col:
        for party in data_obj.basic_df.iloc[:,:16].columns:
            if st.checkbox(party):
                parties.append(party)

    with right_col:
        for party in data_obj.basic_df.iloc[:,16:35].columns:
            if st.checkbox(party):
                parties.append(party)

    if len(parties) > 0:
        st.pyplot(plot_obj.parties_cumulative(data_obj.more_parties(parties)))
    else:
        st.write('Application is waiting for the selection of parties')

def parties_rank_res(data_obj, plot_obj):

    st.write('## Here you can see ranking of the selected party in the individual districts')
    party = st.selectbox('Parties for ranking', ['None'] + data_obj.basic_df.columns.tolist())

    if party != 'None':
        st.pyplot(plot_obj.ranking(data_obj.ranking_table(party)))
    else:
        st.write('Application is waiting for the selection of district')

def relationships_res(data_obj, plot_obj):
    st.write('## Here you can see relationships between votes for the selected parties in individual districts')
    st.write('We propose the correlations between results are important only for the most important parties')
    st.write('Therefore only 10 parties with highest votes are here to be selected')
    st.write('**The correlations are often strongly affected by data from Prague, which are specific!**')
    st.write('Therefore the results including Prague and the results excluding Prague are evaluated separately')

    relevant_parties = data_obj.basic_df.iloc[:, :10].columns.tolist()
    left_col, right_col = st.columns([1, 1])
    with left_col:
        party1 = st.selectbox('Choose first party', relevant_parties)
    with right_col:
        party2 = st.selectbox('Choose second party', relevant_parties[::-1])

    if party1 == party2:
        st.write('The same parties are selected to be compared. Come on, it would be so senseless! :-)')
        st.write('**Please, select a pair of different parties**...')
    else:
        st.write('### Results from all districts')
        st.write('** Data from all districts are used, no exclusions **')
        # corr_comment = plot_obj.corr_comment(data_obj.more_parties([party1, party2]))
        # st.write(corr_comment)
        st.pyplot(plot_obj.correlation_plot(data_obj.more_parties([party1, party2])))

        st.write('### Results with exclusion of Prague')
        st.write('**Here are the results, when data from Prague are excluded from the evaluation**\n\n')
        # corr_comment = plot_obj.corr_comment(data_obj.more_parties([party1, party2]), no_Prague=True)
        # st.write(corr_comment)
        st.pyplot(plot_obj.correlation_plot(data_obj.more_parties([party1, party2]), no_Prague=True))


# Main part (drives the app)
layers = ['Introduction', 'District', 'Party', 'Parties - comparative', 'Parties - cumulative', 'Parties - ranking', 'Relationships']
layer = st.sidebar.radio('Application layer:', layers)
sorting_mode = st.sidebar.radio('Parties selection mode:', ['relevance (recommended)', 'alphabetical order'])

if sorting_mode == 'relevance (recommended)':
    data_handler = DataHandler('relevance')
else:
    data_handler = DataHandler(sorting_mode)

plot_handler = PlotHandler()

if layer == 'Introduction':
    introduction_res()
elif layer == 'District':
    district_res(data_handler, plot_handler)
elif layer == 'Party':
    party_res(data_handler, plot_handler)
elif layer == 'Parties - comparative':
    parties_comparative_res(data_handler, plot_handler)
elif layer == 'Parties - cumulative':
    parties_cumulative_res(data_handler, plot_handler)
elif layer == 'Parties - ranking':
    parties_rank_res(data_handler, plot_handler)
elif layer == 'Relationships':
    relationships_res(data_handler, plot_handler)