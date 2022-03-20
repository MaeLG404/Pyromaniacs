# Tuto used : https://docs.streamlit.io/library/get-started/create-an-app

# Modules
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from PIL import Image




st.set_page_config(layout="wide")
st.title('Wildfires in USA - Analysis from 1992 to 2018')
data_filename = 'wildfires_final_frac0.05.csv'



# ------------   Labels used in the streamlit :
fires_number = 'Number of wildfires'
fires_causes = 'Causes of wildfires'
fires_surf = 'Surface of wildfires'
fires_dur = 'Duration of wildfires'
fires_temp = 'Temporal Data'
fires_state = 'States'

months_labels = ['Jan', 'Feb','Mar', 'Apr','May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
seasons_labels = ['Winter', 'Spring', 'Summer', 'Fall']
causes_labels = ['Individuals\' mistake', 'Criminal', 'Infrastructure accident', 'Natural (lightning)', 'Other/Unknown']
causes_labels_split = ['Individuals\' \nmistake', 'Criminal', 'Infrastructure \naccident', 'Natural \n(lightning)', 'Other/\nUnknown']
regions_list = ['East', 'West', 'North', 'South', 'Center',
     'North-East', 'North-West', 'South-East', 'South-West', 'Tropical']
regions_list_split = ['East', 'West', 'North', 'South', 'Center',
     'North-\nEast', 'North-\nWest', 'South\n-East', 'South-\nWest', 'Tropical']

dico_regions = {
'AL': 'South-East', 'AK': 'North', 'AZ': 'South-West', 'AR': 'Center', 'CA': 'South-West',
'CO': 'Center','CT': 'North-East','DE': 'North-East','DC': 'North-East','FL': 'South-East',
'GA': 'South-East','HI': 'Tropical','ID': 'North-West','IL': 'Center','IN': 'North-East',
'IA': 'Center','KS': 'Center','KY': 'East','LA': 'South-East','ME': 'North-East',
'MD': 'North-East','MA': 'North-East','MI': 'North-East','MN': 'North','MS': 'South-East',
'MO': 'Center','MT': 'North-West','NE': 'Center','NV': 'West','NH': 'North-East',
'NJ': 'North-East','NM': 'South','NY': 'North-East','NC': 'East','ND': 'North','OH': 'North-East',
'OK': 'South','OR': 'North-West','PA': 'North-East','PR': 'Tropical','RI': 'North-East',
'SC': 'East','SD': 'North','TN': 'East','TX': 'South','UT': 'West','VT': 'North-East','VA': 'East',
'WA': 'North-West','WV': 'North-East','WI': 'North','WY': 'North-West'}
df_regions = pd.DataFrame(dico_regions.items(), columns=['State', 'Region'])

# ------------ Colors
color_fire = 'firebrick'
color_surf = '#B2226A'
color_dura = '#B26A22'
categories_palette = ['#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026',
    '#800026', "darkorange", "gold", "goldenrod", "lemonchiffon", "cornsilk"]
month_colors = ['#80cdc1', '#80cdc1', '#018571', '#018571', '#018571', '#dfc27d',
              '#dfc27d', '#dfc27d', '#a6611a', '#a6611a', '#a6611a', '#80cdc1']
days_colors = ['firebrick', 'firebrick','firebrick','firebrick','firebrick','indianred','indianred']
causes_color = ['#332288', '#44AA99', '#AA3377', '#CCBB44', 'grey']
causes_color_rgb = ['[51, 34, 136, 120]', '[68, 170, 153, 120]', '[170, 51, 119, 120]',
    '[204, 187, 68, 120]', '[140, 140, 140, 120]']

seasons_colors = ['#80cdc1', '#018571', '#dfc27d', '#a6611a']
dico_causes = dict([(key, str(value) ) for key, value in zip(causes_labels, range(6))])
dico_causes_colors = dict([(key, value) for key, value in zip(causes_labels, causes_color)])
regions_colors = ['#ffdf87', '#7a3600', '#d39833', '#fbc591', '#efe89f',
'#ffd951', '#ffa44d', '#ffbb5c', '#bf6108', '#fd6628']
dico_regions_order = dict([(key, str(value) ) for key, value in zip(regions_list, range(10))])
dico_regions_colors = dict([(key, value) for key, value in zip(regions_list, regions_colors)])

# -------------- Global plot parameters
sns.set(rc={'axes.facecolor':(235/255, 230/255, 188/255, 1),
            'figure.facecolor':(235/255, 230/255, 188/255, 1)})
plt.style.use('default')
plt.rcParams.update({'font.size': 9})


# ------------------------------------------
# ---------------------------- Fonctions
# ------------------------------------------
@st.cache
def load_data(data_filename):
    data = pd.read_csv(data_filename, index_col = 0)
    data['DISCOVERY_DATE'] = pd.to_datetime(data['DISCOVERY_DATE'])
    data['DISC_DOY'] = data['DISCOVERY_DATE'].dt.dayofyear
    data['Region'] = [dico_regions[x] for x in data.STATE]
    data.rename(columns = {'LATITUDE':'lat', 'LONGITUDE':'lon'}, inplace = True)
    data.drop(['COUNTY', 'OWNER_DESCR', 'NWCG_CAUSE_AGE_CATEGORY',
        'NWCG_REPORTING_AGENCY', 'geometry'], axis = 1, inplace = True)
    return data
# ------------------------------------ Countplots
def make_countplot(data, x, title = '',
    xtitle ='', ytitle = 'Number of \nevents',
    x_rot = 0, rm_legend = False,
    color_plot = None, palette = None,
    order = None, xlabels = None,
    hue = None, hue_order = None, edgecolor = 'black',
    linewidth = 0.8, width = 8, height = 2.5):
    fig, ax = plt.subplots(figsize = (width, height))
    sns.countplot(x = x, data = data,
            hue = hue, hue_order = hue_order, order = order, edgecolor = edgecolor,
            color = color_plot, palette = palette, linewidth = linewidth,
            ax = ax)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title, y = 1.1)
    ax.tick_params(axis='x', labelrotation= x_rot)
    if xlabels :
        ax.set_xticklabels(labels = xlabels, rotation = x_rot, ha = 'center')
    if rm_legend :
        ax.get_legend().remove()
    else :
        plt.legend(ncol=2, title = '', fontsize = 9)
    return(fig)


def make_countplot_with_annot(data, x, title = '',
    xtitle ='', ytitle = 'Number of \nevents',
    x_rot = 0, xlabels = None,
    color_plot = None, palette = None,
    order = None, linewidth = 0.8, edgecolor = 'black',
    width = 8, height = 2.5, rm_legend = False):
# Work only for dataframe with CAUSE
    fig, ax = plt.subplots(figsize = (width, height))
    sns.countplot(x = x, data = data,
            order = order, edgecolor = edgecolor,
            color = color_plot, palette = palette, linewidth = linewidth,
            ax = ax)
    if xlabels :
        ax.set_xticklabels(labels = xlabels, rotation = x_rot, ha = 'center')
    ax.set_xlabel('')
    ax.set_ylim(0, max(data.CAUSE.value_counts()) + max(data.CAUSE.value_counts())*0.3)
    ax.set_ylabel(ytitle)
    ax.set_title(title, y = 1.1);
    for i in range(len(order)) :
        ax.annotate(str(round((data[x] == order[i]).sum() *100/data.shape[0], 1) ) + '%',
            xy = (i, (data[x] == order[i]).sum() + max(data.CAUSE.value_counts())*0.1),
            ha = 'center' )
    if rm_legend :
        ax.get_legend().remove()
    return(fig)

# ------------------------------------ Boxplot
def make_boxplot(data, x, y, title = '',
    xtitle ='', ytitle = 'Number of \nevents',
    x_rot = 0, xlabels = None,
    color_plot = None, palette = None,
    hue = None, hue_order = None,
    width = 8, height = 2.5):
    fig, ax = plt.subplots(figsize = (width, height))
    sns.boxplot(x = x, y = y,
            data = data,
            hue = hue, hue_order = hue_order,
            color = color_plot, palette = palette,
            ax = ax)
    if xlabels :
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelrotation= x_rot)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    plt.title(title, y = 1.1)
    return(fig)

# ------------------------------------ Barplot
def make_barplot(data, x, y, title = '',
    xtitle = '', x_rot = 0, order = None,
    hue = None, hue_order = None,
    xlabels = None, ytitle = '',
    palette = None, color_plot = None, linewidth = 0.8,
    errcolor='.26', errwidth=None, edgecolor = 'black',
    width = 8, height = 2.5, rm_legend = False, ncol = 3):
    fig, ax = plt.subplots(figsize = (8, 2.5))
    sns.barplot( x = x, y = y,
        data = data, order = order,
        hue = hue, hue_order = hue_order,
        color = color_plot, palette = palette, linewidth = linewidth,
        errcolor = errcolor, errwidth = errwidth,
        ax = ax)
    if xlabels :
        ax.set_xticklabels(xlabels)
    if rm_legend :
        ax.get_legend().remove()
    else :
        plt.legend(ncol=ncol, bbox_to_anchor=(1, -0.2))
    ax.tick_params(axis='x', labelrotation= x_rot)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    plt.title(title, y = 1.1)
    return(fig)

def ridgeplot(data, title = 'Distribution of wildfires \nalong a year') :
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(data.sort_values(by = 'DISC_YEAR'),
                      row = 'DISC_YEAR', aspect=10, height=0.4)
    g.map(sns.kdeplot, 'DISC_DOY',
          bw_adjust=1, clip_on=True, color = '#BCC6D1',
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, 'DISC_DOY',
          bw_adjust=1, clip_on=False,
          color="w", lw=3) # White contour
    g.map(plt.axhline, y=0, color =  '#BCC6D1',
          lw=2, clip_on=False)
    g.fig.subplots_adjust(hspace=-0.5)
    for i, ax in enumerate(g.axes.flat):
        ax.text(-60, 0.0005,
                data.sort_values(by = 'DISC_YEAR').DISC_YEAR.unique()[i],
                fontweight='bold', fontsize=15,
                color= 'grey')
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    axes = g.axes.flatten()
    for ax in axes:
        ax.set_ylabel("")
    plt.xlim(-5, 365)
    plt.xticks(ticks = [0, 181, 360], labels=['Jan', 'Jun','Dec'])
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.xlabel('', fontsize=15)
    plt.figure(figsize=(10, 3))
    g.fig.suptitle(title,
                   ha='center',
                   y=1.05,
                   fontsize=18, fontweight='bold')
    return(g)

# ------------------------------------ Lineplot
def make_lineplot(data, x, y, title = '',
    xtitle ='', ytitle = 'Number of \nevents',
    x_rot = 0, xlabels = None,
    color_plot = None, palette = None,
    marker = 'o', hue = None, hue_order = None,
    width = 8, height = 2.5):
    fig, ax = plt.subplots(figsize = (width, height))
    sns.lineplot(x = x, y = y,
            data = data, hue = hue, hue_order = hue_order, marker = marker,
            color = color_plot, palette = palette,
            ax = ax)
    if xlabels :
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelrotation= x_rot)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    plt.legend(ncol=3, bbox_to_anchor=(1, -0.2))
    plt.title(title, y = 1.1)
    return(fig)


# ------------------------------------------
#---------------------------- Dataframe Import
# ------------------------------------------

if st.checkbox('Use sample data', value = True):
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    df_fires = load_data(data_filename)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text("Done! (using st.cache)")
    st.markdown("### Warning : You're using a sample file that contains 5% of the complete dataset.")
    st.markdown("Please refer to our github repository to generate the complete dataset.")
    st.markdown("Or you can downloaded it at : \
            [https://drive.google.com/file/d/1JeuQ8Rx41JkJtYfPXnt4Wsi8TUD9OOJ7/view?usp=sharing](https://drive.google.com/file/d/1JeuQ8Rx41JkJtYfPXnt4Wsi8TUD9OOJ7/view?usp=sharing)")

else :
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data_load_state = st.text('Loading data...')
        df_fires = load_data(uploaded_file)
        data_load_state.text("Done! (using st.cache)")
    else :
        data_load_state = st.text('Loading data...')
        df_fires = load_data(data_filename)
        # Notify the reader that the data was successfully loaded.
        data_load_state.text("Done! (using st.cache)")
        st.markdown("### Warning : You're using a sample file that contains 5% of the complete dataset.")
        st.markdown("Please refer to our github repository to generate the complete dataset.")
        st.markdown("Or you can downloaded it at : \
            [https://drive.google.com/file/d/1JeuQ8Rx41JkJtYfPXnt4Wsi8TUD9OOJ7/view?usp=sharing](https://drive.google.com/file/d/1JeuQ8Rx41JkJtYfPXnt4Wsi8TUD9OOJ7/view?usp=sharing)")

# ------------------------------------------
# -------------------------- Create Sub-datframe used for plots later
# ------------------------------------------
# --- For global analysis
fires_months_tmp_df = df_fires.groupby( [ 'DISC_YEAR', 'DISC_MONTH' ],
    as_index=False).agg({'STATE' : 'count'})
fires_days_tmp_df = df_fires.groupby( ['DISC_DOW', 'DISC_YEAR'],
    as_index=False).agg({ 'STATE' : 'count' })

surface_fires_tmp = pd.DataFrame(df_fires.groupby('DISC_YEAR', as_index=False).agg({'FIRE_SIZE': 'mean'}))
surface = df_fires.groupby('DISC_YEAR', as_index=False)['FIRE_SIZE'].sum()
surface_total_state = df_fires.groupby(['DISC_YEAR', 'STATE', 'STATE_FULL'], as_index=False)['FIRE_SIZE'].sum()
surface_total_state.columns = ['year', 'St', 'State', 'Total burnt area (ha)']
surface_avg_state = df_fires.groupby(['DISC_YEAR', 'STATE', 'STATE_FULL'],
                        as_index=False)['FIRE_SIZE'].mean().groupby(['STATE', 'STATE_FULL'],
                            as_index = False)['FIRE_SIZE'].mean()
surface_avg_state.columns = ['St', 'State', 'Avg burnt area (ha)']

## --- For Barplot with confidence interval
surface_months = df_fires.groupby(['DISC_MONTH'],as_index=False).agg({'FIRE_SIZE':['mean', 'std', 'count']})
surface_months.columns = ['DISC_MONTH', 'FIRE_SIZE_avg', 'FIRE_SIZE_std', 'FIRE_SIZE_count']
surface_months['DISC_MONTH'] = months_labels
surface_months['conf_int'] = 1.96*np.divide(surface_months['FIRE_SIZE_std'],
    np.sqrt(surface_months['FIRE_SIZE_count']))

duration_months = df_fires.groupby(['DISC_MONTH'],as_index=False).agg({'DURATION':['mean', 'std', 'count']})
duration_months.columns = ['DISC_MONTH', 'DURATION_avg', 'DURATION_std', 'DURATION_count']
duration_months['DISC_MONTH'] = months_labels
duration_months['conf_int'] = 1.96*np.divide(duration_months['DURATION_std'],
    np.sqrt(duration_months['DURATION_count']))

duration_months_cause = df_fires.groupby(['DISC_MONTH', 'CAUSE'],as_index=False).agg({'DURATION':['mean', 'std', 'count']})
duration_months_cause.columns = ['DISC_MONTH', 'CAUSE', 'DURATION_avg', 'DURATION_std', 'DURATION_count']
duration_months_cause['DISC_MONTH'] = [j for j in months_labels for i in range(5)]
duration_months_cause['conf_int'] = 1.96*np.divide(duration_months_cause['DURATION_std'],
    np.sqrt(duration_months_cause['DURATION_count']))
surface_months_cause = df_fires.groupby(['DISC_MONTH', 'CAUSE'],as_index=False).agg({'FIRE_SIZE':['mean', 'std', 'count']})
surface_months_cause.columns = ['DISC_MONTH', 'CAUSE', 'FIRE_SIZE_avg', 'FIRE_SIZE_std', 'FIRE_SIZE_count']
surface_months_cause['DISC_MONTH'] = [j for j in months_labels for i in range(5)]
surface_months_cause['conf_int'] = 1.96*np.divide(surface_months_cause['FIRE_SIZE_std'],
    np.sqrt(surface_months_cause['FIRE_SIZE_count']))

cause_month_year = df_fires.groupby(['DISC_YEAR', 'DISC_MONTH', 'CAUSE'],
                                  as_index = False).agg(
                                  {'STATE' : 'count'}).groupby(
                                  ['DISC_MONTH', 'CAUSE'], as_index = False).agg({'STATE' : ['mean', 'std', 'count']})
cause_month_year.columns = ['DISC_MONTH', 'CAUSE', 'N_avg', 'N_std', 'N_count']
cause_month_year['DISC_MONTH'] = [j for j in months_labels for i in range(5)]
cause_month_year['conf_int'] = 1.96*np.divide(cause_month_year['N_std'],
    np.sqrt(cause_month_year['N_count']))

cause_human_month_year = df_fires[df_fires['CAUSE'] == 'Individuals\' mistake'].groupby(
    ['DISC_YEAR', 'DISC_MONTH', 'NWCG_GENERAL_CAUSE'], as_index = False).agg({'STATE' : 'count'}).groupby(
                                  ['DISC_MONTH', 'NWCG_GENERAL_CAUSE'], as_index = False).agg({'STATE' : ['mean', 'std', 'count']})
cause_human_month_year.columns = ['DISC_MONTH', 'NWCG_GENERAL_CAUSE', 'N_avg', 'N_std', 'N_count']
cause_human_month_year['DISC_MONTH'] = [j for j in months_labels for i in range(8)]
cause_human_month_year['conf_int'] = 1.96*np.divide(cause_human_month_year['N_std'],
    np.sqrt(cause_human_month_year['N_count']))





surface_avg = df_fires.groupby(['DISC_YEAR','CAUSE'], as_index=False)['FIRE_SIZE'].mean()
surface_avg_year = df_fires.groupby(['DISC_YEAR','CAUSE'],as_index=False).agg({'FIRE_SIZE': 'mean'})
duration_avg_state = df_fires.groupby(['DISC_YEAR', 'STATE','STATE_FULL'],
    as_index=False).agg({'DURATION':'mean'}).groupby(['STATE','STATE_FULL'], as_index=False)['DURATION'].mean()
duration_avg_state.columns=['St', 'State', 'Avg duration of a fire (days)']
duration_year_state = df_fires.groupby(['DISC_YEAR', 'STATE','STATE_FULL'],
    as_index=False).agg({'DURATION':'mean'})
duration_year_state.columns=['Year','St', 'State', 'Avg duration of a fire (days)']
months_cause = df_fires.groupby(['DISC_MONTH','CAUSE'],as_index=False).agg({'FIRE_SIZE':'mean'})
months_cause_total = df_fires.groupby(['DISC_MONTH', 'CAUSE'],
                                 as_index=False).agg({'FIRE_SIZE':'sum'})
months_year_total = df_fires.groupby(['DISC_MONTH', 'DISC_YEAR'],
                                 as_index=False).agg({'FIRE_SIZE':'sum'})
day_size = df_fires.groupby('DISC_DOW', as_index=False).agg({'FIRE_SIZE':'mean'})
day_size_cause = df_fires.groupby(['DISC_DOW','CAUSE'], as_index=False).agg({'FIRE_SIZE':'mean'}).set_index('DISC_DOW')
weekday_size_cause = pd.pivot_table(data=day_size_cause,
                                  index=day_size_cause.index, columns='CAUSE',
                                  values='FIRE_SIZE', aggfunc='mean')
duration_causes=df_fires.groupby(['DISC_YEAR','CAUSE'], as_index=False)['DURATION'].mean()


causes_year=pd.crosstab(df_fires['DISC_YEAR'], df_fires['CAUSE']).stack().reset_index().rename(columns={'DISC_YEAR':'Year', 'CAUSE':'cause', 0:'count'})
ct_classe_cause = pd.crosstab(df_fires.FIRE_SIZE_CLASS, df_fires.CAUSE)
ct_classe_cause_perc = ct_classe_cause.apply(lambda x : (x/x.sum()) *100, axis  = 1)
ct_classe_cause_perc = ct_classe_cause_perc[causes_labels]


# --- For state analysis
state_year_tmp_df = df_fires.groupby(['STATE', 'STATE_FULL', 'DISC_YEAR'],
    as_index=False).agg({ 'FPA_ID' : 'count', 'lat' : 'mean', 'lon' : 'mean', 'FIRE_SIZE' : 'sum'})
state_year_tmp_df.columns = ['St', 'State', 'Year', 'Number of fires', 'lat', 'lon', 'Surf']
state_year_avg_df = state_year_tmp_df.groupby(['State', 'St'],
    as_index=False).agg({'Number of fires' : 'mean'})
# --- For region anlysis
region_cause_df = pd.crosstab(df_fires['Region'], df_fires['CAUSE'])
region_cause_df['total']=region_cause_df.sum(axis=1)
for col in region_cause_df.columns:
    region_cause_df[col]=region_cause_df[col]/region_cause_df['total']*100
region_cause_df.drop('total', axis=1, inplace=True)
region_fire_number = pd.crosstab(df_fires['DISC_YEAR'], df_fires['Region'])


# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
#---------------------------- Intro text

st.markdown("#### How have fires evolved in the United States since the early 1990s ?")
st.markdown(" &emsp; _In recent years, the media coverage of major fires in the USA, Australia and on the \
    shores of the Mediterranean has highlighted the increase in wildfires, while linking them \
    directly to the climate change occurring everywhere on the planet. When analyzing the data \
    provided by Karen Short, we expected to observe a \
    fairly simple phenomenon of increase - or decrease - of these fires on the US s territory._")
st.markdown(" &emsp;_But the first explorations we made showed nothing of the kind: they put us on the track \
    of more complex mutations. So we built our study in the form of an investigation: \
    do these data collected over 27 years show a worsening of forest fires in the USA, \
    if so, how does it manifest itself?_")

st.caption("Github : [https://github.com/DataScientest-Studio/Pyromaniacs](https://github.com/DataScientest-Studio/Pyromaniacs)")

#---------------------------- Check table is correctly loaded
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df_fires.head())

genre = st.radio(
     "What kind of analysis to you want to perform ?",
     ('Global', 'Regional', 'By State', 'Our process'))

# ---------------------------------------------
# ---------------------------- Plots on full US
#----------------------------------------------
if genre == 'Our process':
    st.header('The project')
    st.markdown("We embarked on this study of wildfires in the United States with ambitious goals: highlight the existence of a correlation between the development of fires in the country and the changes in climate that have been taking place over the last twenty years. We planned to collect weather and climate data that would allow us to model this relationship, and even to anticipate the geography of increasing risks in the coming years.")
    st.markdown("We quickly encountered predictable obstacles: the unavailability of the necessary data, the lack of technical knowledge (especially in climate and weather sciences), unsufficient time to explore further the data banks of the American observatories and to sort the information collected in thousands of stations...).")
    st.markdown("We also quickly noticed that the evolution of fires was not monolithic and homogeneous: there are not *more* fires, at least not everywhere, however fires change, their causes evolve, the damage caused expands in some states, the seasonality of fires extends... We then redefined our project in a more realistic way, as an investigation : **does the data support the hypothesis of increased vulnerability of the US to wildfires, and if so, how does it take shape ?**")



    st.header('The dataset')
    st.markdown("We worked with a large dataset that puts together information collected from 1992 to 2018 on forest fires in the USA by local authorities of all kinds (police, firefighters, forest or park administrations...). The dataset has been made available by the Federal Department of Agriculture, and compiled by K.C Short under the following reference  : Short, K. C., *Spatial wildfire occurrence data for the United States*, 1992-2018, [FPA_FOD_20210617] (5th Edition)")
    st.write('The initial data set consisted of 2,166,753 rows and 37 columns.')

    st.markdown('**We selected 15 variables, and dropped 22.**')
    pertinentes=pertinentes=pd.DataFrame({'Nom Variable': {0: 'FPA_ID',
  1: 'NWCG_REPORTING_AGENCY',
  2: 'DISCOVERY_DATE',
  3: 'DISCOVERY_TIME',
  4: 'NWCG_CAUSE_CLASSIFICATION',
  5: 'NWCG_GENERAL_CAUSE',
  6: 'NWCG_CAUSE_AGE_CATEGORY',
  7: 'CONT_DATE',
  8: 'CONT_TIME',
  9: 'FIRE_SIZE',
  10: 'FIRE_SIZE_CLASS',
  11: 'LATITUDE',
  12: 'LONGITUDE',
  13: 'OWNER_DESCR',
  14: 'STATE',
  15: 'COUNTY'},
 '% NAs': {0: 0.0,
  1: 0.0,
  2: 0.0,
  3: 0.3482,
  4: 0.0,
  5: 0.0,
  6: 0.966,
  7: 0.3944,
  8: 0.4307,
  9: 0.0,
  10: 0.0,
  11: 0.0,
  12: 0.0,
  13: 0.0,
  14: 0.0,
  15: 0.3033},
 'Description': {0: 'ID unique',
  1: 'Info',
  2: 'Date de découverte',
  3: 'Heure de découverte',
  4: 'Classification cause',
  5: 'Classification cause',
  6: 'Classification cause',
  7: 'Date de fin de feu',
  8: 'Heure de fin',
  9: 'Acres brûlées',
  10: 'Classe du feu (en fonction de sa taille)',
  11: 'LATITUDE',
  12: 'LONGITUDE',
  13: 'Info entrée',
  14: 'État',
  15: 'Comté'},
 'Pertinence': {0: "à conserver s'il faut retracer l'entrée d'origine",
  1: "Service ayant rapporté l'accident",
  2: 'Utile pour classer les feux',
  3: 'Utile pour classer les feux',
  4: 'Utile pour classer les feux',
  5: 'Utile pour classer les feux',
  6: 'Utile pour classer les feux',
  7: 'Utile pour classer les feux',
  8: 'Utile pour classer les feux',
  9: 'Utile pour classer les feux',
  10: 'Utile pour classer les feux',
  11: 'Utile pour mapper les feux',
  12: 'Utile pour mapper les feux',
  13: 'Description responsable de la zone',
  14: 'Utile pour classer les feux',
  15: 'Utile pour classer les feux'}})
    if st.button('The selected variables') :
        st.dataframe(pertinentes)

    non_pertinentes=pd.DataFrame({'Nom Variable': {0: 'FOD_ID',
  1: 'SOURCE_SYSTEM_TYPE',
  2: 'SOURCE_SYSTEM',
  3: 'NWCG_REPORTING_UNIT_ID',
  4: 'NWCG_REPORTING_UNIT_NAME',
  5: 'SOURCE_REPORTING_UNIT',
  6: 'SOURCE_REPORTING_UNIT_NAME',
  7: 'LOCAL_FIRE_REPORT_ID',
  8: 'LOCAL_INCIDENT_ID',
  9: 'FIRE_YEAR',
  10: 'FIRE_CODE',
  11: 'FIRE_NAME',
  12: 'ICS_209_PLUS_INCIDENT_JOIN_ID',
  13: 'ICS_209_PLUS_COMPLEX_JOIN_ID',
  14: 'MTBS_ID',
  15: 'MTBS_FIRE_NAME',
  16: 'COMPLEX_NAME',
  17: 'DISCOVERY_DOY',
  18: 'CONT_DOY',
  19: 'FIPS_CODE',
  20: 'FIPS_NAME'},
 '% NAs': {0: 0.0,
  1: 0.0,
  2: 0.0,
  3: 0.0,
  4: 0.0,
  5: 0.0,
  6: 0.0,
  7: 0.7854,
  8: 0.3392,
  9: 0.0,
  10: 0.8294,
  11: 0.4433,
  12: 0.9853,
  13: 0.9996,
  14: 0.994,
  15: 0.994,
  16: 0.9974,
  17: 0.0,
  18: 0.3944,
  19: 0.3033,
  20: 0.3033},
 'Description': {0: 'Id',
  1: 'Info entrée',
  2: 'Info entrée',
  3: 'Info entrée',
  4: 'Info entrée',
  5: 'Info entrée',
  6: 'Info entrée',
  7: 'Info entrée',
  8: 'Info entrée',
  9: 'Année',
  10: 'Info entrée',
  11: 'Info entrée',
  12: 'Info entrée',
  13: 'Info entrée',
  14: 'Info entrée',
  15: 'Info entrée',
  16: 'Info entrée',
  17: "Jour de l'année",
  18: "Jour de l'année",
  19: 'Code ID du comté',
  20: 'Nom du comté dans le rapport FIPS'},
 'Non Pertinence': {0: 'Redondant avec FPA_ID',
  1: 'Redondant avec NWCG_REPORTING_AGENCY',
  2: 'Redondant avec NWCG_REPORTING_AGENCY',
  3: 'Redondant avec NWCG_REPORTING_AGENCY',
  4: 'Redondant avec NWCG_REPORTING_AGENCY',
  5: 'Redondant avec NWCG_REPORTING_AGENCY',
  6: 'Redondant avec NWCG_REPORTING_AGENCY',
  7: 'Trop de NAs',
  8: 'Trop de NAs',
  9: 'Redondant avec DISCOVERY_DATE_YEAR',
  10: "Trop de NAs + inutile pour l'analyse",
  11: "Trop de NAs + inutile pour l'analyse",
  12: "Trop de NAs + inutile pour l'analyse",
  13: "Trop de NAs + inutile pour l'analyse",
  14: "Trop de NAs + inutile pour l'analyse",
  15: "Trop de NAs + inutile pour l'analyse",
  16: "Trop de NAs + inutile pour l'analyse",
  17: 'Redondant avec DATE',
  18: 'Redondant avec CONT_DATE',
  19: 'Redondant avec Comté',
  20: 'Redondant avec Comté'}})
    if st.button("The variables we have not selected"):
        st.dataframe(non_pertinentes)

    st.header('Data processing')
    st.caption('We made 7 variables up.')
    createdvars=pd.DataFrame({'Nom Variable': {0: 'DISC_YEAR',
  1: 'DISC_MONTH',
  2: 'DISC_DAY',
  3: 'DISC_DOW',
  4: 'DURATION',
  5: 'Season',
  6: 'STATE_FULL'},
                          'Description':{0:'Année',
                                         1:'Mois',
                                         2:"Jour de l'année",
                                         3:'Jour de la semaine',
                                         4:'Durée du feu',
                                         5:'Saison du feu',
                                         6:"Nom complet de l'Etat"},
                          'Type':{0:'int',
                                  1:'int',
                                  2:'int',
                                  3:'int',
                                  4:'int',
                                  5:'string',
                                  6:'string'},
                          'Origine':{0:'Decomposition de la variable DISCOVERY_DATE',
                                    1:'Decomposition de la variable DISCOVERY_DATE',
                                    2:'Decomposition de la variable DISCOVERY_DATE',
                                    3:'Calcul à partir datetime.datetime.today().weekday()',
                                    4:'Date de containment - Date de déclaration',
                                    5:'Boucle',
                                    6:"importation d'un jeu externe"}
})
    if st.button('The variables we created'):
        st.dataframe(createdvars)

    st.subheader("Other transformations")
    st.markdown("""
    We also proceeded to the following processings :
    * Conversion of the FIRE_SIZE variable into hectares
    * Imputation of the median of the corresponding 'FIRE_SIZE_CLASS' category to the missing variables of the DURATION variable, which represented 39.4% of the data set.
    * Suppression of fires with a duration greater than 200 days, considering that it is very implausible.
    * Simplification of the variable CAUSE from 14 to 5 categories.
    """)

    st.markdown("**The final dataset has 2166369 rows and 22 columns.**")


    st.header("Additional data")
    st.write("We used external data at different stages of the research.")
    if st.button("View external data"):
        st.markdown("""
    * state capitals.csv : State capitals and their geographical coordinates
    * a shapefile including all the US States' silhouettes, for mapping purposes. It can be downloaded here : https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip
    * uscities.csv : population data for major U.S. cities
    * usstates.csv : a file with the name of each state
    """)
        st.markdown("""
    We also recovered and cleaned two climatological datasets, in the perspective of a correlation analysis, and possible modelling, of the relationships between fire and climate :
    * states_dcsi_clean.csv, about average precipitations from 2000 to 2018
    * states_temp_clean.csv, about average monthly temperatures for each State from 1996 to 2013
    """)




if genre == 'Global':
    # # -------------- What will be the variables to plot ?
    variables_columns = st.columns((5, 1, 1, 1, 1))
    option_main_variable = variables_columns[0].selectbox(
             'Which variable to plot ?',
             (fires_number, fires_causes, fires_surf, fires_dur))
    # # Dummy blank text to align checkboxes to selectbox
    variables_columns[1].markdown(f'<p style="color:#ffffff;font-size:14px;">{"text”"}</p>',
                unsafe_allow_html=True)

    if (option_main_variable == fires_surf) | (option_main_variable == fires_dur) :
        check_cause = variables_columns[1].checkbox('Causes')

    # -------------- Organize layout where plot will be
    # Space out the maps so the first one is 1.25x the size of the other one
    #----------------------------
    #----------------------------  Analysis of fire numbers
    if option_main_variable == fires_number :
        st.header('Overview of the wildfires from 1992 to 2018 -')
        c1, c2 = st.columns((1.25, 1))
        with c1 :
            map_type_fires = st.radio(
     "Map type :", ('Year by year', 'Average over the years'))
            if map_type_fires == 'Year by year' :
                fig = px.choropleth(
                    state_year_tmp_df,
                    locations='St',
                    color='Number of fires',
                    locationmode='USA-states',
                    color_continuous_scale='Reds',
                    range_color = [1, 15000],
                    animation_frame = 'Year',
                    hover_name = 'State',
                    hover_data = {'St' : False, 'Year' : False}
                )
                fig.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=state_year_avg_df['St'],    ###codes for states,
                    text=state_year_avg_df['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig.update_layout(
                    title={'text':'<b>Number of fires per state per year</b>', 'font':{'size':18}},
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                        margin=dict(
                            l=0, r=0, b=0, t=30, pad=2  )
                )
                st.plotly_chart(fig, use_container_width=True)
            else :
                fig2 = px.choropleth(
                    state_year_avg_df,
                    locations='St',
                    color='Number of fires',
                    locationmode='USA-states',
                    color_continuous_scale='Reds',
                    range_color = [1, 10000],
                    hover_name = 'State',
                    hover_data = {'St' : False}
                )
                fig2.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=state_year_avg_df['St'],    ###codes for states,
                    text=state_year_avg_df['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig2.update_layout(
                    title={'text':'<b>Average number of fires per state</b>', 'font':{'size':18}},
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                    margin=dict(l=0, r=0, b=0, t=30, pad=2)

                )
                st.plotly_chart(fig2, use_container_width=True)
        with c2 :
            fig = make_countplot(df_fires, 'DISC_YEAR',
                ytitle = 'Number of fires', xtitle ='', x_rot = 90, color_plot = color_fire)
            plt.title('Number of fires per year', fontsize=14, fontweight='bold')
            st.pyplot(fig, use_container_width=True) # SHOW THE FIGURE
            st.markdown("On the global US territory, \
                the number of fires didn't significately increase since 1992.\
                However, some years have been more affected than others \
                (2006 and 2011 for example).")
            fig = make_boxplot(fires_months_tmp_df, 'DISC_MONTH', 'STATE',
                xtitle = '', x_rot = 0, xlabels = months_labels,
                ytitle = 'Number of fires \nper year', palette = month_colors)
            plt.title("Average number of fires per month", fontsize=14, fontweight='bold')
            st.pyplot(fig, use_container_width=True) # SHOW THE FIGURE
            st.markdown("Wildfires are particularly abundant in the beginning of \
                spring and summer.")


    if option_main_variable == fires_surf :
        st.header('Analysis of the fire size')

        c1, c2 = st.columns((1.25, 1))
        with c1 :
            map_type_fires = st.radio(
     "Map type :", ('Year by year', 'Average over the years'))
            if map_type_fires == 'Year by year' :
                # df_subset = state_year_tmp_df[state_year_tmp_df.DISC_YEAR == year_plot]
                fig = px.choropleth(
                    surface_total_state,
                    locations='St',
                    color='Total burnt area (ha)',
                    locationmode='USA-states',
                    color_continuous_scale='RdPu',
                    range_color = [1, 400000],
                    animation_frame = 'year',
                    hover_name = 'State',
                    hover_data = {'St' : False, 'year' : False}
                )
                fig.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=surface_total_state['St'],    ###codes for states,
                    text=surface_total_state['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig.update_layout(
                    title_text='Number of fires per state per year',
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                        margin=dict(
                            l=0, r=0, b=0, t=30, pad=2  )
                )
                st.plotly_chart(fig, use_container_width=True)
            else :
                fig2 = px.choropleth(
                    surface_avg_state,
                    locations='St',
                    color='Avg burnt area (ha)',
                    locationmode='USA-states',
                    color_continuous_scale='YlOrBr',
                    range_color = [1, 250],
                    hover_name = 'State',
                    hover_data = {'St' : False}
                )
                fig2.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=surface_avg_state['St'],    ###codes for states,
                    text=surface_avg_state['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig2.update_layout(
                    title={'text':'<b>Average surface burnt per state per year</b>', 'font':{'size':18}},
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                    margin=dict(l=0, r=0, b=0, t=30, pad=2)

                )
                st.plotly_chart(fig2, use_container_width=True)
            fig = make_countplot(x='FIRE_SIZE_CLASS', data=df_fires,
                ytitle = 'Number of fires', xtitle = 'Fire class',
                title = 'Count of the fires based on their size category, 1992-2018',
                order=['A','B','C','D','E','F','G'],
                palette = categories_palette)
            st.pyplot(fig, use_container_width=True)
            st.caption("The Federal Administration of the USA classifies \
                the wildfire depending on their size in a 7-letter nomenclature : \
                \nA - less than 1000 m2 approx. ; \nB - between 1000 m2 and 4 ha approx. ;\
                 \nC - from 4 to 40 ha approx. ; \nD - from 40 to 120 ha approx. ; \nE - \
                 from 120 to 400 ha approx. ; \nF - from 400 to 2000 ha approx. ;\
                  \nG - more than 2000 ha approx..")
            st.markdown("The vast majority of fires are relatively small (less than 4 ha). \
                The very large fires are on the whole very few over \
                the observation period. Very large fires are very few in number over the observation period.")

        with c2:
            if check_cause :
                fig=px.bar(pd.DataFrame(ct_classe_cause_perc.stack()).reset_index().rename(columns={'FIRE_SIZE_CLASS':'class', 'CAUSE':'Cause',0:'%'}), x='class', y='%', color='Cause', color_discrete_sequence = causes_color,
                    template = 'simple_white')
                fig.update_layout(title_text='<b>Causes of fires for each fire size category<b>',
                    title_x=0.5, showlegend=False,
                    plot_bgcolor='white',font = dict(family= 'Helvetica', size= 15))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('The largest fires are by far triggered by lightings, whereas individual mistakes are less damaging.')
                fig = make_lineplot(x = 'DISC_YEAR', y = 'FIRE_SIZE',
                    data = surface_avg, hue = 'CAUSE',
                    palette = causes_color,
                    ytitle = 'Average damaged surface \nper fire (ha)',
                    title = 'Change in the average damage surface per year, 1992-2018')
                st.pyplot(fig, use_container_width=True)
                st.markdown('Lightning cause the most extensive fires throughout the study \
                    period, and this has been increasing. Next come technical accidents \
                    on infrastructures (which concern sparks from braking or mechanical \
                    failure, or from work on roads or railways).')


            else:
                fig = px.bar(
                    surface_months,
                    x= 'DISC_MONTH', y = 'FIRE_SIZE_avg', color = 'DISC_MONTH',
                    error_y='conf_int',
                    color_discrete_sequence = month_colors,
                    labels = {'DISC_MONTH' : '', 'FIRE_SIZE_avg' : 'Average burnt area (ha)'},
                    template = 'simple_white'
                )
                fig.update_layout(title_text='<b>Average damaged surface <br> of fires depending on the month<b>',
                    title_x=0.5, showlegend=False,
                    plot_bgcolor='white',
                    font = dict(family= 'Helvetica', size= 15) )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('It is during the summer period, especially in June, \
                    that fires are the most devastating in terms of area burned')
                # fig = make_barplot(months_year_total, 'DISC_MONTH', 'FIRE_SIZE',
                #     title = 'Total burnt surfaces for each month, 1992-2018',
                #     xtitle = '', x_rot = 0, ytitle = 'Total surface damaged per year (ha)',
                #     palette = month_colors, xlabels = months_labels)
                # st.pyplot(fig, use_container_width=True)
                # st.markdown('Nonetheless, when the total area burnt over the entire period is considered, \
                #     July is the most damaging month, as fires are more numerous, even though \
                #     their average area is smaller than in June.')
                fig, ax = plt.subplots(figsize = (8, 2.5))
                slope, intercept, r_value, p_value, std_err = stats.linregress(df_fires.groupby('DISC_YEAR',
                as_index=False).agg({'FIRE_SIZE': 'mean'}).DISC_YEAR,
                df_fires.groupby('DISC_YEAR', as_index=False).agg({'FIRE_SIZE': 'mean'}).FIRE_SIZE)
                line = slope*surface_fires_tmp.DISC_YEAR+intercept
                ax.plot(surface_fires_tmp['DISC_YEAR'] , surface_fires_tmp['FIRE_SIZE'],
                     c = color_surf, marker = 'o')
                plt.plot(surface_fires_tmp.DISC_YEAR, line, color = 'grey', linestyle = 'dotted', lw = 3,
                label='y = {:.2f}x{:.2f}'.format(slope,intercept))
                plt.ylabel('Average surface (ha)')
                plt.title('Average surface burned \nper fire (hectares)', y = 1.1);
                st.pyplot(fig, use_container_width=True)
                st.markdown('&emsp; The average area of a fire increases progressively throughout \
                the period despite significant annual variations; the regression line (grey) \
                confirms this trend.')
        if check_cause :
            fig = px.bar(
                    surface_months_cause,
                    x= 'DISC_MONTH', y = 'FIRE_SIZE_avg', color = 'CAUSE',
                    error_y='conf_int',
                    color_discrete_map = dico_causes_colors, category_orders = dico_causes,
                    labels = {'DISC_MONTH' : '', 'DURATION_avg' : 'Surface (ha)'},
                    template = 'simple_white'
                )
            fig.update_layout(title_text='<b>Surface fires depending on the month and the cause</b>',
                    title_x=0.5, showlegend=True, barmode = 'group',
                    plot_bgcolor='white',
                    font = dict(family= 'Helvetica', size= 15) )
            fig.update_yaxes(range=[0, 300])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("##### Wildfires caused by lightings are always the most extensive one.")


    if option_main_variable == fires_causes :
        st.header('What are the major causes of wildfires ?')
        c1, c2 = st.columns((1.5, 1))
        with c1 :
            st.markdown("""
            ##### The primary cause for fires in the US is individual negligence.
            It includes :
            * use of firearms
            * shooting of fireworks
            * burning of waste
            * individual mechanical accidents
            * cigarette butts
            * campfires or uncontrolled festive fires...
            Criminal causes are the second most important factor in the origin of fires.
            """)

        with c2 :
            fig = make_countplot_with_annot(df_fires, 'CAUSE',
                order = causes_labels, xlabels = causes_labels_split, height = 4,
                ytitle = 'Total number of fires', palette = causes_color)
            plt.title('Causes of wildfires, 1992-2018', fontsize=15, fontweight='bold')
            st.pyplot(fig, use_container_width=True)

        c1, c2 = st.columns((1.8, 1))
        with c1 :
            fig = make_lineplot(causes_year, x='Year', y='count', hue = 'cause', hue_order = causes_labels,
                                ytitle = 'Number of fires', x_rot = 0, palette= causes_color)
            plt.title('Evolution of the causes of wildfires from 1992 to 2018',
                fontsize=13, fontweight='bold')
            plt.legend(ncol=3, bbox_to_anchor=(0.9, -0.25))
            st.pyplot(fig, use_container_width=True)


        with c2 :
            st.markdown("##### The proportion of causes remains globally the same since 1992.")
            st.markdown('##### We clearly have a seasonality of wildfires causes.')
            st.markdown("&emsp; Natural causes, mainly lightning, largely dominate during summer, and are also very \
                important in June and September. Individual negligence is especially visible in the spring, \
                probably due to agricultural and forest fires, which is the high season.")

        c1,c2,c3=st.columns((0.01, 1, 0.01))
        with c2:
            fig = px.bar(
                cause_month_year,
                x= 'DISC_MONTH', y = 'N_avg', color = 'CAUSE',
                error_y='conf_int',
                color_discrete_map = dico_causes_colors, category_orders = dico_causes,
                labels = {'DISC_MONTH' : '', 'N_avg' : 'Number of fires'},
                template = 'simple_white'
            )
            fig.update_layout(title_text='<b>Number of fires depending of the cause and the month</b>',
                title_x=0.5, barmode = 'group',
                plot_bgcolor='white',
                font = dict(family= 'Helvetica', size= 15) )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('&emsp; We can have a closer look to the "Individuals\' mistake" category to better\
                understand the problematic baheviors (see below).')

            fig = px.bar(
                cause_human_month_year,
                x= 'DISC_MONTH', y = 'N_avg', color = 'NWCG_GENERAL_CAUSE',
                error_y='conf_int',
                labels = {'DISC_MONTH' : '', 'N_avg' : 'Number of fires'},
                template = 'simple_white'
            )
            fig.update_layout(title_text='<b>Number of fires depending of the cause and the month</b>',
                title_x=0.5, barmode = 'group',
                plot_bgcolor='white',
                font = dict(family= 'Helvetica', size= 15) )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("&emsp; Burning of garbage is particularly involved from February through May. \
                Festive fires are also more frequent in the summer, until September, and \
                fireworks are especially visible in July (the month of the national holiday).")


    if option_main_variable == fires_dur :
        st.header('Analysis of the fire duration')

        c1, c2 = st.columns((1.75, 1))
        with c2:
            if check_cause:
                fig=make_lineplot(duration_causes, 'DISC_YEAR', 'DURATION', xtitle ='', x_rot = 45,
                    palette = causes_color, marker = 'o', hue = 'CAUSE', hue_order = causes_labels, width = 9, height = 5)
                plt.title('Change in the fire duration \nover the period, depending on its cause',
                    fontsize=15, fontweight='bold')
                plt.ylabel('Avg duration', fontsize=12)
                plt.xlabel('')
                plt.xticks(range(1992,2019))
                st.pyplot(fig.figure)

                st.markdown("&emsp; It is visible that the lightnings cause fires that have been longer and longer since 1992; \
                    it may be due to the evolution of the soils and vegetation, increasingly dry over the years.")
                fig = px.bar(
                    duration_months_cause,
                    x= 'DISC_MONTH', y = 'DURATION_avg', color = 'CAUSE',
                    error_y='conf_int',
                    color_discrete_sequence = causes_color,
                    labels = {'DISC_MONTH' : '', 'DURATION_avg' : 'Duration (days)'},
                    template = 'simple_white'
                )
                fig.update_layout(title_text='<b>Duration of fires depending on the month and the cause</b>',
                    title_x=0.5, showlegend=False, barmode = 'group',
                    plot_bgcolor='white',
                    font = dict(family= 'Helvetica', size= 15) )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("##### Wildfires caused by lightings are always the longest ones to be contained.")

            else:
                duration_global=df_fires.groupby('DISC_YEAR', as_index=False)['DURATION'].mean()
                duration_month=df_fires.groupby('DISC_MONTH', as_index=False)['DURATION'].mean()
                fig=make_lineplot(duration_global, 'DISC_YEAR', 'DURATION', xtitle ='', ytitle = 'Avg duration',
                    x_rot = 45, xlabels =sorted(df_fires['DISC_YEAR'].unique()), color_plot = color_dura,
                    palette = None, marker = 'o', hue = None, width = 8, height = 2.5)
                plt.title('Change in the fire duration over the period',  fontsize=15, fontweight='bold')
                plt.xticks(range(1992,2018))
                st.pyplot(fig.figure)
                st.markdown("&emsp; ##### There has been a slow trend in the average duration of fires since the early 1990s. \
                    This is certainly one of the major signs of the worsening fire phenomenon in the USA.")
                fig = px.bar(
                    duration_months,
                    x= 'DISC_MONTH', y = 'DURATION_avg', color = 'DISC_MONTH',
                    error_y='conf_int',
                    color_discrete_sequence = month_colors,
                    labels = {'DISC_MONTH' : '', 'DURATION_avg' : 'Duration (days)'},
                    template = 'simple_white'
                )
                fig.update_layout(title_text='<b>Duration of fires depending on \nthe month</b>',
                    title_x=0.5, showlegend=False,
                    plot_bgcolor='white',
                    font = dict(family= 'Helvetica', size= 15) )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("##### The duration of the fires rises on average from the spring, and peaks in August.")


        with c1:
            map_type_fires = st.radio("Map type :", ('Year by year', 'Average over the years'))
            if map_type_fires== 'Year by year':
                fig = px.choropleth(
                    duration_year_state,
                    locations='St',
                    color='Avg duration of a fire (days)',
                    locationmode='USA-states',
                    color_continuous_scale='YlOrBr',
                    range_color = [0, 15],
                    animation_frame = 'Year',
                    hover_name = 'State',
                    hover_data = {'St' : False, 'Avg duration of a fire (days)' : True}
                )
                fig.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=duration_year_state['St'],
                    text=duration_year_state['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig.update_layout(
                    title={'text':'<b>Evolution of the average duration of <br> fires by state over the period </b>', 'font':{'size':18}},
                    legend_title_text='Avg duration <br> of a fire (days)',
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True,
                        lakecolor='rgb(255, 255, 255)'),
                        margin=dict(
                            l=0, r=0, b=0, t=30, pad=2  )
                )
                st.plotly_chart(fig, use_container_width=True)


            if map_type_fires == 'Average over the years' :
                fig = px.choropleth(
                    duration_avg_state,
                    locations='St',
                    color='Avg duration of a fire (days)',
                    locationmode='USA-states',
                    color_continuous_scale='YlOrBr',
                    range_color = [0, 8],
                    hover_name = 'State',
                    hover_data = {'St' : False}
                )
                fig.add_trace(go.Scattergeo(
                    locationmode = 'USA-states',
                    locations=duration_avg_state['St'],    ###codes for states,
                    text=duration_avg_state['St'],
                    hoverinfo = 'skip',
                    mode = 'text' )  )
                fig.update_layout(title={'text':'<b>Average duration <br> of a fire by State </b>', 'font':{'size':18}},
                    legend_title_text='Avg duration <br> of a fire (days)',
                    geo = dict(
                        scope='usa',
                        projection=go.layout.geo.Projection(type = 'albers usa'),
                        showlakes=True, # lakes
                        lakecolor='rgb(255, 255, 255)'),
                    margin=dict(l=0, r=0, b=0, t=30, pad=2)

                )
                st.plotly_chart(fig)
                st.markdown('Fires appear longest on average in the West, but this is even more noticeable in Alaska, \
                    which was affected by very long fires \
                    in 2004 and 2005, and then more recently, since 2015 and up to now.')



# ---------------------------------------------
# ---------------------------- Plots State by State
#----------------------------------------------
elif genre == 'By State' :
    st.markdown( "##### &emsp;This tool provides, for each U.S. state, a visualization\
        of the spatial distribution of fires over the period 1992-2018, as well as :")
    st.markdown("&emsp;&emsp;&emsp;&emsp; - The causes of these fires and their respective numbers, as well as the \
        damaged surface area according to each of these factors")
    st.markdown("&emsp;&emsp;&emsp;&emsp; - The evolution of the number of fires and the area burnt \
        (cumulative) over the years, as well as the distribution of these \
        fires by month of the year.", unsafe_allow_html=False)
    st.markdown("---")
    col_state, col_cause = st.columns((1, 1))
    with col_state:
        selected_state = st.selectbox(
         'Select the state you would like to analyse',
         [' -- Rankings -- '] + list(np.sort(df_fires.STATE_FULL.unique())) )
    if selected_state == ' -- Rankings -- ' :
        c_1, c_2 = st.columns((1, 1))
        with c_1 :
            f_number = make_barplot( state_year_tmp_df.sort_values( by = 'Number of fires', ascending = False),
                'Number of fires','State',
                xtitle = '', x_rot = 0,
                order = state_year_tmp_df.groupby( ['State']).agg(
                    {'Number of fires' : 'mean'}).sort_values(
                    'Number of fires', ascending = False).index,
                hue = None, hue_order = None,
                xlabels = None, ytitle = '',
                palette = None, color_plot = color_fire, linewidth = 0.8,
                errcolor='.5', errwidth=0.8)
            plt.title('Number of fires per state per year',
                fontsize=15, fontweight='bold', y = 1.02)
            f_number.set_figheight(8)
            st.pyplot( f_number, use_container_width=True )
        with c_2 :
            f_surf = make_barplot( state_year_tmp_df,
                'Surf','State',
                xtitle = '', x_rot = 0,
                order = state_year_tmp_df.groupby( ['State']).agg(
                    {'Surf' : 'mean'}).sort_values(
                    'Surf', ascending = False).index,
                hue = None, hue_order = None,
                xlabels = None, ytitle = '',
                palette = None, color_plot = color_surf, linewidth = 0.8,
                errcolor='.5', errwidth=0.8)
            plt.title('Total surface burnt (ha) per state per year',
                fontsize=15, fontweight='bold', y = 1.02)
            f_surf.set_figheight(8)
            st.pyplot( f_surf, use_container_width=True )

    else :
        df_sub = df_fires[(df_fires.STATE_FULL == selected_state)]
        df_sub_count=pd.crosstab(df_sub['DISC_YEAR'],
            df_sub['CAUSE']).stack().reset_index().rename(
            columns= {'DISC_YEAR':'Year', 'CAUSE':'cause', 0:'count'})
        state_abb = df_sub.STATE.unique()[0]
        state_nb_month = df_sub.groupby( [ 'DISC_YEAR', 'DISC_MONTH' ],
            as_index=False).agg({'FPA_ID' : 'count'})
        state_surf_year = df_sub.groupby( [ 'DISC_YEAR' ],
            as_index=False).agg({'FIRE_SIZE' : 'sum'})
        ct_cause_state_year = pd.crosstab(df_sub.DISC_YEAR, df_sub.CAUSE)
        ct_cause_state_year_perc = ct_cause_state_year.apply(lambda x : (x/x.sum()) *100, axis  = 1)
        with col_cause :
            cause_on = st.radio(
                "Do you want to visualize the data by separating the \
                information according to the cause of the fires ?",
                ('Yes', 'No'))
        st.header(selected_state)
        if cause_on == 'Yes' :
            c1, c2 = st.columns((1.5, 1)) # "Hide" the 3rd column
        else :
            c1, c2 = st.columns((1.5, 1))
        with c1 :
            map_type_fires = st.radio(
                "Map type :", ('Year by year', 'All years'))
            if map_type_fires == 'Year by year' :
                fig = px.scatter_geo(df_sub.sort_values(by = 'DISC_YEAR'),
                lat = 'lat',
                lon = 'lon',
                locationmode = 'USA-states',
                color = 'CAUSE',
                category_orders = dico_causes,
                color_discrete_map = dico_causes_colors,
                animation_frame = 'DISC_YEAR',
                size = 'FIRE_SIZE',
                size_max = 50,
                opacity = 0.8,
                )
                fig.update_layout(legend = dict(
                    title = '', yanchor="bottom", y=0.5,
                    xanchor="left", x=0),
                                  geo = dict( scope='usa',
                    projection=go.layout.geo.Projection(type = 'albers usa'),
                    showlakes=True, # lakes
                    lakecolor='rgb(255, 255, 255)'),
                    margin=dict(l=0, r=0, b=0, t=30, pad=2),
                    title={'text':'<b>Number of fires per year according to their cause</b>', 'font':{'size':18}}
                )
                fig.update_geos(fitbounds="locations")
                st.plotly_chart(fig, use_container_width=True)

            elif map_type_fires == 'All years' :
                if selected_state=='California':
                    image = Image.open('Pictures/California.png')
                    st.image(image)
                    if st.button('Create the interactive version'):
                        options_layers = st.multiselect( 'Which causes to add :', causes_labels, causes_labels)
                        layers = []
                        for lay in options_layers :
                            i = causes_labels.index(lay)
                            layers.append( pdk.Layer(
                                'ScatterplotLayer',
                                data=df_sub[df_sub.CAUSE == lay],
                                id = 'test',
                                opacity=0.5,
                                get_position='[lon, lat]',
                                get_color=causes_color_rgb[i],
                                get_radius=1500 ))
                        st.pydeck_chart(pdk.Deck(
                                map_style='mapbox://styles/mapbox/light-v9',
                                initial_view_state=pdk.ViewState(
                                    latitude=df_sub['lat'].mean(),
                                    longitude=df_sub['lon'].mean(),
                                    zoom=3,
                                    pitch=0,
                                ),layers= layers,
                            ))
                if selected_state=='Florida':
                    image = Image.open('Pictures/Florida.png')
                    st.image(image)
                    if st.button('Create the interactive version'):
                        options_layers2 = st.multiselect( 'Which causes to add :', causes_labels, causes_labels)
                        layers = []
                        for lay in options_layers2 :
                            i = causes_labels.index(lay)
                            layers.append( pdk.Layer(
                                'ScatterplotLayer',
                                data=df_sub[df_sub.CAUSE == lay],
                                id = 'test',
                                opacity=0.5,
                                get_position='[lon, lat]',
                                get_color=causes_color_rgb[i],
                                get_radius=1500 ))
                        st.pydeck_chart(pdk.Deck(
                                map_style='mapbox://styles/mapbox/light-v9',
                                initial_view_state=pdk.ViewState(
                                    latitude=df_sub['lat'].mean(),
                                    longitude=df_sub['lon'].mean(),
                                    zoom=3,
                                    pitch=0,
                                ),layers= layers,
                            ))
                
                else:
                    options_layers3 = st.multiselect( 'Which causes to add :',
                    causes_labels, causes_labels)
                    layers = []
                    for lay in options_layers3 :
                        i = causes_labels.index(lay)
                        layers.append( pdk.Layer(
                             'ScatterplotLayer',
                             data=df_sub[df_sub.CAUSE == lay],
                             id = 'test',
                            opacity=0.5,
                             get_position='[lon, lat]',
                             get_color=causes_color_rgb[i],
                             get_radius=2000 ))
                    st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
                            initial_view_state=pdk.ViewState(latitude=df_sub['lat'].mean(),
                                                             longitude=df_sub['lon'].mean(),zoom=3,pitch=0),
                            layers= layers,
                        ))
                

            if cause_on == 'Yes':
                f2 = make_lineplot(df_sub_count, x='Year', y='count',
                    hue = 'cause',
                    ytitle = '',
                    palette = causes_color,
                    width = 8, height = 2.5)
                plt.title('Number of fires per year in ' + selected_state +
                    "\ndepending of the cause", fontsize=15, fontweight='bold')
            else : # cause_on = No
                f1 = make_countplot( df_sub,
                'DISC_YEAR', xtitle ='', ytitle = '\n\n', x_rot = 90, color_plot = 'lightslategray')
                plt.title('Number of fires per year', fontsize=15, fontweight='bold')
                f2 = make_lineplot(state_surf_year,
                    'DISC_YEAR', 'FIRE_SIZE',
                    xtitle = '', x_rot = 45, 
                    ytitle = 'Total surface burnt\n(hectares)', color_plot='lightslategray')
                plt.title('Total surface burnt per year', fontsize=15, fontweight='bold')
                f3 = make_boxplot(state_nb_month,
                    'DISC_MONTH', 'FPA_ID',
                    xtitle = '', x_rot = 0, xlabels = months_labels,
                    ytitle = 'Number of fires \nper year', palette = month_colors)
                plt.title('Number of fires per month', fontsize=15, fontweight='bold')
                st.pyplot(f1, use_container_width=True)
                st.pyplot(f2, use_container_width=True)
                st.pyplot(f3, use_container_width=True)

        with c2:
            if cause_on == 'Yes' :
                f1 = make_countplot_with_annot(df_sub, 'CAUSE',
                    xtitle ='', ytitle = '\n\n', x_rot = 0,
                    order = causes_labels, xlabels = causes_labels_split, palette = causes_color)
                plt.title('Distribution of the wildfires causes', fontsize=15, fontweight='bold')
                st.pyplot(f1, use_container_width=True)
                f2 = make_lineplot(df_sub_count, x='Year', y='count',
                    hue = 'cause', hue_order= causes_labels,
                    ytitle = '',
                    palette = causes_color,
                    width = 8, height = 2.5)
                plt.title('Number of fires per year in ' + selected_state + "\ndepending of the cause", fontsize=15, fontweight='bold')
                st.pyplot(f2, use_container_width=True)
            else :
                f4 = ridgeplot(df_sub)
                st.pyplot(f4, use_container_width=True)
                st.markdown("&emsp;The plot above shows the density \
                    of the fires occurences along the year from 1992 to 2018. It offers a complementary \
                    perspective on the impacts of climate change: depending on the state, \
                    we can observe phenomena of lengthening of the fire season \
                    (for example in Puerto Rico, Colorado or Arizona), or on the contrary, \
                    a brutal radicalization of fires over one or two seasons (in Kansas)." )
                f5=make_lineplot(duration_year_state.loc[duration_year_state['State']==selected_state], x='Year', y='Avg duration of a fire (days)', x_rot = 45, marker = 'o', width = 9, height = 5, color_plot='#B26A22')
                plt.ylabel('Average duration (days)', fontsize=12)
                plt.xlabel('', fontsize=12)
                plt.xticks(range(1992,2018))
                plt.title('Change in the fire duration over the period', fontsize=15, fontweight='bold')
                st.pyplot(f5.figure)
                st.markdown("&emsp; It is obvious that fires have been longer and longer since 1992; \
                    it may be due to the evolution of the soils and vegetation, increasingly dry over the years.")

        if cause_on == 'Yes' :
            f3 = px.bar(df_sub.groupby(['DISC_YEAR', 'CAUSE'],
                as_index = False).agg({'FIRE_SIZE':'sum'}),
                x='DISC_YEAR', y='FIRE_SIZE',
                color = 'CAUSE',      color_discrete_map = dico_causes_colors, category_orders = dico_causes, template='simple_white', labels = {'DISC_YEAR' : '', 'FIRE_SIZE' : 'Damaged surface (ha)'}
                )
            f3.update_layout(title_text='<b>Total surface burnt per year depending of the cause<b>',
                    title_x=0.5, showlegend=True, barmode = 'group',
                    plot_bgcolor='white',
                    font = dict(family= 'Helvetica', size= 15))
            st.plotly_chart(f3, use_container_width=True)
            

elif genre == 'Regional':
    st.markdown("##### &emsp; Spatial analysis of fires reveals significant disparities in the causes, \
        and evolution, of fires across the US. We have distinguished **10 major regions**, \
        presented on the map below.")
    st.markdown('---')
    c1, c2 = st.columns((1, 1.5))
    with c1:
        fig = px.choropleth(
            df_regions,
            locations='State',
            locationmode='USA-states',
            color = 'Region',
            hover_name = 'State',
            color_discrete_map  = dico_regions_colors,
            category_orders = dico_regions_order,
        )
        fig.add_trace(go.Scattergeo(
            locationmode = 'USA-states',
            locations=df_regions.State,
            text=df_regions.State,
            hoverinfo = 'skip',
            mode = 'text' ) )
        fig.update_layout(
            title_text='Regions',
            geo = dict(
                scope='usa',
                projection=go.layout.geo.Projection(type = 'albers usa'),
                showlakes=True, # lakes
                lakecolor='rgb(255, 255, 255)'),
                margin=dict(
                    l=0, r=0, b=0, t=30, pad=2  )
        )
        st.plotly_chart(fig)
        st.markdown("&emsp;Fires do not have the same causes depending on whether you are in the east \
            or west of the country.\
            We can see that lightning is the main origin of fires in the West (Utah and Nevada). \
            It is also important in North-West. In these two regions, natural factors are dominant, \
            while individual negligence is the primary cause of fires in all the other parts of \
            the country.\
            Fires of criminal origin are particularly numerous in the eastern part: south-east, \
            north-east, east. The plot also shows that criminal behavior is less at stake in \
            western parts of the USA, and it is likely thanks to an increased awareness and \
            care from inhabitants and tourists, due to a particularly high ecosystemic sensitivity.\
            We also note that technical accidents on infrastructures are more often involved \
            in the southeast and south.texte evolution in number of fires per region")
        st.markdown("&emsp;The number of fires in the country has been evolving in different ways depending \
            on the State and the region.\
            In fact, in some parts of the country, fires are not increasing, contrary to the \
            impression created by the increase in mega-fires. For example, there are no more \
            fire starts in vulnerable western regions (California, Oregon, Arizona), even though \
            fires there are likely to be increasingly large and destructive.\
            On the other hand, between 1992 and 2018, the number of fires increased in \
            three regions: in the South (Texas, New Mexico, Oklahoma), in the Northeast \
            and in the center (Colorado, Kansas, Nebraska, Missouri...), two rather temperate \
            regions, which are not particularly dry. In these regions, individual negligence \
            is the leading cause of fires; more than particularly dangerous or imprudent behavior, \
            it is certainly the drying out of the soil and vegetation that increases vulnerability \
            to fires.")
    with c2:
        fig, ax = plt.subplots()
        region_cause_df.plot(
            kind='barh',
            stacked=True,
            color={'Criminal':'#44AA99',
            "Individuals' mistake":'#332288',
            'Infrastructure accident':'#AA3377',
            'Natural (lightning)':'#CCBB44',
            'Other/Unknown':'grey'},
            alpha = 0.8, ax = ax
        )
        plt.ylabel('')
        ax.tick_params(axis='both', color = 'black', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('% of total fires reported from 1992 to 2018', fontsize = 14, labelpad=15)
        plt.legend(labels=['Criminal',"Individuals'\nmistake",
                           "Infrastructure \naccident",
                           "Natural \n(lightning)",
                           'Other/Unknown'],
                   bbox_to_anchor=(1.02, 0.8),
                   fontsize = 14, ncol = 1 )
        plt.title('Wildfire origins depending on the region', fontsize=15);
        st.pyplot(fig, use_container_width=True)


        fig = px.line(region_fire_number.rolling(10).mean(),
        color_discrete_map = dico_regions_colors,
        template = 'simple_white',
        labels = {'value' : 'Number of fires', 'DISC_YEAR' : ''})
        fig.update_traces(line=dict(width=5))
        fig.update_xaxes(range=[2002, 2018])
        fig.update_yaxes(range=[0, 25000])
        fig.update_layout(title_text='<b> Evolution of the number of fires in the main regions </b>',
            title_x=0.5, legend=dict(font=dict(size= 14) ),
            plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

