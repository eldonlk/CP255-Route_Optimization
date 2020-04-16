from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, \
    Column, \
    Button, \
    DataTable, \
    TableColumn, \
    Row, \
    Slider, \
    DatePicker, \
    TextInput, \
    CustomJS, \
    Div, \
    HoverTool, \
    GeoJSONDataSource
from bokeh.palettes import Viridis
from bokeh.io import curdoc
from bokeh.events import DoubleTap
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import aux_functions as af
from bokeh.themes import built_in_themes
import numpy as np
import math
from itertools import cycle

#bokeh serve --show test2.py
# curdoc().theme = 'dark_minimal'

street_map = gpd.read_file('shp/geo_export_0a23a24b-8a75-4a43-b238-df5c9996dcf4.shp')
street_source = GeoJSONDataSource(geojson=street_map.to_json())

coordList=[]

source = ColumnDataSource(pd.DataFrame(data=dict(x=[], y=[])))
initpop = ColumnDataSource(pd.DataFrame({'init_pop' : [10]}))
numgens = ColumnDataSource(pd.DataFrame({'num_gens' : [5]}))
date = ColumnDataSource(pd.DataFrame({'date' : [datetime.today()]}))
hour = ColumnDataSource(pd.DataFrame({'hour' : [0]}))
minute = ColumnDataSource(pd.DataFrame({'minute' : [0]}))
output = ColumnDataSource(pd.DataFrame({'order' : [], 'total_time' : [], 'chance' : [], 'gen' : []}))
timeseries = ColumnDataSource(pd.DataFrame({'order' : [], 'chance' : [], 'gen' : []}))
# groups = ['1111', '2222', '3333', '4444', '5555']
# count = [3,3,3,3,3]
data_dict = {'x': [],'y': [], 'color': []}
source_table_hist = ColumnDataSource(data=data_dict)

TOOLS = "tap, wheel_zoom, reset"

p = figure(
           toolbar_location="above",
           tools=TOOLS,
           width=700,
           height=700,
           # x_range=(-74.5, -73.5),
           # y_range=(40, 41)
           )

p.multi_line('xs',
             'ys',
             source=street_source,
             color='gray',
             line_width=.5,
             alpha = .5)

p.title.text = "Route Optimizer"
p.title.text_font_size = "25px"
p.xaxis.axis_label = "Longitude"
p.yaxis.axis_label = "Latitude"

p.circle_cross(source = source,
               x ='x',
               y ='y',
               size=20,
               # color="#990F02",
               fill_alpha=0.2,
               line_width=2
               )

h = figure(x_range=data_dict['x'],
           height = 200,
           tools="hover",
           tooltips="@x: @y",
           title="Combination Counts")

h.vbar(x ='x',
       top ='y',
       width = .7,
       # fill_color="#990F02",
       # line_color='black',
       color='color',
       source=source_table_hist)

h.xaxis.major_label_orientation = math.pi/2
h.xaxis.axis_label = "Order"
h.yaxis.axis_label = "Count"

l = figure(title="Fitness",
           y_axis_type="linear",
           tools="hover",
           tooltips="@name",
           plot_height = 400)

l.xaxis.axis_label = 'Generation'
l.yaxis.axis_label = 'Fitness'


#add a dot where the click happened
def callback(event):
    Coords=(event.x, event.y)
    coordList.append(Coords)
    source.data = pd.DataFrame(dict(x=[i[0] for i in coordList], y=[i[1] for i in coordList]))

def solve():
    start_date = pd.DataFrame(date.data)['date'][0]#.to_pydatetime()
    start_hour = int(pd.DataFrame(hour.data)['hour'][0])
    start_minute = int(pd.DataFrame(minute.data)['minute'][0])
    start_date = start_date + timedelta(hours=start_hour) + timedelta(minutes=start_minute)
    init_pop = pd.DataFrame(initpop.data)['init_pop'][0]
    num_gens = pd.DataFrame(numgens.data)['num_gens'][0]
    points_df = pd.DataFrame(coordList)
    points_df.columns = ['latitude', 'longitude']
    start_loc = pd.DataFrame(points_df.loc[0]).transpose()
    visit_points = points_df.loc[1:]
    hold = af.get_all(start_loc,
                      visit_points,
                      start_date,
                      init_pop,
                      num_gens)
    # hold = af.get_all(test_start,
    #                   test_set,
    #                   test_date,
    #                   init_pop,
    #                   num_gens)
    #table data updates
    combos = hold.iloc[0:, :len(hold.columns) - 4]
    a = combos.columns
    hold['order'] = '0->' + combos[a].apply(lambda row: '->'.join((row.values+1).astype(str)), axis=1)
    output.data = hold[['order', 'total_time', 'chance', 'gen']]
    #line plotting update
    plot_order = af.condense(hold.tail(init_pop)).sort_values('total_time').iloc[0, 0:len(hold.columns) - 5]
    plot_dat = visit_points.reindex(plot_order + 1)
    fullplotdat = pd.DataFrame(start_loc.append(plot_dat))
    p.line(x = fullplotdat['latitude'],
           y = fullplotdat['longitude'],
           # color="#990F02",
           name = 'line')
    #histogram data update
    hist_data = hold.groupby('order').count().reset_index()
    print(hist_data)
    color_hold = []
    colors = cycle(Viridis[11])
    for i, color in zip(range(len(hist_data)), colors):
        color_hold.append(color)
    print(color_hold)
    groups = list(hist_data['order'])
    count = list(hist_data['gen'])
    data_dict = {'x': groups, 'y': count, 'color': color_hold}
    h.x_range.factors = data_dict['x']  # update existing range (good)
    source_table_hist.data = data_dict
    #line graph update
    line_data = hold.groupby(['order', 'gen']).sum().reset_index()[['order', 'gen', 'chance']]
    line_data = line_data.groupby('order').apply(lambda x: [list(x['chance']), list(x['gen'])]).apply(pd.Series).reset_index()
    line_data.columns = ['order', 'chance', 'gen']
    line_data['color'] = color_hold
    timeseries.data = line_data
    l.multi_line(xs ='gen',
                 ys ='chance',
                 legend = 'order',
                 line_width=5,
                 line_alpha=0.6,
                 hover_line_alpha=1.0,
                 color = 'color',
                 source = timeseries)
    l.legend.location = 'top_left'
    print('finished')


def update_popsize(attr, old, new):
    initpop.data = pd.DataFrame({'init_pop' : [new]})

def update_numgen(attr, old, new):
    numgens.data = pd.DataFrame({'num_gens' : [new]})

def update_date(attr, old, new):
    date.data = pd.DataFrame({'date': [new]})

def update_hour(attr, old, new):
    hour.data = pd.DataFrame({'hour': [new]})

def update_minute(attr, old, new):
    minute.data = pd.DataFrame({'minute': [new]})

# def update_time(attr, old, new):
#     print(new)
#     print(type(new))
#     time.data = pd.DataFrame({'time': [new]})
#     print(pd.DataFrame(time.data)['time'][0])

#data table
columns = [
        TableColumn(field='x', title='longitude'),
        TableColumn(field='y', title='latitude')
    ]
data_table = DataTable(source=source,
                       columns=columns,
                       width=700,
                       height=150)

columns2 = [
        TableColumn(field='order', title='order'),
        TableColumn(field='total_time', title='total time'),
        TableColumn(field='chance', title='chance'),
        TableColumn(field='gen', title='gen')
    ]

data_table_2 = DataTable(source=output, columns=columns2, width=700, height=280)

p.on_event(DoubleTap, callback)

button = Button(label="Optimize",
                # button_type="success"
                )
button.on_click(solve)

generation = Slider(start = 1, end = 4, value=1, step=1, title="Generations")
# generation.on_change('value', update)
population_size_input = Slider(start = 5, end = 100, value=10, step=1, title="Population Size")
population_size_input.on_change('value', update_popsize)

generation_input = Slider(start = 2, end = 100, value=5, step=1, title="Number of Generations")
generation_input.on_change('value', update_numgen)

dt_pckr = DatePicker(title='Start Date',
                     min_date=datetime(1900,1,1),
                     max_date=datetime.today().replace(year = datetime.today().year + 5))
dt_pckr.on_change('value', update_date)

hour_input = Slider(start = 1, end = 24, value=0, step=1, title="Hour")
hour_input.on_change('value', update_hour)

minute_input = Slider(start = 0, end = 60, value=0, step=1, title="Minute")
minute_input.on_change('value', update_minute)

table1_title = Div(text="""<b>Inputted Locations</b>""")
table2_title = Div(text="""<b>History</b>""")

layout = Row(Column(p,
                    button,
                    table2_title,
                    data_table_2,
                    ),
             Column(
                    dt_pckr,
                    hour_input,
                    minute_input,
                    population_size_input,
                    generation_input,
                    table1_title,
                    data_table,
                    l,
                    h
                    )
            )

curdoc().add_root(layout)