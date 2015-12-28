
# coding: utf-8

# In[1]:

import pandas as pd
import jinja2

from collections import OrderedDict
from json import dumps
from IPython.html.widgets import interact
from IPython.html import widgets
from IPython.display import display, display_pretty, Javascript, HTML
from IPython.utils.traitlets import Any, Bool, Dict, List, Unicode
from threading import Lock
from urllib2 import urlopen


# # Embedding Interactive Charts on an IPython Notebook

# ## Introduction

# In this three part post we’ll show you how easy it is to integrate D3.js, Chart.js and HighCharts chart into an notebook and how to make them interactive using HTML widgets.

# ## Requirements

# The only requirement to run the examples is IPython Notebook version 2.0 or greater. All the modules that we reference are either in the standard Python distribution, or are dependencies of IPython.

# ## About Pandas

# Although Pandas is not strictly necessary to accomplish what we do in the examples, it is such a popular data analysis tool that we wanted to use it anyway. We recommend that you read the 10 Minutes to Pandas tutorial to get and idea of what it can do or buy *Python for Data Analysis* for an in depth guide of data analysis using Python, Pandas and NumPy.

# ## About the Data

# All the data that we use in the examples are taken from the [United States Census Bureau](http://www.census.gov) site. We're going to use 2012 population estimates and we're going to plot the sex and age groups by the state, region and division.

# ### Population by State

# We're going to build a Pandas DataFrame from the dataset of [Incorporated Places and Minor Civil Divisions](http://www.census.gov/popest/data/cities/totals/2012/SUB-EST2012.html). We could have just grabbed the [estimates for the states](http://www.census.gov/popest/data/state/totals/2012/index.html), but also wanted to show you how easy it is to work with data using Pandas. First, we fetch the data using [urlopen](https://docs.python.org/2/library/urllib2.html#urllib2.urlopen) and we parse the response as CSV using Pandas' [read_csv](http://pandas.pydata.org/pandas-docs/version/0.13.1/generated/pandas.io.parsers.read_csv.html) function:

# In[7]:

sub_est_2012_df = pd.read_csv(
    urlopen('http://www.census.gov/popest/data/cities/totals/2012/files/SUB-EST2012.csv'),
    encoding='latin-1',
    dtype={'STATE': 'str', 'COUNTY': 'str', 'PLACE': 'str'}
)


# In[8]:

sub_est_2012_df.head()


# The resulting data frame has a lot of information that we don’t need and can be discarded. According to the [file layout description](http://www.census.gov/popest/data/cities/totals/2012/files/SUB-EST2012.pdf), the data is summarized at the nation, state, county and place levels according to the SUMLEV column. Since we’re only interested in the population for each state we can just filter the rows with SUMLEV ‘40’, but wanted to show you how to use the aggregate feature of Pandas’ DataFrames, so we’ll take the data summarized at the count level (SUMLEV ‘50’), then we’ll group by state, and sum the population estimates.

# In[9]:

sub_est_2012_df_by_county = sub_est_2012_df[sub_est_2012_df.SUMLEV == 50]
sub_est_2012_df_by_state = sub_est_2012_df_by_county.groupby(['STATE']).sum()

# Alternatively we could have just taken the summary rows for the states

# sub_est_2012_df_by_state = sub_est_2012_df[sub_est_2012_df.SUMLEV == 40]


# If you see the table, the states are referenced using their ANSI codes. We can augment the table to include the state names and abbreviations by merging with [another resource](http://www.census.gov/geo/reference/ansi_statetables.html) from the Geography section of the US Census Bureau site. We use `read_csv` Pandas function making sure that we use the pipe character (|) as separator.

# In[11]:

# Taken from http://www.census.gov/geo/reference/ansi_statetables.html

state = pd.read_csv(urlopen('http://www2.census.gov/geo/docs/reference/state.txt'), sep='|', dtype={'STATE': 'str'})


# In[12]:

state.drop(
    ['STATENS'],
    inplace=True, axis=1
)


# In[13]:

sub_est_2012_df_by_state = pd.merge(sub_est_2012_df_by_state, state, left_index=True, right_on='STATE')
sub_est_2012_df_by_state.drop(
    ['SUMLEV', 'COUSUB', 'CONCIT', 'ESTIMATESBASE2010', 'POPESTIMATE2010', 'POPESTIMATE2011'],
    inplace=True, axis=1
)


# We're also interested in plotting the information about the age and sex of the people, and for that we can use the [Annual Estimates of the Civilian Population by Single Year of Age and Sex](http://www.census.gov/popest/data/state/asrh/2012/SC-EST2012-AGESEX-CIV.html).

# In[14]:

# Taken from http://www.census.gov/popest/data/state/asrh/2012/SC-EST2012-AGESEX-CIV.html

sc_est2012_agesex_civ_df = pd.read_csv(
    urlopen('http://www.census.gov/popest/data/state/asrh/2012/files/SC-EST2012-AGESEX-CIV.csv'),
    encoding='latin-1',
    dtype={'SUMLEV': 'str'}
)


# Once again, the table is summarized at many levels, but we're only interested in the information at the state level, so we filter out the unnecessary rows. We also do a little bit of processing to the STATE column so it can be used to merge with the state DataFrame.

# In[15]:

sc_est2012_agesex_civ_df_sumlev040 = sc_est2012_agesex_civ_df[
    (sc_est2012_agesex_civ_df.SUMLEV == '040') &
    (sc_est2012_agesex_civ_df.SEX != 0) &
    (sc_est2012_agesex_civ_df.AGE != 999)
]
sc_est2012_agesex_civ_df_sumlev040.drop(
    ['SUMLEV', 'NAME', 'ESTBASE2010_CIV', 'POPEST2010_CIV', 'POPEST2011_CIV'],
    inplace=True, axis=1
)
sc_est2012_agesex_civ_df_sumlev040['STATE'] = sc_est2012_agesex_civ_df_sumlev040['STATE'].apply(lambda x: '%02d' % (x,))


# What we need to do is group the rows by state, region, division and sex, and sum across all ages. Afterwards, we augment the result with the names and abbreviations of the states. 

# In[16]:

sc_est2012_sex = sc_est2012_agesex_civ_df_sumlev040.groupby(['STATE', 'REGION', 'DIVISION', 'SEX'], as_index=False)[['POPEST2012_CIV']].sum()
sc_est2012_sex = pd.merge(sc_est2012_sex, state, left_on='STATE', right_on='STATE')


# For the age information, we group by state, region, division and age and we sum across all sexes. If you see the result, you'll notice that there's a row for each year. This is pretty useful for analysis, but it can be problematic to plot, so we're going to group the rows according to age buckets of 20 years. Once again, we add the state information at the end.

# In[17]:

sc_est2012_age = sc_est2012_agesex_civ_df_sumlev040.groupby(['STATE', 'REGION', 'DIVISION', 'AGE'], as_index=False)[['POPEST2012_CIV']].sum()


# In[18]:

age_buckets = pd.cut(sc_est2012_age.AGE, range(0,100,20))


# In[19]:

sc_est2012_age = sc_est2012_age.groupby(['STATE', 'REGION', 'DIVISION', age_buckets], as_index=False)['POPEST2012_CIV'].sum()
sc_est2012_age = pd.merge(sc_est2012_age, state, left_on='STATE', right_on='STATE')


# We also need information about regions and divisions, but since the dataset is small, we'll build the dictionaries by hand.

# In[20]:

region_codes = {
    0: 'United States Total',
    1: 'Northeast',
    2: 'Midwest',
    3: 'South',
    4: 'West'
}
division_codes = {
    0: 'United States Total',
    1: 'New England',
    2: 'Middle Atlantic',
    3: 'East North Central',
    4: 'West North Central',
    5: 'South Atlantic',
    6: 'East South Central',
    7: 'West South Central',
    8: 'Mountain',
    9: 'Pacific'
}


# ## Part 1 - Embedding D3.js

# [D3.js](http://d3js.org/) is an incredibly flexible JavaScript chart library. Although it is primarily used to plot data, it can be used to draw arbitrary graphics and animations.
# 
# Let's build a column chart of the five most populated states in the USA. IPython Notebooks are regular web pages so in order to use any JavaScript library in it, we need to load the necessary requirements. IPython Notebook uses [RequireJS](http://requirejs.org/) to load its own requirements, so we can make use of it with the `%%javascript` cell magic to load external dependencies.
# 
# In all the examples of this notebook we'll load the libraries from [cdnjs.com](http://cdnjs.com/), so to declare the requirement of  D3.js we do

# In[21]:

get_ipython().run_cell_magic(u'javascript', u'', u"require.config({\n    paths: {\n        d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min'\n    }\n});")


# Now we'll make use of the `display` function and `HTML` from the IPython Notebook [API](http://ipython.org/ipython-doc/2/api/generated/IPython.core.display.html#module-IPython.core.display) to render HTML content within the notebook itself. We're declaring styles to change the look and feel of the plots, and we define a new `div` with id `"chart_d3"` that the library is going to use as the target of the plot.

# In[22]:

display(HTML("""
<style>
.bar {
  fill: steelblue;
}
.bar:hover {
  fill: brown;
}
.axis {
  font: 10px sans-serif;
}
.axis path,
.axis line {
  fill: none;
  stroke: #000;
}
.x.axis path {
  display: none;
}
</style>
<div id="chart_d3"/>
"""))


# Next, we define a template with the JavaScript code that is going to render the chart. Notice that we iterate over the "data" parameter to populate the "data" variable in JavaScript. Afterwards, we use the `display` method once again to force the execution of the JavaScript code, which renders the chart on the target div.

# ### sub_est_2012_df_by_state_template

# In[23]:

sub_est_2012_df_by_state_template = jinja2.Template(
"""
// Based on http://bl.ocks.org/mbostock/3885304

require(["d3"], function(d3) {
    var data = []

    {% for row in data %}
    data.push({ 'state': '{{ row[4] }}', 'population': {{ row[1] }} });
    {% endfor %}

    d3.select("#chart_d3 svg").remove()

    var margin = {top: 20, right: 20, bottom: 30, left: 40},
        width = 800 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

    var x = d3.scale.ordinal()
        .rangeRoundBands([0, width], .25);

    var y = d3.scale.linear()
        .range([height, 0]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom");

    var yAxis = d3.svg.axis()
        .scale(y)
        .orient("left")
        .ticks(10)
        .tickFormat(d3.format('.1s'));
        
    var svg = d3.select("#chart_d3").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    x.domain(data.map(function(d) { return d.state; }));
    y.domain([0, d3.max(data, function(d) { return d.population; })]);

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .text("Population");

    svg.selectAll(".bar")
        .data(data)
        .enter().append("rect")
        .attr("class", "bar")
        .attr("x", function(d) { return x(d.state); })
        .attr("width", x.rangeBand())
        .attr("y", function(d) { return y(d.population); })
        .attr("height", function(d) { return height - y(d.population); });
});
"""
)
display(Javascript(sub_est_2012_df_by_state_template.render(
    data=sub_est_2012_df_by_state.sort(['POPESTIMATE2012'], ascending=False)[:5].itertuples()))
)


# The chart shows that California, Texas, New York, Florida and Illinois are the most populated states. What about the other states? Let's build an interactive chart that allows us to show whichever state we chose. IPython Notebook provides widgets that allow us to get information from the user in an intuitive manner. Sadly, at the time of this writing, there's no widget to select multiple items from a list, so before we move on, let's define such widget.

# ### MultipleSelectWidget

# In[24]:

class MultipleSelectWidget(widgets.DOMWidget):
    _view_name = Unicode('MultipleSelectView', sync=True)
    
    value = List(sync=True)
    values = Dict(sync=True)
    values_order = List(sync=True)
    description = Unicode(sync=True)

    def __init__(self, *args, **kwargs):
        self.value_lock = Lock()

        self.values = kwargs.get('values', [])
        self.value = kwargs.get('value', [])
        self.values_order = kwargs.get('values_order', [])
            
        widgets.DOMWidget.__init__(self, *args, **kwargs)


# In[25]:

get_ipython().run_cell_magic(u'javascript', u'', u'require(["widgets/js/widget"], function(WidgetManager){\n    var MultipleSelectView = IPython.DOMWidgetView.extend({\n        initialize: function(parameters) {\n            this.model.on(\'change\',this.update,this);\n            this.options = parameters.options;\n            this.child_views = [];\n            // I had to override DOMWidgetView\'s initialize to set model.views otherwise\n            // multiple views would get attached to the model\n            this.model.views = [this];\n        },\n        \n        render : function(){\n            this.$el\n                .addClass(\'widget-hbox\');\n            this.$label = $(\'<div />\')\n                .appendTo(this.$el)\n                .addClass(\'widget-hlabel\')\n                .hide();\n            this.$listbox = $(\'<select/>\')\n                .addClass(\'widget-listbox\')\n                .attr(\'multiple\', \'\')\n                .attr(\'size\', 6)\n                .appendTo(this.$el);\n            this.$el_to_style = this.$listbox;\n            this.update();\n        },\n        \n        update : function(options){\n            if (typeof(options) === \'undefined\' || options.updated_view != this) {\n                var values = this.model.get(\'values\');\n                var values_order = this.model.get(\'values_order\');\n                \n                var that = this;\n                _.each(values_order, function(key, index) {\n                    if (that.$listbox.find(\'option[key="\' + key + \'"]\').length === 0) {\n                        $(\'<option />\')\n                            .text(values[key])\n                            .attr(\'key\', key)\n                            .appendTo(that.$listbox)\n                            .on(\'click\', $.proxy(that.handle_click, that));\n                    }                    \n                });\n                \n                var value = this.model.get(\'value\') || [];\n                \n                this.$listbox.find(\'option\').each(function(index, element) {\n                    var key = $(element).attr(\'key\');\n                    \n                    if (key in values) {\n                        if (value.indexOf(key) != -1) {\n                            $(element).prop(\'selected\', true);\n                        }\n                    } else {\n                        $(element).remove();\n                    }\n                });\n\n                var description = this.model.get(\'description\');\n                if (description.length === 0) {\n                    this.$label.hide();\n                } else {\n                    this.$label.text(description);\n                    this.$label.show();\n                }\n            }\n            return MultipleSelectView.__super__.update.apply(this);\n        },\n\n        handle_click: function (event) {\n            var value = $(event.target).parent().children(\'option:selected\').map(function() { return $(this).attr(\'key\') }).get()\n            \n            this.model.set(\'value\', value, {updated_view: this});\n            this.touch();\n        },    \n    });\n    WidgetManager.register_widget_view(\'MultipleSelectView\', MultipleSelectView);\n});')


# We're going to use IPython's [interact](http://ipython.org/ipython-doc/dev/api/generated/IPython.html.widgets.interaction.html#module-IPython.html.widgets.interaction) function to display the widgets and execute the callback function responsible to draw the chart. As we mentioned before, d3 requires a target element to draw the chart, so we use an [HTMLWidget](http://ipython.org/ipython-doc/dev/api/generated/IPython.html.widgets.widget_string.html#module-IPython.html.widgets.widget_string) to make sure the div is properly rendered before the callback is executed.

# ### display_chart_d3

# In[26]:

def display_chart_d3(data, show_javascript, states, div):
    sub_est_2012_df_by_state_template = jinja2.Template(
    """
    // Based on http://www.recursion.org/d3-for-mere-mortals/

    require(["d3"], function(d3) {
        var data = []

        {% for row in data %}
        data.push({ 'state': '{{ row[4] }}', 'population': {{ row[1] }} });
        {% endfor %}

        d3.select("#chart_d3_interactive svg").remove()

        var margin = {top: 20, right: 20, bottom: 30, left: 40},
            width = 800 - margin.left - margin.right,
            height = 400 - margin.top - margin.bottom;

        var x = d3.scale.ordinal()
            .rangeRoundBands([0, width], .25);

        var y = d3.scale.linear()
            .range([height, 0]);

        var xAxis = d3.svg.axis()
            .scale(x)
            .orient("bottom");

        var yAxis = d3.svg.axis()
            .scale(y)
            .orient("left")
            .ticks(10)
            .tickFormat(d3.format('.1s'));

        var svg = d3.select("#chart_d3_interactive").append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        x.domain(data.map(function(d) { return d.state; }));
        y.domain([0, d3.max(data, function(d) { return d.population; })]);

        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis);

        svg.append("g")
            .attr("class", "y axis")
            .call(yAxis)
            .append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", ".71em")
            .style("text-anchor", "end")
            .text("Population");

        svg.selectAll(".bar")
            .data(data)
            .enter().append("rect")
            .attr("class", "bar")
            .attr("x", function(d) { return x(d.state); })
            .attr("width", x.rangeBand())
            .attr("y", function(d) { return y(d.population); })
            .attr("height", function(d) { return height - y(d.population); });
    });
    """
    )
    rendered_template = sub_est_2012_df_by_state_template.render(
        data=data[data['STUSAB'].map(lambda v: v in states)].itertuples()
    )
    
    if show_javascript:
        display(widgets.PopupWidget(children=(widgets.HTMLWidget(value='<div style="width:600px; height: 400px; overflow:scroll;"><pre>{}</pre></div>'.format(rendered_template)),)))
        
    display(Javascript(rendered_template))


# In[27]:

values = {
    record['STUSAB']: "{0} - {1}".format(record['STUSAB'], record['STATE_NAME']) for record in state[['STUSAB', 'STATE_NAME']].sort('STUSAB').to_dict(outtype='records')
}
i = interact(
    display_chart_d3,
    data=widgets.fixed(sub_est_2012_df_by_state),
    show_javascript=widgets.CheckboxWidget(value=False),
    states=MultipleSelectWidget(
        value=['CA', 'NY'],
        values=values,
        values_order=sorted(values.keys())
    ),
    div=widgets.HTMLWidget(value='<div id="chart_d3_interactive"></div>')
)


# We've also added a show_javascript checkbox to display the generated code on a pop-up.

# Although D3 is capable of creating [incredible charts](https://github.com/mbostock/d3/wiki/Gallery), it has a steep learning curve and it can be overkill if what you want are just simple charts. Let us explore simpler alternatives.

# ## Part 2 - Embedding Chart.js

# On the previous part we used [D3.js](http://d3js.org/) to embed interactive charts onto an IPython Notebook. D3.js is an extremely flexible library capable of creating incredible charts, but it can be hard to use. If all we need is just a simple chart without much work, there are other charting libraries that can do just that.
# 
# [Chart.js](http://www.chartjs.org/) is an HTML5 charting library capable of producing beautiful graphics with very little code. First we need to declare the requirement using RequireJS:

# In[28]:

get_ipython().run_cell_magic(u'javascript', u'', u"require.config({\n    paths: {\n        chartjs: '//cdnjs.cloudflare.com/ajax/libs/Chart.js/0.2.0/Chart.min'\n    }\n});")


# The procedure is the same as before, we define a template that will contain the rendered JavaScript code, and we use `display` to embed the code into the running page.
# 
# Now we want to plot the male and female population by region, division and state. We'll use `interact` again, but this time we don't need custom widgets as we're only selecting a single item (we'll use a [DropdownWidget](http://ipython.org/ipython-doc/dev/api/generated/IPython.html.widgets.widget_selection.html#module-IPython.html.widgets.widget_selection)). We've also included the `show_javascript` checkbox.

# ### display_chart_chartjs

# In[29]:

def display_chart_chartjs(sc_est2012_sex, show_javascript, show, div):
    sc_est2012_sex_template = jinja2.Template(
    """
    require(["chartjs"], function() {
        var data = {
            labels : {{ labels }},
            datasets : [
            {
                fillColor : "rgba(220,220,220,0.5)",
                strokeColor : "rgba(220,220,220,1)",
                data : {{ dataset_male }}
            },
            {
                fillColor : "rgba(151,187,205,0.5)",
                strokeColor : "rgba(151,187,205,1)",
                data : {{ dataset_female }}
            }
            ]
        }
    
    
        var ctx = $('#chart_chartjs')[0].getContext('2d');
        new Chart(ctx).Bar(data,{});
    });
    """
    )
    
    if show == 'by_region':
        data_frame = sc_est2012_sex.groupby(['REGION', 'SEX'], as_index=False)[['POPEST2012_CIV']].sum()
        labels = [region_codes[code] for code in data_frame['REGION'].drop_duplicates()]
    elif show == 'by_division':
        data_frame = sc_est2012_sex.groupby(['DIVISION', 'SEX'], as_index=False)[['POPEST2012_CIV']].sum()
        labels = [division_codes[code] for code in data_frame['DIVISION'].drop_duplicates()]
    elif show == 'by_state':
        data_frame = sc_est2012_sex.groupby(['STATE', 'SEX'], as_index=False)[['POPEST2012_CIV']].sum()
        data_frame = pd.merge(data_frame, state, left_on='STATE', right_on='STATE')
        labels = data_frame['STATE_NAME'].drop_duplicates().tolist()

    dataset_male = data_frame[data_frame.SEX == 1]['POPEST2012_CIV'].tolist()
    dataset_female = data_frame[data_frame.SEX == 2]['POPEST2012_CIV'].tolist()

    rendered_template = sc_est2012_sex_template.render(
        labels=dumps(labels),
        dataset_male=dumps(dataset_male),
        dataset_female=dumps(dataset_female),
    )

    if show_javascript:
        display(widgets.PopupWidget(children=(widgets.HTMLWidget(value='<div style="width:600px; height: 400px; overflow:scroll;"><pre>{}</pre></div>'.format(rendered_template)),)))
    
    display(Javascript(rendered_template))


# In[30]:

i = interact(
    display_chart_chartjs,
    sc_est2012_sex=widgets.fixed(sc_est2012_sex),
    show_javascript=widgets.Widget(CheckboxWidget(value=False),
    show=widgets.DropdownWidget(
        values={'By Region':'by_region', 'By Division': 'by_division', 'By State': 'by_state'},
        value='by_region'
    ),
    div=widgets.HTMLWidget(value='<canvas width=800 height=400 id="chart_chartjs"></canvas>')
)


# The library generates an beautiful and simple animated column chart. There's not much in terms of customization of the Chart.js charts, but that makes it very easy to use.

# ## Part 3 - Embedding HighCharts

# Somewhat in between Charts.js and D3 is [HighCharts](http://www.highcharts.com/). It can produce beautiful and professional looking animated charts without much coding and it can be customized if needed. This time we'll plot the age of the population by region and division. We're going to show one data series per bucket of 20 years. Before we can begin, we need to declare the requirement:

# In[ ]:

get_ipython().run_cell_magic(u'javascript', u'', u"require.config({\n    paths: {\n        highcharts: '//cdnjs.cloudflare.com/ajax/libs/highcharts/4.0.1/highcharts'\n    }\n});")


# Then, we use the same trick with `interact`, `display` and `JavaScript`:

# ### display_chart_highcharts

# In[ ]:

def display_chart_highcharts(sc_est2012_age, show_javascript, show, div):
    sc_est2012_age_template = jinja2.Template(
    """
    require(["highcharts"], function() {
        $('#chart_highcharts').highcharts({
            chart: {
                type: 'column'
            },
            title: {
                text: 'Population By Age'
            },
            subtitle: {
                text: 'Source: www.census.gov'
            },
            xAxis: {
                categories: {{ categories }}
            },
            yAxis: {
                min: 0,
                title: {
                    text: 'Population'
                }
            },
            plotOptions: {
                column: {
                    pointPadding: 0.2,
                    borderWidth: 0
                }
            },
            series: {{ series }}
        });
    });
    """
    )
    
    if show == 'by_region':
        data_frame = sc_est2012_age.groupby(['REGION', 'AGE'], as_index=False)[['POPEST2012_CIV']].sum()
        categories = [region_codes[code] for code in data_frame['REGION'].drop_duplicates()]
    elif show == 'by_division':
        data_frame = sc_est2012_age.groupby(['DIVISION', 'AGE'], as_index=False)[['POPEST2012_CIV']].sum()
        categories = [division_codes[code] for code in data_frame['DIVISION'].drop_duplicates()]
        
    series_names = data_frame['AGE'].drop_duplicates().tolist()
    series = [{
        'name': series_name, 'data': data_frame[data_frame.AGE == series_name]['POPEST2012_CIV'].tolist()
    } for series_name in series_names]
    
    rendered_template = sc_est2012_age_template.render(
        categories=dumps(categories),
        series=dumps(series)
    )

    if show_javascript:
        display(widgets.PopupWidget(children=(widgets.HTMLWidget(value='<div style="width:600px; height: 400px; overflow:scroll;"><pre>{}</pre></div>'.format(rendered_template)),)))

    display(Javascript(rendered_template))


# In[ ]:

i = interact(
    display_chart_highcharts,
    sc_est2012_age=widgets.fixed(sc_est2012_age),
    show_javascript=widgets.CheckboxWidget(value=False),
    show=widgets.DropdownWidget(
        values={'By Region':'by_region', 'By Division': 'by_division'},
        value='by_region'
    ),
    div=widgets.HTMLWidget(value='<div id="chart_highcharts"></div>')
)


# ## Conclusions

# We showed you how to integrate three popular JavaScript charting solutions within an IPython Notebook, and how to make them interactive using HTML widgets. Following this pattern it is quite easy to integrate other libraries like [Google Charts](https://developers.google.com/chart/) (it's trivial using the [goog](https://github.com/millermedeiros/requirejs-plugins) RequireJS plugins), [AmCharts](http://www.amcharts.com/) or [JqPlot](http://www.jqplot.com/).

# We also used a little bit of Pandas magic to arrange the data to fit our needs.
