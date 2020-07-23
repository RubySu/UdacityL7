import json
import plotly
import pandas as pd
import findspark
findspark.init()

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf, isnull,from_unixtime,instr,when
from pyspark.ml.pipeline import PipelineModel


app = Flask(__name__)




# load data
spark = SparkSession.builder.appName("Spark").getOrCreate()


# load model
model = PipelineModel.load("../models/lr")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    path = "../data/page_churn_byUser.csv"
    df = spark.read.csv(path, header=True, inferSchema=True)
    df.persist()
    genre_counts1 = df.filter(df.churn == 0)
    genre_names1 = genre_counts1.select("page").distinct().toPandas()["page"].tolist()
    genre_counts2 = df.filter(df.churn == 1)
    genre_names2 = genre_counts2.select("page").distinct().toPandas()["page"].tolist()
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    trace1 = {
        "name": "unchurn",
        "type": "bar",
        "x": genre_names1,
        "y": genre_counts1.toPandas()['count_s'].tolist(),
        # "x":["about"],
        # "y":[5],
        "marker": {
            "line": {
                "color": "grey",
                "width": 3
            },
            "color": "rgb(111,168,220)"
        }
    }
    trace2 = {
        "name": "churn",
        "type": "bar",
        "x": genre_names2,
        "y": genre_counts2.toPandas()['count_s'].tolist(),
        # "x": ["about"],
        # "y": [10],
        "marker": {
            "line": {
                "color": "grey",
                "width": 3
            },
            "color": "rgb(255,165,0)"
        }
    }
    graphs = [
        {
            'data': [
                trace1, trace2
            ],

            'layout': {
                'title': 'Distribution of Churn by Page',
                'yaxis': {
                    'title': "Churn Count"
                },
                'xaxis': {
                    'title': "Page"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('reports.html', ids=ids, graphJSON=graphJSON)



@app.route('/multireports')
def multireports():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    path = "../data/level_churn_byUser.csv"
    df = spark.read.csv(path, header=True, inferSchema=True)
    df.persist()
    genre_counts1 = df.filter(df.churn == 0).groupby("level").count()
    genre_names1 = genre_counts1.select("level").distinct().toPandas()["level"].tolist()
    genre_counts2 = df.filter(df.churn == 1).groupby("level").count()
    genre_names2 = genre_counts2.select("level").distinct().toPandas()["level"].tolist()
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    trace1 = {
        "name": "unchurn",
        "type": "bar",
        "x": genre_names1,
        "y": genre_counts1.toPandas()['count'].tolist(),
        "marker": {
            "line": {
                "color": "grey",
                "width": 3
            },
            "color": "rgb(111,168,220)"
        }
    }
    trace2 = {
        "name": "churn",
        "type": "bar",
        "x": genre_names2,
        "y": genre_counts2.toPandas()['count'].tolist(),
        "marker": {
            "line": {
                "color": "grey",
                "width": 3
            },
            "color": "rgb(255,165,0)"
        }
    }
    graphs = [
        {
            'data': [
                trace1, trace2
            ],

            'layout': {
                'title': 'Distribution of Churn by User Level',
                'yaxis': {
                    'title': "Churn Count"
                },
                'xaxis': {
                    'title': "Level"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


def main():
    app.run(host='localhost', port=3001, debug=True)


if __name__ == '__main__':
    main()