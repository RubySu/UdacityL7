#!/usr/bin/env python
# coding: utf-8
import findspark
findspark.init()
# import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf, isnull, from_unixtime, \
    instr, when

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys

# get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import os.path
from pyspark.ml.pipeline import PipelineModel


# load data from Spark
def load_data(path):
    # create a Spark session
    spark = SparkSession.builder.appName("Spark").getOrCreate()
    df = spark.read.csv(path,header=True,inferSchema=True)
    df.persist()
    return df

# transform features for dataframe
def features_transform(df):
    labelIndexer = StringIndexer(inputCol="churn", outputCol="label").fit(df)
    df = VectorAssembler(inputCols=df.columns[2:], outputCol="indexFeatures").transform(df)
    return df,labelIndexer


# build RandomForest model
def rf_model(trainingData,testData,labelIndexer,save_path):
    rf = RandomForestClassifier(labelCol="label", featuresCol="indexFeatures")
    paramGrid = ParamGridBuilder().build()
    # paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [1, 3, 5]).addGrid(rf.maxDepth, [3, 5, 7, 10]).addGrid(rf.maxBins, [20, 30, 40]).build()
    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, rf])

    evaluator = MulticlassClassificationEvaluator(metricName='f1')

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)
    # Train model.  This also runs the indexers.
    model = cv.fit(trainingData)
    bestModel = model.bestModel
    bestModel.write().overwrite().save("rf")
    predictions = model.transform(testData)
    return model, predictions, evaluator

# build LogisticRegression model
def lr_model(trainingData,testData,labelIndexer,save_path):
    lr = LogisticRegression(labelCol="label", featuresCol="indexFeatures")
    #     paramGrid = ParamGridBuilder() \
    #         .addGrid(lr.elasticNetParam,[0.1, 1.0]) \
    #         .addGrid(lr.regParam,[0.0, 0.05]) \
    #         .build()
    paramGrid = ParamGridBuilder().build()
    # paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [1, 3, 5]).addGrid(rf.maxDepth, [3, 5, 7, 10]).addGrid(rf.maxBins, [20, 30, 40]).build()
    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, lr])
    evaluator = MulticlassClassificationEvaluator(metricName='f1')
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=2)
    # Train model.  This also runs the indexers.
    model = cv.fit(trainingData)
    bestModel = model.bestModel
    bestModel.write().overwrite().save("lr")
    # Make predictions.
    predictions = model.transform(testData)
    return model, predictions, evaluator

# build DecitionTree model
def dt_model(trainingData,testData,labelIndexer,save_path):
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="indexFeatures")
    paramGrid = ParamGridBuilder().addGrid(dt.impurity, ['entropy', 'gini']).addGrid(dt.maxDepth, [2, 4, 6, 8]).build()
    # paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [1, 3, 5]).addGrid(rf.maxDepth, [3, 5, 7, 10]).addGrid(rf.maxBins, [20, 30, 40]).build()
    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, dt])
    evaluator = MulticlassClassificationEvaluator(metricName='f1')
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)
    # Train model.  This also runs the indexers.
    model = cv.fit(trainingData)
    bestModel = model.bestModel
    bestModel.write().overwrite().save("dt")
    # Make predictions.
    predictions = model.transform(testData)
    return model, predictions, evaluator


def print_scores(predictions, evaluator):
    F1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    print("F1_score : "+str(F1_score))


def main():

    if len(sys.argv) == 5:

        print("load data and transform features...")
        df= load_data(sys.argv[1])
        df, labelIndexer = features_transform(df)
        trainingData, testData = df.randomSplit([0.7, 0.3], seed=6)
        trainingData.cache()


        print("train or load RandomForest models with tuning parameters, then make a prediction on testData...")
        if os.path.exists(sys.argv[2]):
            persistedModel = PipelineModel.load(sys.argv[2])
            evaluator = MulticlassClassificationEvaluator(metricName='f1')
            predictions = persistedModel.transform(testData)
        else:
            model1, predictions, evaluator = rf_model(trainingData,testData,labelIndexer)
        print("rf model evaluation...")
        # print F1 score of the prediction
        print_scores(predictions, evaluator)

        print("train or load LogisticRegression models with tuning parameters, then make a prediction on testData")
        if os.path.exists(sys.argv[3]):
            persistedModel = PipelineModel.load(sys.argv[3])
            evaluator = MulticlassClassificationEvaluator(metricName='f1')
            predictions = persistedModel.transform(testData)
        else:
            model2, predictions, evaluator = lr_model(trainingData,testData,labelIndexer)
        print("lr model evaluation...")
        # print F1 score of the prediction
        print_scores(predictions, evaluator)

        print("train or load DecisionTree models with tuning parameters, then make a prediction on testData")
        if os.path.exists(sys.argv[4]):
            persistedModel = PipelineModel.load(sys.argv[4])
            evaluator = MulticlassClassificationEvaluator(metricName='f1')
            predictions = persistedModel.transform(testData)
        else:
            model3, predictions, evaluator = dt_model(trainingData,testData,labelIndexer)
        print("dt model evaluation...")
        # print F1 score of the prediction
        print_scores(predictions, evaluator)


    else:
        print('Please provide the filepath of the features data '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second to fourth argument. \n\nExample: python '\
              'train_classifier.py ../data/user_item.csv rf lr dt')


if __name__ == '__main__':

    main()


