import sys

from pyspark import SparkConf, SparkContext
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, StandardScaler, VectorAssembler, IndexToString, \
    ChiSqSelector
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Number of arguments required: 2')
        print('Command: event_prediction.py <input_events_file>')
        exit()

    input_events_file = sys.argv[1]

    conf = SparkConf().setAppName("EventPrediction").setMaster("local")
    spark = SparkContext(conf=conf)
    session = SparkSession(spark)

    try:
        # Read JSON
        eventsDf = session.read.json(input_events_file, multiLine=False)

        # Drop unused columns and missing data rows
        eventsDf = eventsDf.drop('date8_val')
        eventsDf = eventsDf.drop('_id')
        eventsDf = eventsDf.drop('url')
        eventsDf = eventsDf.drop('id')
        eventsDf = eventsDf.drop('code')
        eventsDf = eventsDf.drop('quad_class')
        eventsDf = eventsDf.drop('day')
        eventsDf = eventsDf.drop('year')
        eventsDf = eventsDf.drop('source_text')
        eventsDf = eventsDf.drop('date8')
        eventsDf = eventsDf.drop('latitude')
        eventsDf = eventsDf.drop('longitude')
        eventsDf = eventsDf.dropna()
        eventsDf.printSchema()

        # Drop rows with no geographical information
        eventsDf = eventsDf.filter(eventsDf.country_code != '')

        # Replace missing values with NA
        eventsDf = eventsDf.withColumn('source', F.when(eventsDf['source'] == "", 'NA')
                                       .otherwise(eventsDf['source']))
        eventsDf = eventsDf.withColumn('src_actor', F.when(eventsDf['src_actor'] == "", 'NA')
                                       .otherwise(eventsDf['src_actor']))
        eventsDf = eventsDf.withColumn('src_agent', F.when(eventsDf['src_agent'] == "", 'NA')
                                       .otherwise(eventsDf['src_agent']))
        eventsDf = eventsDf.withColumn('src_other_agent', F.when(eventsDf['src_other_agent'] == "", 'NA')
                                       .otherwise(eventsDf['src_other_agent']))
        eventsDf = eventsDf.withColumn('target', F.when(eventsDf['target'] == "", 'NA')
                                       .otherwise(eventsDf['target']))
        eventsDf = eventsDf.withColumn('tgt_actor', F.when(eventsDf['tgt_actor'] == "", 'NA')
                                       .otherwise(eventsDf['tgt_actor']))
        eventsDf = eventsDf.withColumn('tgt_agent', F.when(eventsDf['tgt_agent'] == "", 'NA')
                                       .otherwise(eventsDf['tgt_agent']))
        eventsDf = eventsDf.withColumn('tgt_other_agent', F.when(eventsDf['tgt_other_agent'] == "", 'NA')
                                       .otherwise(eventsDf['tgt_other_agent']))

        # Convert goldstein values to vector type
        vector_udf = F.udf(lambda vs: Vectors.dense([vs]), VectorUDT())
        eventsDf = eventsDf.withColumn('goldstein', vector_udf(eventsDf['goldstein']))

        # Train test split
        # (eventsTrainDf, eventsTestDf) = eventsDf.randomSplit([0.7, 0.3])
        eventsTrainDf = eventsDf.filter(eventsDf.month.isin(["09", "10", "11"]))
        eventsTestDf = eventsDf.filter(eventsDf.month.isin(["12"]))
        eventsTrainDf.show()
        eventsTestDf.show()

        # Scaling goldstein values
        goldstein_scaler = StandardScaler().setInputCol("goldstein").setOutputCol("scaled_goldstein").setWithStd(True)\
            .setWithMean(False)

        # String indexing and one hot encoding
        country_indexer = StringIndexer().setInputCol("country_code").setOutputCol("country_code_index")\
            .setHandleInvalid('skip')
        geoname_indexer = StringIndexer().setInputCol("geoname").setOutputCol("geoname_index").setHandleInvalid('skip')
        source_indexer = StringIndexer().setInputCol("source").setOutputCol("source_index").setHandleInvalid('skip')
        src_actor_indexer = StringIndexer().setInputCol("src_actor").setOutputCol("src_actor_index")\
            .setHandleInvalid('skip')
        src_agent_indexer = StringIndexer().setInputCol("src_agent").setOutputCol("src_agent_index")\
            .setHandleInvalid('skip')
        src_other_agent_indexer = StringIndexer().setInputCol("src_other_agent").setOutputCol("src_other_agent_index")\
            .setHandleInvalid('skip')
        target_indexer = StringIndexer().setInputCol("target").setOutputCol("target_index").setHandleInvalid('skip')
        tgt_actor_indexer = StringIndexer().setInputCol("tgt_actor").setOutputCol("tgt_actor_index")\
            .setHandleInvalid('skip')
        tgt_agent_indexer = StringIndexer().setInputCol("tgt_agent").setOutputCol("tgt_agent_index")\
            .setHandleInvalid('skip')
        tgt_other_agent_indexer = StringIndexer().setInputCol("tgt_other_agent").setOutputCol("tgt_other_agent_index")\
            .setHandleInvalid('skip')

        ohe = OneHotEncoderEstimator(inputCols=[country_indexer.getOutputCol(), geoname_indexer.getOutputCol(),
                                                source_indexer.getOutputCol(),
                                                src_actor_indexer.getOutputCol(),
                                                src_agent_indexer.getOutputCol(),
                                                src_other_agent_indexer.getOutputCol(),
                                                target_indexer.getOutputCol(), tgt_actor_indexer.getOutputCol(),
                                                tgt_agent_indexer.getOutputCol(),
                                                tgt_other_agent_indexer.getOutputCol()],
                                     outputCols=["country_code_ohe", "geoname_ohe", "source_ohe",
                                                 "src_actor_ohe", "src_agent_ohe", "src_other_agent_ohe",
                                                 "target_ohe", "tgt_actor_ohe", "tgt_agent_ohe",
                                                 "tgt_other_agent_ohe"],
                                     handleInvalid='keep', dropLast=True)

        # Combine all features into a single column
        feature_assembler = VectorAssembler(inputCols=ohe.getOutputCols() + [goldstein_scaler.getOutputCol()],
                                            outputCol="features")

        # Index root_code labels
        label_indexer = StringIndexer(inputCol="root_code", outputCol="indexedLabel").setHandleInvalid('skip')

        # Select a subset of important features
        feature_selector = ChiSqSelector(percentile=0.5, featuresCol=feature_assembler.getOutputCol(),
                                         labelCol=label_indexer.getOutputCol(),
                                         outputCol="selected_features")

        # Train a RandomForest model
        # rf_classifier = RandomForestClassifier(labelCol=label_indexer.getOutputCol(),
        #                                        featuresCol=feature_selector.getOutputCol(), numTrees=10, maxDepth=20)
        # rf_classifier = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8,
        #                                    labelCol=label_indexer.getOutputCol(),
        #                                    featuresCol=feature_selector.getOutputCol(), family='multinomial')
        rf_classifier = DecisionTreeClassifier(labelCol=label_indexer.getOutputCol(),
                                               featuresCol=feature_selector.getOutputCol(), maxDepth=20)

        # Convert indexed labels back to original labels
        label_converter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                        labels=label_indexer.fit(eventsTrainDf).labels)

        # Pipeline model to combine all transformers
        pipeline = Pipeline(stages=[country_indexer, geoname_indexer, source_indexer, src_actor_indexer,
                                    src_agent_indexer, src_other_agent_indexer, target_indexer, tgt_actor_indexer,
                                    tgt_agent_indexer, tgt_other_agent_indexer, ohe, goldstein_scaler,
                                    feature_assembler, label_indexer, feature_selector, rf_classifier, label_converter])

        # Train model
        trainedModel = pipeline.fit(eventsTrainDf)

        # Make train predictions
        trainPredictions = trainedModel.transform(eventsTrainDf)
        trainPredictions.select("root_code", "indexedLabel", "prediction", "predictedLabel").show()

        # Compute metrics
        trainAccuracy = MulticlassClassificationEvaluator(labelCol=label_indexer.getOutputCol(),
                                                          predictionCol=label_converter.getInputCol(),
                                                          metricName="accuracy").evaluate(trainPredictions)
        print("Train Accuracy:", trainAccuracy * 100)

        # Make test predictions
        testPredictions = trainedModel.transform(eventsTestDf)
        testPredictions.select("root_code", "indexedLabel", "prediction", "predictedLabel").show()

        # Compute metrics
        testAccuracy = MulticlassClassificationEvaluator(labelCol=label_indexer.getOutputCol(),
                                                         predictionCol=label_converter.getInputCol(),
                                                         metricName="accuracy").evaluate(testPredictions)
        print("Test Accuracy:", testAccuracy * 100)

    finally:
        spark.stop()
