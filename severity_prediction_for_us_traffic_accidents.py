from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("AccidentSeverityPrediction").getOrCreate()

df = spark.read.csv("hdfs:///user/cloudera/hnew/accidents.csv", header=True, inferSchema=True)

selected_features = [
    'Start_Lat', 'Start_Lng', 'Distance(mi)', 'Temperature(F)', 'Humidity(%)',
    'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Severity'
]
ml_df = df.select(selected_features).dropna()
ml_df = ml_df.withColumnRenamed('Severity', 'label')

feature_cols = [c for c in ml_df.columns if c != 'label']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_df = assembler.transform(ml_df)

(training_data, test_data) = final_df.randomSplit([0.7, 0.3], seed=42)
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
model = dt.fit(training_data)

predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)

print("-" * 40)
print(f"Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print("-" * 40)

print("Feature Importances:")
feature_names = assembler.getInputCols()
importances = model.featureImportances.toArray()
feature_importance_list = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
max_importance = feature_importance_list[0][1] if feature_importance_list else 1
chart_width = 50
for feature, importance in feature_importance_list:
    bar_length = int((importance / max_importance) * chart_width)
    bar = '#' * bar_length
    print("{0:<20}: [{1:<{width}}] {2:.4f}".format(feature, bar, importance, width=chart_width))
print("-" * 40)

predictions.select("label", "prediction", "features").show(5)
spark.stop()

