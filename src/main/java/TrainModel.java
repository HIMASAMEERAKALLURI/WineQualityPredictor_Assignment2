import org.apache.spark.sql.*;
import org.apache.spark.ml.feature.VectorAssembler;
import java.io.IOException;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;

import org.apache.spark.api.java.JavaRDD;
import scala.Tuple2;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;

public class TrainModel {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Trainer")
                .master("local[4]")
                .getOrCreate();

        // Load datasets
        Dataset<Row> rawTrainData = spark.read()
                .option("header", "true")
                .option("sep", ",")
                .option("inferSchema", "true")
                .csv("TrainingDataset.csv");

        Dataset<Row> trainData = rawTrainData.toDF(
                "fixed_acidity", "volatile_acidity", "citric_acid",
                "residual_sugar", "chlorides", "free_sulfur_dioxide",
                "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"
        );

        Dataset<Row> rawValidationData = spark.read()
                .option("header", "true")
                .option("sep", ",")
                .option("inferSchema", "true")
                .csv("ValidationDataset.csv");

        Dataset<Row> validationData = rawValidationData.toDF(
                "fixed_acidity", "volatile_acidity", "citric_acid",
                "residual_sugar", "chlorides", "free_sulfur_dioxide",
                "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality"
        );

        String[] featureCols = new String[]{
                "fixed_acidity", "volatile_acidity", "citric_acid",
                "residual_sugar", "chlorides", "free_sulfur_dioxide",
                "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
        };

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        Dataset<Row> trainingFinal = assembler.transform(trainData)
                .select("features", "quality")
                .withColumnRenamed("quality", "label");

        Dataset<Row> validationFinal = assembler.transform(validationData)
                .select("features", "quality")
                .withColumnRenamed("quality", "label");

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.1);

        LogisticRegressionModel model = lr.fit(trainingFinal);

        Dataset<Row> predictions = model.transform(validationFinal);

        // Cast labels to double for compatibility
        Dataset<Row> castedPreds = predictions.selectExpr("cast(label as double) as label", "cast(prediction as double) as prediction");

        // Convert to JavaRDD for MulticlassMetrics
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = castedPreds.javaRDD()
                .map(row -> new Tuple2<>(row.getDouble(1), row.getDouble(0)));

        // Instantiate metrics
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

        // Print metrics
        System.out.println("========= Validation Metrics =========");
        System.out.println("Accuracy = " + metrics.accuracy());
        System.out.println("Weighted Precision = " + metrics.weightedPrecision());
        System.out.println("Weighted Recall = " + metrics.weightedRecall());
        System.out.println("Weighted F1 Score = " + metrics.weightedFMeasure());
        System.out.println("Confusion Matrix:\n" + metrics.confusionMatrix().toString());

        model.save("WineQualityModel");
        spark.stop();
    }
}
