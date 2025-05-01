import org.apache.spark.sql.*;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;


public class PredictModel {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Predictor")
                .master("local[2]")
                .getOrCreate();

        // Disable unnecessary logs
        org.apache.log4j.Logger.getLogger("org").setLevel(org.apache.log4j.Level.ERROR);

        // Load validation dataset
        Dataset<Row> rawTestData = spark.read()
                .option("header", "true")
                .option("sep", ",")
                .option("inferSchema", "true")
                .csv("ValidationDataset.csv");

        Dataset<Row> testData = rawTestData.toDF(
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

        Dataset<Row> testFinal = assembler.transform(testData)
                .select("features", "quality")
                .withColumnRenamed("quality", "label");

        // Load trained model
        LogisticRegressionModel model = LogisticRegressionModel.load("WineQualityModel");

        // Make predictions
        Dataset<Row> predictions = model.transform(testFinal);

        // Initialize evaluators
        MulticlassClassificationEvaluator f1Eval = new MulticlassClassificationEvaluator()
                .setLabelCol("label").setPredictionCol("prediction").setMetricName("f1");
        MulticlassClassificationEvaluator precisionEval = new MulticlassClassificationEvaluator()
                .setLabelCol("label").setPredictionCol("prediction").setMetricName("weightedPrecision");
        MulticlassClassificationEvaluator recallEval = new MulticlassClassificationEvaluator()
                .setLabelCol("label").setPredictionCol("prediction").setMetricName("weightedRecall");
        MulticlassClassificationEvaluator accEval = new MulticlassClassificationEvaluator()
                .setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy");

        // Compute metrics
        double f1 = f1Eval.evaluate(predictions);
        double precision = precisionEval.evaluate(predictions);
        double recall = recallEval.evaluate(predictions);
        double accuracy = accEval.evaluate(predictions);

        // Print results with style
        System.out.println("\n\033[1;36m================= Wine Quality Prediction Results =================\033[0m");
        System.out.printf("\033[1;33m%-30s\033[0m %.4f%n", "✔ Accuracy:", accuracy);
        System.out.printf("\033[1;33m%-30s\033[0m %.4f%n", "✔ Weighted Precision:", precision);
        System.out.printf("\033[1;33m%-30s\033[0m %.4f%n", "✔ Weighted Recall:", recall);
        System.out.printf("\033[1;33m%-30s\033[0m %.4f%n", "✔ F1 Score:", f1);
        System.out.println("\033[1;36m==================================================================\033[0m");

        spark.stop();
    }
}
