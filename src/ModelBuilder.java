import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;

public class ModelBuilder {

    public static LinearRegression  buildRegressionModelByGender(Instances data, String gender) throws Exception {
        Instances filteredData = new Instances(data);
        for (int i = data.size() - 1; i >= 0; i--) {
            if (!data.get(i).stringValue(data.attribute("Gender")).equals(gender)) {
                filteredData.remove(i);
            }
        }
        return buildRegressionModel(filteredData,gender);
    }


    private static LinearRegression buildRegressionModel(Instances data ,String gender) throws Exception {
        data.setClassIndex(data.numAttributes() - 1);

        LinearRegression model = new LinearRegression();
        model.buildClassifier(data);

        evaluateModel(model,data,gender);
        return model;
    }

    private static void evaluateModel(LinearRegression model, Instances testData, String gender) throws Exception {
        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(model, testData);

        // Print regression metrics as a table
//        System.out.println("Regression Metrics [" + gender + "] :");
//        System.out.println("+----------------------------------------+");
//        System.out.printf("| %-25s | %-10s |\n", "Metric", "Value");
//        System.out.println("+----------------------------------------+");
//        System.out.printf("| %-25s | %-10.4f |\n", "Mean Absolute Error", eval.meanAbsoluteError());
//        System.out.printf("| %-25s | %-10.4f |\n", "Root Mean Squared Error", eval.rootMeanSquaredError());
//        // Add more metrics as needed
//        System.out.println("+----------------------------------------+");

        // Print detailed evaluation results with a separator line
//        System.out.println("*".repeat(50));
//        System.out.println("Detailed Evaluation Results:");
//        System.out.println(eval.toSummaryString("\nResults\n======", false));
//        System.out.println("*".repeat(50));
    }
}
