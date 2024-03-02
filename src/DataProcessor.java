import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

public class DataProcessor {
    public DataProcessor() {
    }

    public static void convertHeightAndWeight(Instances data) {
        int heightIndex = 1;
        int weightIndex = 2;
        double inchesToCm = 2.54;
        double poundsToKg = 0.453592;

        for(int i = 0; i < data.numInstances(); ++i) {
            Instance instance = data.instance(i);
            double heightInInches = instance.value(heightIndex);
            double weightInPounds = instance.value(weightIndex);
            if (!Double.isNaN(heightInInches) && !Double.isNaN(weightInPounds)) {
                double heightInCm = heightInInches * inchesToCm;
                double weightInKg = weightInPounds * poundsToKg;
                instance.setValue(heightIndex, heightInCm);
                instance.setValue(weightIndex, weightInKg);
            } else {
                System.out.println("Instance " + i + " - Invalid values detected (NaN). Skipping conversion.");
                System.out.println();
            }
        }

    }

    public static void displayStatistics(Instances data) {
        System.out.println("Feature Statistics:");
        System.out.println("+---------------+---------------+---------------+---------------+---------------+---------------+");
        System.out.printf("| %-13s | %-13s | %-13s | %-13s | %-13s | %-13s |\n", "Attribute", "Mean", "Median", "StdDev", "Min", "Max");
        System.out.println("+---------------+---------------+---------------+---------------+---------------+---------------+");

        for(int i = 1; i < data.numAttributes(); ++i) {
            if (i != data.classIndex()) {
                String attributeName = data.attribute(i).name();
                if ("Height".equals(attributeName) || "Weight".equals(attributeName)) {
                    double[] values = data.attributeToDoubleArray(i);
                    Arrays.sort(values);
                    int middle = values.length / 2;
                    double median;
                    if (values.length % 2 == 0) {
                        median = (values[middle - 1] + values[middle]) / 2.0;
                    } else {
                        median = values[middle];
                    }

                    System.out.printf("| %-13s | %-13.2f | %-13.2f | %-13.2f | %-13.2f | %-13.2f |\n", attributeName, data.attributeStats(i).numericStats.mean, median, data.attributeStats(i).numericStats.stdDev, data.attributeStats(i).numericStats.min, data.attributeStats(i).numericStats.max);
                    System.out.println("+---------------+---------------+---------------+---------------+---------------+---------------+");
                }
            }
        }
    }

    public static Instances[] splitData(Instances data) {
        int trainSize = (int)Math.round((double)data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        Instances[] result = new Instances[]{train, test};
        return result;
    }

    public static Instances selectRandomSubset(Instances data, int subsetSize) throws Exception {
        if (subsetSize > 0 && subsetSize <= data.size()) {
            List<Integer> instanceIndices = new ArrayList();

            for(int i = 0; i < data.size(); ++i) {
                instanceIndices.add(i);
            }

            Collections.shuffle(instanceIndices, new Random());
            Instances randomSubset = new Instances(data, 0, 0);

            for(int i = 0; i < subsetSize; ++i) {
                int randomIndex = (Integer)instanceIndices.get(i);
                randomSubset.add(data.instance(randomIndex));
            }

            return randomSubset;
        } else {
            throw new IllegalArgumentException("Invalid subset size");
        }
    }

    public static void evaluateModel(Instances testData, LinearRegression model, String genderToEvaluate, String modelName) {
        // Evaluate the model on the test set for a specific gender
        double sumSquaredError = 0.0;
        double sumAbsoluteError = 0.0;
        int numInstances = testData.numInstances();

        int heightIndex = testData.attribute("Height").index();
        int weightIndex = testData.attribute("Weight").index();
        int genderIndex = testData.attribute("Gender").index();

        for (int i = 0; i < numInstances; i++) {
            Instance instance = testData.instance(i);
            double height = instance.value(heightIndex);
            double weight = instance.value(weightIndex);
            String gender = instance.stringValue(genderIndex);

            if (!gender.equals(genderToEvaluate)) {
                continue;  // Skip instances of the opposite gender
            }

            double actualWeight = weight;

            double predictedWeight = model.coefficients()[3] + (model.coefficients()[1] * height);

            // Calculate errors
            double squaredError = Math.pow(actualWeight - predictedWeight, 2);
            double absoluteError = Math.abs(actualWeight - predictedWeight);

            sumSquaredError += squaredError;
            sumAbsoluteError += absoluteError;
        }

        // Calculate metrics
        double meanSquaredError = sumSquaredError / numInstances;
        double meanAbsoluteError = sumAbsoluteError / numInstances;
        double rootMeanSquaredError = Math.sqrt(meanSquaredError);

        // Print performance metrics
        System.out.println(modelName + " Performance Metrics:");
        System.out.println("Mean Absolute Error: " + meanAbsoluteError);
        System.out.println("Root Mean Squared Error: " + rootMeanSquaredError);
        System.out.println("-----------------------------------------");
    }
}
