import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.DatasetRenderingOrder;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.renderer.xy.XYShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.io.File;

public class Main extends JFrame {

    public Main() {
        JTabbedPane tabbedPane = new JTabbedPane();

        System.setProperty("com.github.fommil.netlib.ARPACK", "com.github.fommil.netlib.F2jARPACK");
        System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
        System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");

        try {
            // Load the dataset from CSV file
            CSVLoader loader = new CSVLoader();
            loader.setFile(new File("data/Height_Weight.csv"));
            Instances data = loader.getDataSet();

            // Convert height from inches to centimeters and weight from pounds to kilograms
            DataProcessor.convertHeightAndWeight(data);
            System.out.println("Data Conversion Complete\n");

            // Display statistics for each attribute
            DataProcessor.displayStatistics(data);
            System.out.println();

            // Model 1
            Instances dataM1 = DataProcessor.selectRandomSubset(data, 100);
            Instances[] splitsM1 = DataProcessor.splitData(dataM1);
            Instances trainSetDataM1 = splitsM1[0];
            Instances testSetDataM1 = splitsM1[1];
            LinearRegression maleModel1 = ModelBuilder.buildRegressionModelByGender(trainSetDataM1, "Male");
            LinearRegression femaleModel1 = ModelBuilder.buildRegressionModelByGender(trainSetDataM1, "Female");
            JPanel panel1 = createCombinedChart(trainSetDataM1, maleModel1, femaleModel1, testSetDataM1, "Model 1");

            tabbedPane.addTab("Model 1", panel1);

            // Model 2
            Instances dataM2 = DataProcessor.selectRandomSubset(data, 1000);
            Instances[] splitsM2 = DataProcessor.splitData(dataM2);
            Instances trainSetDataM2 = splitsM2[0];
            Instances testSetDataM2 = splitsM2[1];
            LinearRegression maleModel2 = ModelBuilder.buildRegressionModelByGender(trainSetDataM2, "Male");
            LinearRegression femaleModel2 = ModelBuilder.buildRegressionModelByGender(trainSetDataM2, "Female");
            JPanel panel2 = createCombinedChart(trainSetDataM2, maleModel2, femaleModel2, testSetDataM2, "Model 2");

            tabbedPane.addTab("Model 2", panel2);

            // Model 3
            Instances dataM3 = DataProcessor.selectRandomSubset(data, 5000);
            Instances[] splitsM3 = DataProcessor.splitData(dataM3);
            Instances trainSetDataM3 = splitsM3[0];
            Instances testSetDataM3 = splitsM3[1];
            LinearRegression maleModel3 = ModelBuilder.buildRegressionModelByGender(trainSetDataM3, "Male");
            LinearRegression femaleModel3 = ModelBuilder.buildRegressionModelByGender(trainSetDataM3, "Female");
            JPanel panel3 = createCombinedChart(trainSetDataM3, maleModel3, femaleModel3, testSetDataM3, "Model 3");

            tabbedPane.addTab("Model 3", panel3);

            // Model 4
            Instances dataM4 = DataProcessor.selectRandomSubset(data, data.numInstances());
            Instances[] splitsM4 = DataProcessor.splitData(dataM4);
            Instances trainSetDataM4 = splitsM4[0];
            Instances testSetDataM4 = splitsM4[1];
            LinearRegression maleModel4 = ModelBuilder.buildRegressionModelByGender(trainSetDataM4, "Male");
            LinearRegression femaleModel4 = ModelBuilder.buildRegressionModelByGender(trainSetDataM4, "Female");
            JPanel panel4 = createCombinedChart(trainSetDataM4, maleModel4, femaleModel4, testSetDataM4, "Model 4");

            tabbedPane.addTab("Model 4", panel4);

            add(tabbedPane);
            setTitle("Combined Scatter Plot and Regression Lines");
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            pack();
            setLocationRelativeTo(null);
            setVisible(true);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public JPanel createCombinedChart(Instances data, LinearRegression maleModel, LinearRegression femaleModel, Instances testData, String modelName) {
        XYSeries scatterSeries = new XYSeries("");
        XYSeries maleRegressionSeries = new XYSeries("Male Regression Line");
        XYSeries femaleRegressionSeries = new XYSeries("Female Regression Line");
        XYSeries maleSeries = new XYSeries("Male");
        XYSeries femaleSeries = new XYSeries("Female");

        int heightIndex = data.attribute("Height").index();
        int weightIndex = data.attribute("Weight").index();
        int genderIndex = data.attribute("Gender").index();

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            double height = instance.value(heightIndex);
            double weight = instance.value(weightIndex);
            String gender = instance.stringValue(genderIndex);

            scatterSeries.add(height, weight);

            if (gender.equals("Male")) {
                maleSeries.add(height, weight);
                double predictedWeight = maleModel.coefficients()[3] + (maleModel.coefficients()[1] * height);
                maleRegressionSeries.add(height, predictedWeight);
            } else if (gender.equals("Female")) {
                femaleSeries.add(height, weight);
                double predictedWeight = femaleModel.coefficients()[3] + (femaleModel.coefficients()[1] * height);
                femaleRegressionSeries.add(height, predictedWeight);
            }
        }

        XYDataset scatterDataset = new XYSeriesCollection(scatterSeries);
        XYDataset maleRegressionDataset = new XYSeriesCollection(maleRegressionSeries);
        XYDataset femaleRegressionDataset = new XYSeriesCollection(femaleRegressionSeries);
        XYDataset maleDataset = new XYSeriesCollection(maleSeries);
        XYDataset femaleDataset = new XYSeriesCollection(femaleSeries);

        DataProcessor.evaluateModel(testData, maleModel, "Male", modelName + " (Male)");
        DataProcessor.evaluateModel(testData, femaleModel, "Female", modelName + " (Female)");

        JFreeChart chart = ChartFactory.createScatterPlot(
                "Scatter Plot with Regression Lines: Height vs Weight",
                "Height (cm)",
                "Weight (kg)",
                scatterDataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        XYShapeRenderer scatterPointRenderer = createPointRenderer(Color.BLACK, new Ellipse2D.Double(-1, -1, 3, 3));
        XYLineAndShapeRenderer maleLineRenderer = createLineRenderer(Color.YELLOW,20.0f);
        XYLineAndShapeRenderer femaleLineRenderer = createLineRenderer(Color.RED,20.0f);
        XYShapeRenderer malePointRenderer = createPointRenderer(Color.BLUE, new Ellipse2D.Double(-2, -2, 4, 4));
        XYShapeRenderer femalePointRenderer = createPointRenderer(Color.GREEN, new Ellipse2D.Double(-2, -2, 4, 4));

        scatterPointRenderer.setSeriesVisible(0, false);
        maleLineRenderer.setSeriesShapesVisible(0, false);
        femaleLineRenderer.setSeriesShapesVisible(0, false);

        setDatasetAndRenderer(chart, 0, scatterDataset, scatterPointRenderer);
        setDatasetAndRenderer(chart, 1, maleDataset, malePointRenderer);
        setDatasetAndRenderer(chart, 2, femaleDataset, femalePointRenderer);
        setDatasetAndRenderer(chart, 3, maleRegressionDataset, maleLineRenderer);
        setDatasetAndRenderer(chart, 4, femaleRegressionDataset, femaleLineRenderer);

        XYPlot plot = chart.getXYPlot();
        plot.setDataset(3, maleRegressionDataset);
        plot.setRenderer(3, maleLineRenderer);
        plot.setDataset(4, femaleRegressionDataset);
        plot.setRenderer(4, femaleLineRenderer);
        plot.setDataset(1, maleDataset);
        plot.setRenderer(1, malePointRenderer);
        plot.setDataset(2, femaleDataset);
        plot.setRenderer(2, femalePointRenderer);

        plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);
        plot.setRenderer(3, new XYLineAndShapeRenderer(true, false));
        plot.setRenderer(4, new XYLineAndShapeRenderer(true, false));

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(1000, 600));
        JPanel panel = new JPanel();
        panel.add(chartPanel);
        return panel;
    }

    private XYShapeRenderer createPointRenderer(Paint paint, Shape shape) {
        XYShapeRenderer renderer = new XYShapeRenderer();
        renderer.setBasePaint(paint);
        renderer.setSeriesShape(0, shape);
        return renderer;
    }

    private XYLineAndShapeRenderer createLineRenderer(Paint paint ,float strokeWidth) {
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(true, false);
        renderer.setSeriesPaint(0, paint);

        BasicStroke stroke = new BasicStroke(strokeWidth);
        renderer.setSeriesStroke(0, stroke);
        return renderer;
    }

    private void setDatasetAndRenderer(JFreeChart chart, int index, XYDataset dataset, XYItemRenderer renderer) {
        chart.getXYPlot().setDataset(index, dataset);
        chart.getXYPlot().setRenderer(index, renderer);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Main::new);
    }
}
