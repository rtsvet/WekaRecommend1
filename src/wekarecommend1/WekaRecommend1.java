package wekarecommend1;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.rules.ZeroR;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Debug;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class WekaRecommend1 {

    private static final String DATAFILE = "pref-train.arff";
    private static final String TESTFILE = "pref-test.arff";

    public static void main(final String[] args) throws Exception {
        System.out.println("Running");

        // Read Train Data
        Instances data = new Instances(new BufferedReader(new FileReader(DATAFILE)));
        data.setClassIndex(data.numAttributes() - 1);

        // Read Test Data
        Instances dataTest = new Instances(new BufferedReader(new FileReader(TESTFILE)));
        dataTest.setClassIndex(dataTest.numAttributes() - 1);
        // train clasiffier
        final Classifier cf = new LinearRegression();
        cf.buildClassifier(data);

        // make predictions
        for (int i = 0; i < dataTest.numInstances(); i++) {
            double pred = cf.classifyInstance(dataTest.instance(i));
            System.out.print("Prod_ID: " + dataTest.instance(i).value(1));
            System.out.print(", actual: " + data.instance(i).classValue());
            System.out.println(", predicted: " +  pred);
        }



    }
}
