package wekarecommend1;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
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
import weka.core.EuclideanDistance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.neighboursearch.*;

public class WekaRecommend1 {

    private static final String DATAFILE = "pref-nominal-train.arff";
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

        NearestNeighbourSearch nnS = new LinearNNSearch();
        nnS.setDistanceFunction(new EuclideanDistance());
        IBk cf = new IBk(1);
        
        cf.setNearestNeighbourSearchAlgorithm(nnS);
        
        System.out.println("Neares Neighbour " + cf.getNearestNeighbourSearchAlgorithm() + " distance func " + nnS.getDistanceFunction().getClass());


        cf.buildClassifier(data);

        // make predictions
        for (int i = 0; i < dataTest.numInstances(); i++) {
            double pred = cf.classifyInstance(dataTest.instance(i));
            System.out.print("Prod_ID: " + dataTest.instance(i).value(1));
            System.out.println(", predicted: " + dataTest.classAttribute().value((int) pred) + " pred val=" + pred);
        }



    }
}
