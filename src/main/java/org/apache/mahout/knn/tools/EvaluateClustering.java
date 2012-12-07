package org.apache.mahout.knn.tools;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.BallKMeans;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.search.UpdatableSearcher;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.random.WeightedThing;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class EvaluateClustering {
  private static final int NUM_CLUSTERS = 20;
  private static final int MAX_NUM_ITERATIONS = 10;

  private int reducedDimension;

  private Map<String, Centroid> actualClusters = Maps.newHashMap();
  private List<String> inputPaths = Lists.newArrayList();
  private List<Vector> inputVectors = Lists.newArrayList();
  private List<Centroid> reducedVectors = Lists.newArrayList();

  public static List<Object[]> generateData() {
    return Arrays.asList(new Object[][]{
        {"unprojected-tfidf-vectors.seqfile", 50}
    }
    );
  }

  public EvaluateClustering(/*String inPath, int reducedDimension*/) throws IOException {
    this.reducedDimension = 1000;
    getInputVectors("unprojected-tfidf-vectors.seqfile", reducedDimension, inputPaths,
        inputVectors, reducedVectors);
  }

  public static void getInputVectors(String inPath, int reducedDimension,
                                     List<String> inputPaths,
                                     List<Vector> inputVectors,
                                     List<Centroid> reducedVectors) throws IOException {
    System.out.println("Started reading data");
    Path inFile = new Path(inPath);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, inFile, conf);
    Text key = new Text();
    VectorWritable value = new VectorWritable();

    while (reader.next(key, value)) {
      inputPaths.add(key.toString());
      inputVectors.add(value.get().clone());
    }

    int initialDimension = inputVectors.get(0).size();
    List<Vector> basisVectors = ProjectionSearch.generateBasis(initialDimension, reducedDimension);
    int numVectors = 0;
    for (Vector v : inputVectors) {
      Vector reducedVector = new DenseVector(reducedDimension);
      for (int i = 0; i < reducedDimension; ++i) {
        reducedVector.setQuick(i, basisVectors.get(i).dot(v));
      }
      reducedVectors.add(new Centroid(numVectors++, reducedVector, 1));
    }
    System.out.println("Finished reading data");
  }

  public static void computeActualClusters(List<String> inputPaths,
                                           List<Centroid> reducedVectors,
                                           Map<String, Centroid> actualClusters) {
    System.out.println("Starting input vectors clustering.");
    int numClusters = 0;
    for (int i = 0; i < reducedVectors.size(); ++i) {
      String filePath = inputPaths.get(i);
      int lastSlash = filePath.lastIndexOf('/');
      int postNextLastSlash = filePath.lastIndexOf('/', lastSlash - 1) + 1;
      String clusterName = filePath.substring(postNextLastSlash, lastSlash);

      Vector datapoint = reducedVectors.get(i);
      Centroid centroid = actualClusters.get(clusterName);
      if (centroid == null) {
        centroid = new Centroid(numClusters++, datapoint, 1);
        actualClusters.put(clusterName, centroid);
      } else {
        centroid.update(datapoint);
      }
    }
    System.out.println("Finished input vectors clustering.");
  }

  public static BallKMeans createBallKMeans(int numClusters, int maxNumIterations) {
    return new BallKMeans(new BruteSearch(new EuclideanDistanceMeasure()), numClusters,
        maxNumIterations);
  }

  public static BallKMeans createBallKMeans() {
    return createBallKMeans(NUM_CLUSTERS, MAX_NUM_ITERATIONS);
  }

  public static Iterable<Centroid> clusterBallKMeans(List<Centroid> datapoints) {
    BallKMeans clusterer = createBallKMeans();
    clusterer.cluster(datapoints);
    return clusterer;
  }

  public static Iterable<Centroid> clusterStreamingKMeans(Iterable<Centroid> datapoints) {
    /*
    StreamingKMeans clusterer = new StreamingKMeans(new ProjectionSearch(new
        EuclideanDistanceMeasure(), 10, 10), NUM_CLUSTERS, 10e-6);
     */
    StreamingKMeans clusterer = new StreamingKMeans(new BruteSearch(new EuclideanDistanceMeasure()),
        NUM_CLUSTERS, 10e-6);
    clusterer.cluster(datapoints);
    List<Centroid> intermediateCentroids = Lists.newArrayList(clusterer);
    return clusterBallKMeans(intermediateCentroids);
  }

  public static Map<Integer, Integer> countClusterPoints(List<Centroid> datapoints,
                                                         Iterable<Centroid> centroids) {
    UpdatableSearcher searcher = new BruteSearch(new EuclideanDistanceMeasure());
    searcher.addAll(centroids);
    Map<Integer, Integer> centroidMap = Maps.newHashMap();
    for (Centroid v : datapoints) {
      Centroid closest = (Centroid)searcher.search(v,  1).get(0).getValue();
      Integer existingCount = centroidMap.get(closest.getIndex());
      if (existingCount == null) {
        existingCount = new Integer(1);
      } else {
        existingCount = new Integer(existingCount.intValue() + 1);
      }
      centroidMap.put(closest.getIndex(), existingCount);
    }
    return centroidMap;
  }
  
  public static void generateCSVFromVectors(List<? extends Vector> datapoints,
                                            String outPath) throws FileNotFoundException {
    if (datapoints.isEmpty()) {
      return;
    }
    int numDimensions = datapoints.get(0).size();
    PrintStream outputStream = new PrintStream(new FileOutputStream(outPath));
    for (int i = 0; i < numDimensions; ++i) {
      outputStream.printf("x%d", i);
      if (i < numDimensions - 1) {
        outputStream.printf(", ");
      } else {
        outputStream.println();
      }
    }
    for (Vector v : datapoints) {
      Iterator<Vector.Element> vi = v.iterator();
      while (vi.hasNext()) {
        outputStream.printf("%f ", vi.next().get());
        if (vi.hasNext()) {
          outputStream.printf(", ");
        } else {
          outputStream.println();
        }
      }
    }
    outputStream.close();
  }

  public static void generateMMFromVectors(List<? extends Vector> datapoints,
                                           String outPath) throws FileNotFoundException {
    if (datapoints.isEmpty()) {
      return;
    }
    int numDimensions = datapoints.get(0).size();
    PrintStream outputStream = new PrintStream(new FileOutputStream(outPath));
    outputStream.printf("%%%%MatrixMarket matrix coordinate real general\n");
    int numNonZero = 0;
    for (int i = 0; i < datapoints.size(); ++i) {
      Vector datapoint = datapoints.get(i);
      for (int j = 0; j < numDimensions; ++j) {
        double coord = datapoint.get(j);
        if (coord != 0) {
          ++numNonZero;
        }
      }
    }
    outputStream.printf("%d %d %d\n", datapoints.size(), numDimensions, numNonZero);
    for (int i = 0; i < datapoints.size(); ++i) {
      Vector datapoint = datapoints.get(i);
      for (int j = 0; j < numDimensions; ++j) {
        double coord = datapoint.get(j);
        if (coord != 0) {
          outputStream.printf("%d %d %f\n", i + 1, j + 1, coord);
        }
      }
    }
    outputStream.close();
  }

  public static void evaluateCloseness(Iterable<Centroid> centroids,
                                       Map<String, Centroid> actualClusters) {
    BruteSearch searcher = new BruteSearch(new EuclideanDistanceMeasure());
    int numClusters = 0;
    List<String> clusterNames = Lists.newArrayList();
    for (Map.Entry<String, Centroid> entry : actualClusters.entrySet()) {
      clusterNames.add(entry.getKey());
      entry.getValue().setWeight(numClusters++);
      searcher.add(entry.getValue());
    }

    Map<Integer, Integer> actualCounts = Maps.newHashMap();
    for (Centroid centroid : centroids) {
      WeightedThing<Vector> closestPair = searcher.search(centroid, 1).get(0);
      int closestClusterIndex = ((Centroid)closestPair.getValue()).getIndex();

      System.out.printf("Centroid %d closest to actual cluster %d [%s]\n", centroid.getIndex(),
          closestClusterIndex, clusterNames.get(closestClusterIndex));
    }
  }

  public void testGenerateReducedCSV() throws FileNotFoundException {
    generateCSVFromVectors(reducedVectors, "vectors-reduced.csv");
  }

  public void testGenerateInitialMM() throws FileNotFoundException {
    generateMMFromVectors(inputVectors, "vectors-initial.mm");
  }

  public void testGenerateInitialCSV() throws FileNotFoundException {
    generateCSVFromVectors(inputVectors, "vectors-initial2.csv");
  }

  public void testBallKMeans() {
    System.out.println("Clustering with BallKMeans");
    Iterable<Centroid> ballKMeansCentroids = clusterBallKMeans(reducedVectors);
    Map<Integer, Integer> countMap = countClusterPoints(reducedVectors, ballKMeansCentroids);
    for (Map.Entry<Integer, Integer> entry : countMap.entrySet()) {
      System.out.printf("%d: %d\n", entry.getKey(), entry.getValue());
    }
    computeActualClusters(inputPaths, reducedVectors, actualClusters);
    evaluateCloseness(ballKMeansCentroids, actualClusters);
  }

  public void testStreamingKMeans() {
    System.out.println("Clustering with StreamingKMeans");
    Iterable<Centroid> streamingKMeansCentroids = clusterStreamingKMeans(reducedVectors);
    Map<Integer, Integer> countMap = countClusterPoints(reducedVectors, streamingKMeansCentroids);
    for (Map.Entry<Integer, Integer> entry : countMap.entrySet()) {
      System.out.printf("%d: %d\n", entry.getKey(), entry.getValue());
    }
    computeActualClusters(inputPaths, reducedVectors, actualClusters);
    evaluateCloseness(streamingKMeansCentroids, actualClusters);
  }

  public void testInOutCluster() throws FileNotFoundException{
    testGenerateInitialMM();
    testGenerateReducedCSV();
    testBallKMeans();
    testStreamingKMeans();
  }

  public void testAverageDistances() {
    Iterable<Centroid> ballKMeansCentroids = clusterBallKMeans(reducedVectors);
    computeActualClusters(inputPaths, reducedVectors, actualClusters);
    DistanceMeasure distanceMeasure = new EuclideanDistanceMeasure();
    BruteSearch bruteSearch = new BruteSearch(distanceMeasure);
    bruteSearch.addAll(ballKMeansCentroids);

    Map<String, Double> averageRealClusterDistances = Maps.newHashMap();
    Map<Integer, Double> averageComputedClusterDistances = Maps.newHashMap();

    for (int i = 0; i < reducedVectors.size(); ++i) {
      String inputPath = inputPaths.get(i);
      Vector reducedVector = reducedVectors.get(i);

      Double totalRealDistance = averageRealClusterDistances.get(inputPath);
      double newDistance = distanceMeasure.distance(reducedVector, actualClusters.get(inputPath));
      if (totalRealDistance == null) {
        totalRealDistance = new Double(newDistance);
      } else {
        totalRealDistance = new Double(totalRealDistance.doubleValue() + newDistance);
      }
      averageRealClusterDistances.put(inputPath, totalRealDistance);

      WeightedThing<Vector> closestPair = bruteSearch.search(reducedVector, 1).get(0);
      int clusterIndex = ((Centroid)closestPair.getValue()).getIndex();
      Double totalComputedDistance = averageComputedClusterDistances.get(clusterIndex);
      newDistance = closestPair.getWeight();
      if (totalComputedDistance == null) {
        totalComputedDistance = new Double(newDistance);
      } else {
        totalComputedDistance = new Double(totalComputedDistance.doubleValue() + newDistance);
      }
      averageComputedClusterDistances.put(clusterIndex, totalComputedDistance);
    }

  }

  public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    Preconditions.checkArgument(args.length > 2, "Invalid number of arguments. Need input and " +
        "output paths");
    EvaluateClustering tester = new EvaluateClustering();
    tester.testAverageDistances();
  }
}
