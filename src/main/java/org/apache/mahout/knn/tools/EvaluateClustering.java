package org.apache.mahout.knn.tools;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Utils;
import org.apache.mahout.clustering.OnlineGaussianAccumulator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.BallKMeans;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.experimental.CentroidWritable;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.search.UpdatableSearcher;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.random.WeightedThing;
import org.apache.mahout.math.stats.OnlineSummarizer;
import org.slf4j.Logger;

import java.io.*;
import java.util.*;

public class EvaluateClustering {
  private static final int NUM_CLUSTERS = 20;
  private static final int MAX_NUM_ITERATIONS = 10;

  // Map of the actual clusters (the clusters being the means in each OnlineGaussianAccumulator).
  private Map<String, OnlineGaussianAccumulator> actualClusters = Maps.newHashMap();
  // The list of input paths (for each document, its path).
  private List<String> inputPaths = Lists.newArrayList();
  // The list of vectors (for each document, its corresponding feature vector).
  private List<Vector> inputVectors = Lists.newArrayList();
  // The projected vectors.
  private List<Centroid> reducedVectors = Lists.newArrayList();

  private List<Centroid> ballKMeansCentroids = null;
  private List<Centroid> streamingKMeansCentroids = null;

  public static List<Object[]> generateData() {
    return Arrays.asList(new Object[][]{
        {"unprojected-tfidf-vectors.seqfile", 50}
    }
    );
  }

  public EvaluateClustering(String inPath, int reducedDimension) throws IOException, IllegalAccessException, InstantiationException {
    getInputVectors(inPath, reducedDimension, inputPaths,
        inputVectors, reducedVectors);
    computeActualClusters(inputPaths, reducedVectors, actualClusters);
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

    double start = System.currentTimeMillis();
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
    double end = System.currentTimeMillis();
    System.out.printf("Finished reading data; initial dimension %d; projected to %d; took %f\n",
        initialDimension, reducedDimension, end - start);
  }

  public static String getNameBase(String name) {
    int lastSlash = name.lastIndexOf('/');
    int postNextLastSlash = name.lastIndexOf('/', lastSlash - 1) + 1;
    return name.substring(postNextLastSlash, lastSlash);
  }

  public static void computeActualClusters(List<String> inputPaths,
                                           List<Centroid> reducedVectors,
                                           Map<String, OnlineGaussianAccumulator> actualClusters) throws InstantiationException, IllegalAccessException {
    System.out.printf("Started computing actual clusters.\n");
    for (int i = 0; i < reducedVectors.size(); ++i) {
      OnlineGaussianAccumulator actualClusterAccumulator = MapUtils.findOrInitialize
          (actualClusters, OnlineGaussianAccumulator.class, getNameBase(inputPaths.get(i)));
      actualClusterAccumulator.observe(reducedVectors.get(i), 1);
    }
    System.out.printf("Finished computing actual clusters.\n");
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

  public Iterable<Centroid> clusterStreamingKMeans(Iterable<Centroid> datapoints) {
    /*
    StreamingKMeans clusterer = new StreamingKMeans(new ProjectionSearch(new
        EuclideanDistanceMeasure(), 10, 10), NUM_CLUSTERS, 10e-6);
     */
    StreamingKMeans clusterer = new StreamingKMeans(new BruteSearch(new EuclideanDistanceMeasure()),
        NUM_CLUSTERS * (int)Math.log(reducedVectors.size()), 10e-6);
    clusterer.cluster(datapoints);
    List<Centroid> intermediateCentroids = Lists.newArrayList(clusterer);
    return clusterBallKMeans(intermediateCentroids);
  }

  public static List<Integer> countClusterPoints(List<Centroid> datapoints,
                                                 Iterable<Centroid> centroids) {
    UpdatableSearcher searcher = new BruteSearch(new EuclideanDistanceMeasure());
    searcher.addAll(centroids);
    List<Integer> centroidMap = Lists.newArrayList(Collections.nCopies(searcher.size(), 0));
    for (Centroid v : datapoints) {
      Centroid closest = (Centroid)searcher.search(v,  1).get(0).getValue();
      centroidMap.set(closest.getIndex(), centroidMap.get(closest.getIndex()) + 1);
    }
    return centroidMap;
  }

  public static void generateCSVFromVectors(List<? extends Vector> datapoints,
                                            String outPath) throws FileNotFoundException {
    if (datapoints.isEmpty()) {
      return;
    }
    int numDimensions = datapoints.get(0).size();
    PrintStream  outputStream = new PrintStream(new FileOutputStream(outPath));
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
    PrintStream  outputStream = new PrintStream(new FileOutputStream(outPath));
    outputStream.printf("%%%%MatrixMarket matrix coordinate real general\n");
    int numNonZero = 0;
    for (Vector datapoint : datapoints) {
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

  public void testGenerateReducedCSV() throws FileNotFoundException {
    generateCSVFromVectors(reducedVectors, "vectors-reduced.csv");
  }

  public void testGenerateInitialMM() throws FileNotFoundException {
    generateMMFromVectors(inputVectors, "vectors-initial.mm");
  }

  public void testGenerateInitialCSV() throws FileNotFoundException {
    generateCSVFromVectors(inputVectors, "vectors-initial2.csv");
  }

  public void printClusterCounts(List<Integer> countMap) {
    for (int i = 0; i < countMap.size(); ++i) {
      System.out.printf("%d: %d\n", i, countMap.get(i));
    }
  }

  public void testBallKMeans() throws IllegalAccessException, InstantiationException {
    System.out.println("Clustering with BallKMeans");
    ballKMeansCentroids = Lists.newArrayList(clusterBallKMeans(reducedVectors));
    List<Integer> countMap = countClusterPoints(reducedVectors, ballKMeansCentroids);
    printClusterCounts(countMap);
    computeAverageDistance(ballKMeansCentroids);
  }

  public void testStreamingKMeans() throws IllegalAccessException, InstantiationException {
    System.out.println("Clustering with StreamingKMeans");
    streamingKMeansCentroids = Lists.newArrayList(clusterBallKMeans(Lists.newArrayList
        (clusterStreamingKMeans(reducedVectors))));
    List<Integer> countMap = countClusterPoints(reducedVectors, streamingKMeansCentroids);
    printClusterCounts(countMap);
    computeAverageDistance(streamingKMeansCentroids);
  }

  public void testInOutCluster() throws FileNotFoundException, InstantiationException, IllegalAccessException {
    testGenerateInitialMM();
    testGenerateReducedCSV();
    testBallKMeans();
    testStreamingKMeans();
  }

  public void computeAverageDistanceForActualClusters() {
    Preconditions.checkArgument(!actualClusters.isEmpty(), "Run computeActualClusters() before " +
        "computing the average distances");
    OnlineSummarizer realClusterSummarizer = new OnlineSummarizer();
    System.out.printf("Standard deviations for real clusters:\n");
    for (Map.Entry<String, OnlineGaussianAccumulator> entry : actualClusters.entrySet()) {
      double clusterAvgStd = entry.getValue().getAverageStd();
      System.out.printf("%s: %f\n", entry.getKey(), clusterAvgStd);
      realClusterSummarizer.add(clusterAvgStd);
    }
    System.out.printf("Average deviation for real clusters: %f\n\n",
        realClusterSummarizer.getMean());

  }

  public void computeAverageDistance(Iterable<Centroid> centroids) throws InstantiationException, IllegalAccessException {
    System.out.printf("Computing average distance for computed clusters\n\n");
    DistanceMeasure distanceMeasure = new EuclideanDistanceMeasure();
    BruteSearch bruteSearch = new BruteSearch(distanceMeasure);
    bruteSearch.addAll(centroids);

    // These pairs go from the cluster identifier to a Gaussian accumulator to get the average
    // of the standard deviations of the components in each cluster.
    Map<Integer, OnlineGaussianAccumulator> averageComputedClusterDistances = Maps.newHashMap();

    for (Centroid reducedVector : reducedVectors) {
      // Get the index of the closest computed cluster for this vector.
      WeightedThing<Vector> closestPair = bruteSearch.search(reducedVector, 1).get(0);
      int clusterIndex = ((Centroid) closestPair.getValue()).getIndex();
      OnlineGaussianAccumulator computedAccumulator = MapUtils.findOrInitialize
          (averageComputedClusterDistances, OnlineGaussianAccumulator.class, clusterIndex);
      computedAccumulator.observe(reducedVector, 1);
    }

    OnlineSummarizer computedClusterSummarizer = new OnlineSummarizer();
    System.out.printf("Standard deviations for computed clusters:\n");
    for (Map.Entry<Integer, OnlineGaussianAccumulator> entry : averageComputedClusterDistances.entrySet()) {
      double clusterAvgStd = entry.getValue().getAverageStd();
      System.out.printf("%d: %f\n", entry.getKey(), clusterAvgStd);
      computedClusterSummarizer.add(clusterAvgStd);
    }
    System.out.printf("Average deviation for computed clusters: %f\n",
        computedClusterSummarizer.getMean());
  }

  public void compareCentroids() {
    Preconditions.checkNotNull(ballKMeansCentroids, "Run testBallKMeans() first");
    Preconditions.checkNotNull(streamingKMeansCentroids, "Run testStreamingKMeans() first");

    BruteSearch searcher = new BruteSearch(new EuclideanDistanceMeasure());
    searcher.addAll(ballKMeansCentroids);

    List<Integer> centroidsComparison = countClusterPoints(streamingKMeansCentroids,
        ballKMeansCentroids);
    System.out.printf("Number of streamingKMeans centroids: %d\n", streamingKMeansCentroids.size());
    System.out.printf("Number of ballKMeans centroids: %d\n", ballKMeansCentroids.size());

    System.out.printf("Number of streamingKMeans centroids for each ballKMeans centroid:\n");
    printClusterCounts(centroidsComparison);
  }

  public void printClusterPredictions(Iterable<Centroid> centroids) {
    int numVectors = reducedVectors.size();
    BruteSearch searcher = new BruteSearch(new EuclideanDistanceMeasure());
    searcher.addAll(centroids);
    Map<Integer, Map<String, Integer>> distribution = Maps.newHashMap();
    for (int i = 0; i < numVectors; ++i) {
      Vector vector = reducedVectors.get(i);
      String clusterName = getNameBase(inputPaths.get(i));
      Centroid closest = (Centroid)(searcher.search(vector, 1).get(0).getValue());

      Map<String, Integer> partitionMap = distribution.get(closest.getIndex());
      if (partitionMap == null) {
        partitionMap = Maps.newHashMap();
        distribution.put(closest.getIndex(), partitionMap);
      }
      MapUtils.findAndApplyFunctionOrInitialize(partitionMap, new MapUtils.PlusOne(), 1,
          clusterName);
    }

    for (Map.Entry<Integer, Map<String, Integer>> partitionEntry : distribution.entrySet()) {
      float total = 0;
      for (Integer count : partitionEntry.getValue().values()) {
        total += count;
      }
      System.out.printf("%d: [%d | %f] ", partitionEntry.getKey(),
          partitionEntry.getValue().size(), total);
      for (Map.Entry<String, Integer> entry : partitionEntry.getValue().entrySet()) {
        System.out.printf("(%s %f) ", entry.getKey(), entry.getValue() / total);
      }
      System.out.printf("\n");
    }
  }

  public static void summarize(Configuration outputConf, Path outputPath,
                               Logger log) throws IOException {
    FileSystem outputFs = FileSystem.get(outputConf);
    Path outputPaths[];
    if (outputFs.getFileStatus(outputPath).isDir()) {
      outputPaths = FileUtil.stat2Paths(outputFs.listStatus(outputPath,
          new Utils.OutputFileUtils.OutputFilesFilter()));
    } else {
      outputPaths = new Path[1];
      outputPaths[0] = outputPath;
    }

    OnlineSummarizer weightSummarizer = new OnlineSummarizer();
    int numClusters = 0;
    int totalWeight = 0;

    for (Path path : outputPaths) {
      SequenceFile.Reader reader = new SequenceFile.Reader(outputFs, path, outputConf);
      IntWritable key = new IntWritable();
      CentroidWritable value = new CentroidWritable();

      while (reader.next(key, value)) {
        ++numClusters;
        Centroid centroid = value.getCentroid().clone();
        totalWeight += (int)centroid.getWeight();
        weightSummarizer.add(centroid.getWeight());
      }
      reader.close();
    }

    log.info("Clustered {} points into {} clusters\n", totalWeight, numClusters);
    log.info("Average number of points per cluster is {} with deviation {}\n",
        weightSummarizer.getMean(), weightSummarizer.getSD());
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException, IllegalAccessException, InstantiationException {
    Preconditions.checkArgument(args.length == 2, "Invalid number of arguments. Need input and " +
        "output paths " + args.length);
    String inputPath = args[0];
    int numReducedDimension = Integer.parseInt(args[1]);
    EvaluateClustering tester = new EvaluateClustering(inputPath, numReducedDimension);

    System.out.println("Running a set of tests on the quality of StreamingKMeans for the given " +
        "sequence file.");
    tester.computeAverageDistanceForActualClusters();

    tester.testBallKMeans();
    tester.testStreamingKMeans();

    tester.compareCentroids();

    System.out.println("BallKMeans predictions");
    tester.printClusterPredictions(tester.ballKMeansCentroids);

    System.out.println("StreamingKMeans predictions");
    tester.printClusterPredictions(tester.streamingKMeansCentroids);
  }
}
