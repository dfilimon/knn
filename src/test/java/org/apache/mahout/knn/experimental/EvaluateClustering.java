package org.apache.mahout.knn.experimental;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.BallKMeans;
import org.apache.mahout.knn.cluster.DataUtils;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.knn.search.UpdatableSearcher;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class EvaluateClustering {
  private static final int NUM_CLUSTERS = 20;
  private static final int MAX_NUM_ITERATIONS = 10;

  /**
   * Reads in a sequence file of the vectorized documents. These vectors have very high dimension
   * which we reduce to some specified dimension by projecting on some fixed number of vectors.
   * @param inPath
   * @return
   */
  public static Pair<List<Vector>, List<Centroid>> getInputVectors(String inPath,
                                                                   int reducedDimension) throws IOException {
    List<Vector> inputVectors = Lists.newArrayList();

    Path inFile = new Path(inPath);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, inFile, conf);
    Text key = new Text();
    VectorWritable value = new VectorWritable();

    while (reader.next(key, value)) {
      inputVectors.add(value.get().clone());
    }
    int initialDimension = inputVectors.get(0).size();

    List<Vector> basisVectors = ProjectionSearch.generateBasis(initialDimension, reducedDimension);
    List<Centroid> reducedVectors = Lists.newArrayList();
    int numVectors = 0;
    for (Vector v : inputVectors) {
      Vector reducedVector = new DenseVector(reducedDimension);
      for (int i = 0; i < reducedDimension; ++i) {
        reducedVector.setQuick(i, basisVectors.get(i).dot(v));
      }
      reducedVectors.add(new Centroid(numVectors++, reducedVector, 1));
    }

    return new Pair<List<Vector>, List<Centroid>>(inputVectors, reducedVectors);
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
      for (int i = 0; i < numDimensions; ++i) {
        outputStream.printf("%f", v.getQuick(i));
        if (i < numDimensions - 1) {
          outputStream.printf(", ");
        } else {
          outputStream.println();
        }
      }
    }
    outputStream.close();
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    System.out.println("Processing input vectors");
    Pair<List<Vector>, List<Centroid>> returnPair = getInputVectors(args[0], 50);
    List<Vector> inputVectors = returnPair.getFirst();
    List<Centroid> reducedVectors = returnPair.getSecond();

    System.out.println("Total number of vectors " + inputVectors.size());
    Preconditions.checkArgument(inputVectors.size() == reducedVectors.size(),
        "Different number of reduced vectors", inputVectors.size(), reducedVectors.size());

    System.out.println("Clustering with BallKMeans");
    Iterable<Centroid> ballKMeansCentroids = clusterBallKMeans(reducedVectors);
    Map<Integer, Integer> countMap = countClusterPoints(reducedVectors, ballKMeansCentroids);
    for (Map.Entry<Integer, Integer> entry : countMap.entrySet()) {
      System.out.printf("%d: %d\n", entry.getKey(), entry.getValue());
    }

    System.out.println("Clustering with StreamingKMeans");
    Iterable<Centroid> streamingKMeansCentroids = clusterStreamingKMeans(reducedVectors);
    countMap = countClusterPoints(reducedVectors, streamingKMeansCentroids);
    for (Map.Entry<Integer, Integer> entry : countMap.entrySet()) {
      System.out.printf("%d: %d\n", entry.getKey(), entry.getValue());
    }

    System.out.println("Generating CSV file from vectors");
    generateCSVFromVectors(reducedVectors, args[0] + "-reduced");
  }
}
