/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.knn.tools;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.*;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.cli.CommandLine;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.sgd.*;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.knn.cluster.BallKMeans;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.LoggerFactory;

/**
 * Reads and trains an adaptive logistic regression model on the 20 newsgroups data.
 * The features used are the distances to the centers of the 20 newsgroup clusters.
 * These are computed from the current vector to:
 * <ul>
 *  <li>
 *    the centroids of the 20 clusters computed using the path-based assignment.
 *  </li>
 *  <li>
 *    the centroids of the 20 clusters computed using ball k-means clustering;
 *  </li>
 *  <li>
 *    the centroids of the ~86 clusters (k log n) computed using streaming k-means (without a
 *    ball k-means step) to get back to 20 clusters;
 *  </li>
 * </ul>
 */

public final class TrainNewsGroupsKMeansLogisticRegression {
  /**
   * Constants are set knowing what the 20 newsgroups data set looks like.
   */
  private static final int NUM_VECTORS = 18898;
  private static final int NUM_CLASSES = 20;
  private static final int NUM_FEATURES_ACTUAL = 20;
  private static final int NUM_FEATURES_BKM = 20;
  private static final int NUM_FEATURES_SKM = NUM_CLASSES * (int)Math.log(NUM_VECTORS);

  private TrainNewsGroupsKMeansLogisticRegression() {
  }

  public static void computeActualClusters(Iterable<Pair<Text, VectorWritable>> dirIterable,
                                           Map<String, Centroid> actualClusters) {
    int clusterId = 0;
    for (Pair<Text, VectorWritable> pair : dirIterable) {
      String clusterName = pair.getFirst().toString();
      Centroid centroid = actualClusters.get(clusterName);
      if (centroid == null) {
        centroid = new Centroid(++clusterId, pair.getSecond().get().clone(), 1);
        actualClusters.put(clusterName, centroid);
        continue;
      }
      centroid.update(pair.getSecond().get());
    }
  }

  public static void trainActual(Iterable<Pair<Text, VectorWritable>> inputIterable, String outBase,
                                 Map<String, Integer> clusterNamesToIds) throws  IOException {
    Map<String, Centroid> actualClusters = Maps.newHashMap();
    computeActualClusters(inputIterable, actualClusters);

    OnlineLogisticRegression learningAlgorithm =
        new OnlineLogisticRegression(NUM_CLASSES, NUM_FEATURES_ACTUAL, new L1());

    int vectorId = 0;
    for (Pair<Text, VectorWritable> pair : inputIterable) {
      Vector actualCentroid = pair.getSecond().get();
      Vector features = new DenseVector(NUM_FEATURES_ACTUAL);
      int i = 0;
      for (Centroid centroid : actualClusters.values()) {
        features.set(i++, centroid.getDistanceSquared(actualCentroid));
      }
      String clusterName = pair.getFirst().toString();
      learningAlgorithm.train(clusterNamesToIds.get(clusterName), features);
      ++vectorId;
      if (vectorId % 100 == 0) {
        System.out.printf("[actual] Training %f complete\n", (float)vectorId / NUM_VECTORS);
      }
    }
    learningAlgorithm.close();

    ModelSerializer.writeBinary(outBase + "-append.model", learningAlgorithm);
  }

  public static void trainComputed(Iterable<Pair<Text, VectorWritable>> inputIterable,
                                   String outBase, String suffix,
                                   Map<String, Integer> clusterNamesToIds,
                                   Pair<Integer, Iterable<Centroid>> numFeaturesCentroidsPair) throws IOException {
    System.out.printf("[%s] Starting training\n", suffix);
    int numFeatures = numFeaturesCentroidsPair.getFirst();
    AdaptiveLogisticRegression learningAlgorithm =
        new AdaptiveLogisticRegression(NUM_CLASSES, numFeatures, new L1());

    int vectorId = 0;
    for (Pair<Text, VectorWritable> pair : inputIterable) {
      Vector actualCentroid = pair.getSecond().get();
      Vector features = new DenseVector(numFeatures);
      int i = 0;
      for (Centroid centroid : numFeaturesCentroidsPair.getSecond()) {
        features.set(i++, centroid.getDistanceSquared(actualCentroid));
      }
      String clusterName = pair.getFirst().toString();
      learningAlgorithm.train(clusterNamesToIds.get(clusterName), features);
      ++vectorId;
      if (vectorId % 100 == 0) {
        System.out.printf("[%s] Training %f complete\n", suffix, (float)vectorId / NUM_VECTORS);
      }
    }
    learningAlgorithm.close();

    ModelSerializer.writeBinary(outBase + "-" + suffix + ".model",
        learningAlgorithm.getBest().getPayload().getLearner().getModels().get(0));
  }

  public static Iterable<Centroid> getCentroidsFromSeqFileIterable(
      Iterable<Pair<Text, VectorWritable>> dirIterable) {
    return Iterables.transform(dirIterable, new Function<Pair<Text, VectorWritable>, Centroid>() {
      private int count = 0;

      @Override
      public Centroid apply(Pair<Text, VectorWritable> input) {
        Preconditions.checkNotNull(input);
        return new Centroid(count++, input.getSecond().get().clone(), 1);
      }
    });
  }

  public static Pair<Integer, Iterable<Centroid>> clusterBallKMeans(
      Iterable<Centroid>  dataPoints) {
    List<Centroid> dataPointsList = Lists.newArrayList(dataPoints);
    BallKMeans clusterer = new BallKMeans(new BruteSearch(new EuclideanDistanceMeasure()),
        NUM_FEATURES_BKM, 20);
    clusterer.cluster(dataPointsList);
    return new Pair<Integer, Iterable<Centroid>>(NUM_FEATURES_BKM, clusterer);
  }

  public static Pair<Integer, Iterable<Centroid>> clusterStreamingKMeans(Iterable<Centroid> dataPoints) {
    ((LoggerContext)LoggerFactory.getILoggerFactory()).getLogger(StreamingKMeans.class).setLevel(Level.INFO);
    StreamingKMeans clusterer = new StreamingKMeans(new ProjectionSearch(new
        EuclideanDistanceMeasure(), 20, 10), NUM_FEATURES_SKM, 10e-6);
    clusterer.cluster(dataPoints);
    return new Pair<Integer, Iterable<Centroid>>(clusterer.getCentroids().size(), clusterer);
  }

  public static void main(String[] args) throws IOException, ParseException {
    Options options = new Options();
    options.addOption("i", "input", true, "Path to the input folder containing the training set's" +
        " sequence files.");
    options.addOption("o", "output", true, "Base path to the output file. The name will be " +
        "appended with a suffix for each type of training.");
    options.addOption("a", "actual", false, "If set, runs the training with the actual cluster " +
        "assignments and outputs the model to the output path with a -actual suffix.");
    options.addOption("b", "ballkmeans", false, "If set, runs the training with the ball k-means " +
        "cluster assignments and outputs the model to the output path with a -ballkmeans suffix.");
    options.addOption("s", "streamingkmeans", false, "If set, runs the training with the " +
        "streaming k-means cluster assignments and outputs the model to the output path with a " +
        "-streamingkmeans suffix.");

    CommandLine cmd = (new PosixParser()).parse(options, args);

    String inputPath = cmd.getOptionValue("input");
    Preconditions.checkNotNull(inputPath);

    String outputBase = cmd.getOptionValue("output");
    Preconditions.checkNotNull(outputBase);

    Configuration conf = new Configuration();
    SequenceFileDirIterable<Text, VectorWritable> inputIterable = new
        SequenceFileDirIterable<Text, VectorWritable>(new Path(inputPath), PathType.LIST, conf);

    PrintStream clusterIdOut = new PrintStream(new FileOutputStream("cluster-ids.csv"));
    clusterIdOut.printf("clusterName, clusterId\n");
    int clusterId = 0;
    Map<String, Integer> clusterNamesToIds = Maps.newHashMapWithExpectedSize(NUM_CLASSES);
    for (Pair<Text, VectorWritable> pair : inputIterable) {
      String clusterName = pair.getFirst().toString();
      if (!clusterNamesToIds.containsKey(clusterName)) {
        clusterIdOut.printf("%s, %d\n", clusterName, clusterId);
        clusterNamesToIds.put(clusterName, clusterId++);
      }
    }
    clusterIdOut.close();

    if (cmd.hasOption("actual")) {
      System.out.printf("\nActual clusters models\n");
      System.out.printf("----------------------\n");
      long start = System.currentTimeMillis();
      trainActual(inputIterable, outputBase,clusterNamesToIds);
      long end = System.currentTimeMillis();
      System.out.printf("Trained models for actual clusters. Took %d ms\n", end - start);
    }

    if (cmd.hasOption("ballkmeans")) {
      System.out.printf("\nBall k-means clusters models\n");
      System.out.printf("----------------------------\n");
      long start = System.currentTimeMillis();
      trainComputed(inputIterable, outputBase, "ballkmeans", clusterNamesToIds,
          clusterBallKMeans(getCentroidsFromSeqFileIterable(inputIterable)));
      long end = System.currentTimeMillis();
      System.out.printf("Trained models for ballkmeans clusters. Took %d ms\n", end - start);
    }

    if (cmd.hasOption("streamingkmeans")) {
      System.out.printf("\nStreaming k-means clusters models\n");
      System.out.printf("---------------------------------\n");
      long start = System.currentTimeMillis();
      trainComputed(inputIterable, outputBase, "streamingkmeans",
          clusterNamesToIds, clusterStreamingKMeans(getCentroidsFromSeqFileIterable(inputIterable)));
      long end = System.currentTimeMillis();
      System.out.printf("Trained models for streamingkmeans clusters. Took %d ms\n", end - start);
    }
  }
}
