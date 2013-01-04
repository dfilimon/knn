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

package org.apache.mahout.knn.cluster;


import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.search.*;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.runners.Parameterized.Parameters;


@RunWith(value = Parameterized.class)
public class StreamingKMeansTest {
  private static final int NUM_DATA_POINTS = 2 * 32768;
  private static final int NUM_DIMENSIONS = 6;
  private static final int NUM_PROJECTIONS = 2;
  private static final int SEARCH_SIZE = 10;

  private static final Pair<List<Centroid>, List<Centroid>> syntheticData =
      DataUtils.sampleMultiNormalHypercube(NUM_DIMENSIONS, NUM_DATA_POINTS);

  private UpdatableSearcher searcher;
  private boolean allAtOnce;

  public StreamingKMeansTest(UpdatableSearcher searcher, boolean allAtOnce) {
    this.searcher = searcher;
    this.allAtOnce = allAtOnce;
    LoggerContext loggerContext = (LoggerContext)LoggerFactory.getILoggerFactory();
    loggerContext.getLogger(StreamingKMeans.class).setLevel(Level.INFO);
  }

  @Parameters
  public static List<Object[]> generateData() {
    return Arrays.asList(new Object[][] {
        {new ProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE), true},
        {new FastProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE),
            true},
        {new LocalitySensitiveHashSearch(new EuclideanDistanceMeasure(), SEARCH_SIZE), true},
        {new ProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE),
        false},
        {new FastProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE),
            false},
        {new LocalitySensitiveHashSearch(new EuclideanDistanceMeasure(), SEARCH_SIZE), false}
    }
    );
  }

  @Test
  public void testAverageDistanceCutoff() {
    double avgDistanceCutoff = 0;
    double avgNumClusters = 0;
    int numTests = 8;
    System.out.printf("Distance cutoff for %s\n", searcher.getClass().getName());
    for (int i = 0; i < numTests; ++i) {
      searcher.clear();
      Pair<List<Centroid>, List<Centroid>> syntheticData = DataUtils.sampleMultiNormalHypercube
          (NUM_DIMENSIONS, NUM_DATA_POINTS);
      int numStreamingClusters = (int)Math.log(syntheticData.getFirst().size()) * (1 <<
          NUM_DIMENSIONS);
      StreamingKMeans clusterer =
          new StreamingKMeans(searcher, numStreamingClusters, DataUtils.estimateDistanceCutoff
              (syntheticData.getFirst()));
      clusterer.cluster(syntheticData.getFirst());
      avgDistanceCutoff += clusterer.getDistanceCutoff();
      avgNumClusters += clusterer.getEstimatedNumClusters();
      System.out.printf("%d %f\n", i, clusterer.getDistanceCutoff());
    }
    avgDistanceCutoff /= numTests;
    avgNumClusters /= numTests;
    System.out.printf("Final: distanceCutoff: %f estNumClusters: %f\n", avgDistanceCutoff, avgNumClusters);
  }

  @Test
  public void testClustering() {
    searcher.clear();
    System.out.printf("k log n = %d\n", (int)Math.log(syntheticData.getFirst().size()) * (1 <<
        NUM_DIMENSIONS));
    StreamingKMeans clusterer =
        new StreamingKMeans(searcher,
            (int)Math.log(syntheticData.getFirst().size()) * (1 << NUM_DIMENSIONS),
            DataUtils.estimateDistanceCutoff(syntheticData.getFirst()));
    long startTime = System.currentTimeMillis();
    if (allAtOnce) {
      clusterer.cluster(syntheticData.getFirst());
    } else {
      for (Centroid datapoint : syntheticData.getFirst()) {
        clusterer.cluster(datapoint);
      }
    }
    long endTime = System.currentTimeMillis();

    System.out.printf("%s %s\n", searcher.getClass().getName(), searcher.getDistanceMeasure()
        .getClass().getName());
    System.out.printf("Total number of clusters %d\n", clusterer.getCentroids().size());

    System.out.printf("Weights: %f %f\n", totalWeight(syntheticData.getFirst()),
        totalWeight(clusterer.getCentroids()));
    assertEquals("Total weight not preserved", totalWeight(syntheticData.getFirst()),
        totalWeight(clusterer.getCentroids()), 1e-9);

    // and verify that each corner of the cube has a centroid very nearby
    double maxWeight = 0;
    for (Vector mean : syntheticData.getSecond()) {
      WeightedThing<Vector> v = searcher.search(mean, 1).get(0);
      maxWeight = Math.max(v.getWeight(), maxWeight);
    }
    assertTrue("Maximum weight too large " + maxWeight, maxWeight < 0.05);
    double clusterTime = (endTime - startTime) / 1000.0;
    System.out.printf("%s\n%.2f for clustering\n%.1f us per row\n\n",
        searcher.getClass().getName(), clusterTime,
        clusterTime / syntheticData.getFirst().size() * 1e6);

    // verify that the total weight of the centroids near each corner is correct
    double[] cornerWeights = new double[1 << NUM_DIMENSIONS];
    Searcher trueFinder = new BruteSearch(new EuclideanDistanceMeasure());
    for (Vector trueCluster : syntheticData.getSecond()) {
      trueFinder.add(trueCluster);
    }
    for (Centroid centroid : clusterer) {
      WeightedThing<Vector> closest = trueFinder.search(centroid, 1).get(0);
      cornerWeights[((Centroid)closest.getValue()).getIndex()] += centroid.getWeight();
    }
    int expectedNumPoints = NUM_DATA_POINTS / (1 << NUM_DIMENSIONS);
    for (double v : cornerWeights) {
      System.out.printf("%f ", v);
    }
    System.out.println();
    for (double v : cornerWeights) {
      assertEquals(expectedNumPoints, v, 0);
    }
  }

  private double totalWeight(Iterable<? extends Vector> data) {
    double sum = 0;
    for (Vector row : data) {
      if (row instanceof WeightedVector) {
        sum += ((WeightedVector)row).getWeight();
      } else {
        sum++;
      }
    }
    return sum;
  }
}
