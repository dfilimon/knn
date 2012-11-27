/**
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

package org.apache.mahout.knn.experimental;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mrunit.mapreduce.ReduceDriver;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.DataUtils;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.*;
import org.apache.mahout.math.*;
import org.apache.hadoop.mrunit.mapreduce.MapDriver;
import org.apache.mahout.math.random.WeightedThing;
import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.Test;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.util.List;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

import java.util.Arrays;

@RunWith(value = Parameterized.class)
public class StreamingKMeansTestMR {

  private static final int NUM_DATA_POINTS = 100000;
  private static final int NUM_DIMENSIONS = 3;
  private static final int NUM_PROJECTIONS = 4;
  private static final int SEARCH_SIZE = 10;
  private static final int MAX_NUM_ITERATIONS = 10;

  private static Pair<List<Centroid>, List<Centroid>> syntheticData =
      DataUtils.sampleMultiNormalHypercube(NUM_DIMENSIONS, NUM_DATA_POINTS);

  private Configuration configuration;

  public StreamingKMeansTestMR(String searcherClassName, String distanceMeasureClassName) {
    configuration = new Configuration();
    configuration.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, distanceMeasureClassName);
    configuration.setInt(StreamingKMeansDriver.SEARCH_SIZE_OPTION, SEARCH_SIZE);
    configuration.setInt(StreamingKMeansDriver.NUM_PROJECTIONS_OPTION, NUM_PROJECTIONS);
    configuration.set(StreamingKMeansDriver.SEARCHER_CLASS_OPTION, searcherClassName);
    configuration.setInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 1 << NUM_DIMENSIONS);
    configuration.setInt(StreamingKMeansDriver.MAX_NUM_ITERATIONS, MAX_NUM_ITERATIONS);
  }

  @Parameterized.Parameters
  public static List<Object[]> generateData() {
    return Arrays.asList(new Object[][]{
        {ProjectionSearch.class.getName(), EuclideanDistanceMeasure.class.getName()},
        {FastProjectionSearch.class.getName(), EuclideanDistanceMeasure.class.getName()},
        {LocalitySensitiveHashSearch.class.getName(), EuclideanDistanceMeasure.class.getName()}
    });
  }

  @Before
  public void setUp() {
  }

  @Test
  public void testHypercubeMapper() throws IOException {
    MapDriver<IntWritable, CentroidWritable, IntWritable, CentroidWritable> mapDriver =
        MapDriver.newMapDriver(new StreamingKMeansMapper());
    mapDriver.setConfiguration(configuration);
    for (Centroid datapoint : syntheticData.getFirst()) {
      mapDriver.addInput(new IntWritable(0), new CentroidWritable(datapoint));
    }
    List<org.apache.hadoop.mrunit.types.Pair<IntWritable,CentroidWritable>> results = mapDriver.run();
    BruteSearch resultSearcher = new BruteSearch(new EuclideanDistanceMeasure());
    for (org.apache.hadoop.mrunit.types.Pair<IntWritable, CentroidWritable> result : results) {
      resultSearcher.add(result.getSecond().getCentroid());
    }
    for (Vector mean : syntheticData.getSecond()) {
      WeightedThing<Vector> closest = resultSearcher.search(mean, 1).get(0);
      assertThat(closest.getWeight(), is(Matchers.lessThan(0.05)));
    }
  }

  @Test
  public void testHypercubeReducer() throws IOException {
    StreamingKMeans clusterer = new StreamingKMeans(StreamingKMeansMapper
        .searcherFromConfiguration(configuration), 1 << NUM_DIMENSIONS,
        DataUtils.estimateDistanceCutoff(syntheticData.getFirst()));
    clusterer.cluster(syntheticData.getFirst());
    ReduceDriver<IntWritable, CentroidWritable, IntWritable, CentroidWritable> reduceDriver =
        ReduceDriver.newReduceDriver(new StreamingKMeansReducer());
    reduceDriver.setConfiguration(configuration);
    List<CentroidWritable> reducerInputs = Lists.newArrayList();
    int postMapperTotalWeight = 0;
    for (Centroid intermediateCentroid : clusterer.getCentroidsIterable()) {
      reducerInputs.add(new CentroidWritable(intermediateCentroid));
      postMapperTotalWeight += intermediateCentroid.getWeight();
    }
    reduceDriver.addInput(new IntWritable(0), reducerInputs);
    List<org.apache.hadoop.mrunit.types.Pair<IntWritable,CentroidWritable>> results =
        reduceDriver.run();
    int numClusters = 0;
    double expectedWeight = postMapperTotalWeight / (1 << NUM_DIMENSIONS);
    for (org.apache.hadoop.mrunit.types.Pair<IntWritable, CentroidWritable> result : results) {
      assertEquals("Final centroid index is invalid", numClusters, result.getFirst().get());
      assertEquals("Unbalanced weight for centroid", expectedWeight,
          result.getSecond().getCentroid().getWeight(), 0);
      ++numClusters;
    }
    assertEquals("Invalid number of clusters", 1 << NUM_DIMENSIONS, numClusters);
  }
}
