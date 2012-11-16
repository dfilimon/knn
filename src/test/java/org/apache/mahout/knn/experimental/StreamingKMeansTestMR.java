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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.DataUtils;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.*;
import org.apache.hadoop.mrunit.mapreduce.MapDriver;
import org.apache.mahout.math.random.WeightedThing;
import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.Test;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class StreamingKMeansTestMR {
  private  MapDriver<IntWritable, CentroidWritable, IntWritable, CentroidWritable> mapDriver;

  @Before
  public void setUp() {
    StreamingKMeansMapper mapper = new StreamingKMeansMapper();
    mapDriver = MapDriver.newMapDriver(mapper);
    Configuration conf = new Configuration();
    conf.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, EuclideanDistanceMeasure.class.getName());
    conf.setInt(StreamingKMeansDriver.SEARCH_SIZE_OPTION, 5);
    conf.setInt(StreamingKMeansDriver.NUM_PROJECTIONS_OPTION, 2);
    conf.set(StreamingKMeansDriver.SEARCHER_CLASS_OPTION, ProjectionSearch.class.getName());
    conf.setInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 2);
    mapDriver.setConfiguration(conf);
  }

  @Test
  public void testHypercubeMapper() throws IOException {
    int numDimensions = 3;
    int numVertices = 1 << numDimensions;
    int numPoints = 1000;
    Pair<List<Centroid>, List<Centroid>> data =
        DataUtils.sampleMultiNormalHypercube(numDimensions, numPoints);
    StreamingKMeans clusterer = new StreamingKMeans(new ProjectionSearch(new
        EuclideanDistanceMeasure(), 4, 10), numVertices, DataUtils.estimateDistanceCutoff(data
        .getFirst()));
    for (Centroid datapoint : data.getFirst()) {
      clusterer.cluster(datapoint);
    }
    BruteSearch resultSearcher = new BruteSearch(new EuclideanDistanceMeasure());
    for (Vector centroid : clusterer.getCentroids()) {
      resultSearcher.add(centroid);
    }
    int i = 0;
    for (Vector mean : data.getSecond()) {
      WeightedThing<Vector> closest = resultSearcher.search(mean, 1).get(0);
      assertThat(closest.getWeight(), is(Matchers.lessThan(0.05)));
    }
  }
}
