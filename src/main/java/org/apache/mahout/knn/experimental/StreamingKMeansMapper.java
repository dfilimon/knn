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

package org.apache.mahout.knn.experimental;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.*;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class StreamingKMeansMapper extends Mapper<Writable, VectorWritable,
    IntWritable, CentroidWritable> {

  /**
   * The clusterer object used to cluster the points received by this mapper online.
   */
  private StreamingKMeans clusterer;
  private int numPoints = 0;

  private static final Logger log = LoggerFactory.getLogger(Mapper.class);

  @Override
  public void setup(Context context) {
    // At this point the configuration received from the Driver is assumed to be valid.
    // No other checks are made.
    Configuration conf = context.getConfiguration();
    UpdatableSearcher searcher;
    searcher = searcherFromConfiguration(conf);
    int numClusters = conf.getInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 1);
    clusterer = new StreamingKMeans(searcher, numClusters,
        conf.getFloat(StreamingKMeansDriver.ESTIMATED_DISTANCE_CUTOFF, (float) 10e-6));
  }

  @Override
  public void map(Writable key, VectorWritable point, Context context) {
    clusterer.cluster(new Centroid(numPoints++, point.get().clone(), 1));
  }

  @Override
  public void cleanup(Context context) throws IOException, InterruptedException {
    // All outputs have the same key to go to the same final reducer.
    for (Centroid centroid : clusterer) {
      context.write(new IntWritable(0), new CentroidWritable(centroid));
    }
  }

  @SuppressWarnings("ConstantConditions")
  public static UpdatableSearcher searcherFromConfiguration(Configuration conf) {
    DistanceMeasure distanceMeasure;
    String distanceMeasureClass = conf.get(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    try {
      distanceMeasure = (DistanceMeasure)Class.forName(distanceMeasureClass).newInstance();
    } catch (Exception e) {
      log.error("Failed to instantiate distanceMeasure", e);
      throw new RuntimeException("Failed to instantiate distanceMeasure", e);
    }

    int numProjections =  conf.getInt(StreamingKMeansDriver.NUM_PROJECTIONS_OPTION, 20);
    int searchSize =  conf.getInt(StreamingKMeansDriver.SEARCH_SIZE_OPTION, 10);

    UpdatableSearcher searcher;
    String searcherClass = conf.get(StreamingKMeansDriver.SEARCHER_CLASS_OPTION);
    try {
      if (searcherClass.equals(BruteSearch.class.getName())) {
        searcher = (UpdatableSearcher)Class.forName(searcherClass).getConstructor(DistanceMeasure
            .class).newInstance(distanceMeasure);
      } else if (searcherClass.equals(FastProjectionSearch.class.getName()) ||
          searcherClass.equals(ProjectionSearch.class.getName())) {
        searcher = (UpdatableSearcher)Class.forName(searcherClass).getConstructor(DistanceMeasure
            .class, int.class, int.class).newInstance(distanceMeasure, numProjections, searchSize);
      } else if (searcherClass.equals(LocalitySensitiveHashSearch.class.getName())) {
        searcher = (UpdatableSearcher)Class.forName(searcherClass).getConstructor(DistanceMeasure
            .class, int.class).newInstance(distanceMeasure, searchSize);
      } else {
        log.error("Unknown searcher class instantiation requested {}", searcherClass);
        throw new InstantiationException();
      }
    } catch (Exception e) {
      log.error("Failed to instantiate searcher", e);
      throw new RuntimeException("Failed to instantiate searcher.", e);
    }
    return searcher;
  }

}
