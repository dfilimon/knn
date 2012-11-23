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

import java.io.IOException;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.LocalitySensitiveHashSearch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Classifies the vectors into different clusters found by the clustering
 * algorithm.
 */
public final class StreamingKMeansDriver extends AbstractJob {
  // TODO(dfilimon): These constants should move to DefaultOptionCreator and so should the code
  // that handles their creation.
  public static final String SEARCHER_CLASS_OPTION = "searcherClass";
  public static final String NUM_PROJECTIONS_OPTION = "numProjections";
  public static final String SEARCH_SIZE_OPTION = "searchSize";
  public static final String MAX_NUM_ITERATIONS = "maxNumIterations";

  private static final Logger log = LoggerFactory.getLogger(StreamingKMeansDriver.class);

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator
        .numClustersOption()
        .withDescription(
            "The k in k-Means. Approximately this many clusters will be generated.").create());
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption("searcherClass", "sc", "The type of searcher to be used when performing nearest " +
        "neighbor searches. Defaults to BruteSearch.", "org.apache.mahout.knn.search" +
        ".BruteSearch");
    addOption("numProjections", "np", "The number of projections considered in estimating the " +
        "distances between vectors. Only used when the distance measure requested is either " +
        "ProjectionSearch or FastProjectionSearch. If no value is given, defaults to 20.", "20");
    addOption("searchSize", "s", "In more efficient searches (non BruteSearch), " +
        "not all distances are calculated for determining the nearest neighbors. The number of " +
        "elements whose distances from the query vector is actually computer is proportional to " +
        "searchSize. If no value is given, defaults to 10.", "10");
    addOption("maxNumIterations", "i", "The maximum number of iterations to run for the " +
        "BallKMeans algorithm used by the reducer.", "10");

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    configureOptionsForWorkers();

    run(getConf(), input, output);
    return 0;
  }

  /**
   * Set up the Configuration object used by Mappers and Reducers to configure
   * themselves with the requested options. The options control:
   * <ul>
   *   <li>how many clusters to generate</li>
   *   <li>which distance measure to use</li>
   *   <li>which searcher class to use, and what parameters to instantiate it with</li>
   * </ul>
   */
  private void configureOptionsForWorkers() throws ClassNotFoundException, IllegalAccessException,
      InstantiationException {
    Configuration conf = getConf();
    log.info("Starting to configure options for workers");

    // The number of clusters to generate.
    String numClustersStr = getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION);
    Preconditions.checkNotNull(numClustersStr, "No number of clusters specified");
    int numClusters = Integer.parseInt(numClustersStr);

    // The distance measure class to use.
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = EuclideanDistanceMeasure.class.getName();
      log.info("No measure class given, using EuclideanDistanceMeasure");
    }

    // The searcher class to use. This should never be null because of the default value.
    String searcherClass = getOption(SEARCHER_CLASS_OPTION);
    Preconditions.checkNotNull(searcherClass, "No searcher class specified");

    // Get more parameters depending on the kind of search class we're working with. BruteSearch
    // doesn't need anything else.
    // LocalitySensitiveHashSearch and ProjectionSearches need searchSize.
    // ProjectionSearches also need the number of projections.
    boolean getSearchSize = false;
    boolean getNumProjections = false;
    if (!searcherClass.equals(BruteSearch.class.getName())) {
      getSearchSize = true;
      if (!searcherClass.equals(LocalitySensitiveHashSearch.class.getName())) {
        getNumProjections = true;
      }
    }

    // The search size to use. This is quite fuzzy and might end up not being configurable at all.
    int searchSize = 0;
    if (getSearchSize) {
      String searchSizeStr = getOption(SEARCH_SIZE_OPTION);
      Preconditions.checkNotNull(searchSize, "No searcher size given and the searcher class is " +
          searcherClass);
      searchSize = Integer.parseInt(searchSizeStr);
    }

    // The number of projections to use. This is only useful in projection searches which
    // project the vectors on multiple basis vectors to get distance estimates that are faster to
    // calculate.
    int numProjections = 0;
    if (getNumProjections) {
      String numProjectionsStr = getOption(NUM_PROJECTIONS_OPTION);
      Preconditions.checkNotNull(numProjections, "No number of projections given and the " +
          "searcher class is " + searcherClass);
      numProjections = Integer.parseInt(numProjectionsStr);
    }

    String maxNumIterationsStr = getOption(MAX_NUM_ITERATIONS);
    Preconditions.checkNotNull(maxNumIterationsStr, "No maximum number of iterations specified");
    int maxNumIterations = Integer.parseInt(maxNumIterationsStr);

    configureOptionsForWorkers(getConf(), numClusters, measureClass, searcherClass, searchSize,
        numProjections, maxNumIterations);
  }

  public static void configureOptionsForWorkers(Configuration conf, int numClusters,
                                                String measureClass, String searcherClass,
                                                int searchSize, int numProjections,
                                                int maxNumIterations) {
    conf.setInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, numClusters);
    try {
      Class.forName(measureClass);
    }  catch (ClassNotFoundException e) {
      log.error("Measure class not found " + measureClass, e);
    }
    conf.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, measureClass);
    try {
      Class.forName(searcherClass);
    } catch (ClassNotFoundException e) {
      log.error("Searcher class not found " + measureClass, e);
    }
    conf.set(SEARCHER_CLASS_OPTION, searcherClass);
    conf.setInt(SEARCH_SIZE_OPTION, searchSize);
    conf.setInt(NUM_PROJECTIONS_OPTION, numProjections);
    conf.setInt(MAX_NUM_ITERATIONS, maxNumIterations);
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the results of the final iteration to
   * cluster the input vectors.
   *
   * @param input
   *          the directory pathname for input points
   * @param output
   *          the directory pathname for output points
   */
  public static void run(Configuration conf, Path input, Path output)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("Starting StreamingKMeans clustering");

    // Prepare Job for submission.
    Job job = new Job(conf, "StreamingKMeans");

    // Input and output file format.
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    // Mapper output Key and Value classes.
    // We don't really need to output anything as a key, since there will only be 1 reducer.
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(CentroidWritable.class);

    // Reducer output Key and Value classes.
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(CentroidWritable.class);

    // Mapper and Reducer classes.
    job.setMapperClass(StreamingKMeansMapper.class);
    job.setReducerClass(StreamingKMeansReducer.class);

    // There is only one reducer so that the intermediate centroids get collected on one
    // machine and are clustered in memory to get the right number of clusters.
    job.setNumReduceTasks(1);

    // Set input and output paths for the job.
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    // Set the JAR (so that the required libraries are available) and run.
    job.setJarByClass(StreamingKMeansDriver.class);
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("StreamingKMeans interrupted");
    }

    log.info("StreamignKMeans job complete");
  }

  /**
   * Constructor to be used by the ToolRunner.
   */
  private StreamingKMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new StreamingKMeansDriver(), args);
  }
}
