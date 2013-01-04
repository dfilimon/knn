package org.apache.mahout.knn.tools;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import com.google.common.base.Charsets;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.knn.cluster.BallKMeans;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.experimental.CentroidWritable;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

public class CreateCentroids {
  private String inputFile;
  private String outputFileBase;
  private boolean computeActualCentroids;
  private boolean computeBallKMeansCentroids;
  private boolean computeStreamingKMeansCentroids;
  private Integer numClusters;

  public static void main(String[] args) throws IOException {
    CreateCentroids runner = new CreateCentroids();
    if (runner.parseArgs(args)) {
      runner.run(new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
    }
  }

  // TODO(dfilimon): Make more configurable.
  public static Pair<Integer, Iterable<Centroid>> clusterStreamingKMeans(
      Iterable<Centroid> dataPoints, int numClusters) {
    ((LoggerContext) LoggerFactory.getILoggerFactory()).getLogger(StreamingKMeans.class).setLevel(Level.INFO);
    StreamingKMeans clusterer = new StreamingKMeans(new ProjectionSearch(new
        EuclideanDistanceMeasure(), 20, 10), numClusters, 10e-6);
    clusterer.cluster(dataPoints);
    return new Pair<Integer, Iterable<Centroid>>(clusterer.getCentroids().size(), clusterer);
  }

  // TODO(dfilimon): Make more configurable.
  public static Pair<Integer, Iterable<Centroid>> clusterBallKMeans(
      List<Centroid> dataPoints, int numClusters) {
    BallKMeans clusterer = new BallKMeans(new BruteSearch(new EuclideanDistanceMeasure()),
        numClusters, 20);
    clusterer.cluster(dataPoints);
    return new Pair<Integer, Iterable<Centroid>>(numClusters, clusterer);
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

  public static void writeCentroidsToSequenceFile(Configuration conf, String name,
                                                  Iterable<Centroid> centroids) throws IOException {
    SequenceFile.Writer writer = SequenceFile.createWriter(FileSystem.get(conf), conf,
        new Path(name), IntWritable.class, CentroidWritable.class);
    int i = 0;
    for (Centroid centroid : centroids) {
      writer.append(new IntWritable(i++), new CentroidWritable(centroid));
    }
    writer.close();
  }

  public static Iterable<Centroid> getCentroidsFromPairIterable(
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

  public static Iterable<Centroid> getCentroidsFromCentroidWritableIterable(
      Iterable<CentroidWritable>  dirIterable) {
    return Iterables.transform(dirIterable, new Function<CentroidWritable, Centroid>() {
      @Override
      public Centroid apply(CentroidWritable input) {
        Preconditions.checkNotNull(input);
        return input.getCentroid().clone();
      }
    });
  }

  private void run(PrintWriter printWriter) throws IOException {
    Configuration conf = new Configuration();
    SequenceFileDirIterable<Text, VectorWritable> inputIterable = new
        SequenceFileDirIterable<Text, VectorWritable>(new Path(inputFile), PathType.LIST, conf);

    if (computeActualCentroids) {
      Map<String, Centroid> actualClusters = Maps.newHashMap();
      printWriter.printf("Computing actual clusters\n");
      computeActualClusters(inputIterable, actualClusters);
      String outputFile = outputFileBase + "-actual.seqfile";
      printWriter.printf("Writing actual clusters to %s\n", outputFile);
      writeCentroidsToSequenceFile(conf, outputFile,  actualClusters.values());
    }

    if (computeBallKMeansCentroids || computeStreamingKMeansCentroids) {
      List<Centroid> centroids =
          Lists.newArrayList(getCentroidsFromPairIterable(inputIterable));
      Pair<Integer, Iterable<Centroid>> computedClusterPair;
      String suffix;
      printWriter.printf("Computing clusters for %d points\n", centroids.size());
      if (computeBallKMeansCentroids) {
        computedClusterPair =  clusterBallKMeans(centroids, numClusters);
        suffix = "-ballkmeans";
      } else {
        computedClusterPair =  clusterStreamingKMeans(centroids, numClusters);
        suffix = "-streamingkmeans";
      }
      String outputFile = outputFileBase + suffix + ".seqfile";
      printWriter.printf("Writing %s computed clusters to %s\n", suffix, outputFile);
      writeCentroidsToSequenceFile(conf, outputFile,  computedClusterPair.getSecond());
    }
  }

  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder.withLongName("input")
        .withShortName("i")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
        .withDescription("where to get test data (encoded with tf-idf)")
        .create();

    Option actualCentroidsOption = builder.withLongName("actual")
        .withShortName("a")
        .withDescription("if set, writes the actual cluster centroids to <output_base>-actual")
        .create();

    Option ballKMeansCentroidsOption = builder.withLongName("ballkmeans")
        .withShortName("bkm")
        .withDescription("if set, writes the ball k-means cluster centroids to " +
            "<output_base>-ballkmeans")
        .create();

    Option streamingKMeansCentroidsOption = builder.withLongName("streamingkmeans")
        .withShortName("skm")
        .withDescription("if set, writes the ball k-means cluster centroids to " +
            "<output_base>-streamingkmeans; note that the number of clusters for streaming " +
            "k-means is the estimated number of clusters and that no ball k-means step is " +
            "performed")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("the base of the centroids sequence file; will be appended with " +
            "-<algorithm> where algorithm is the method used to compute the centroids")
        .create();

    Option numClustersOption = builder.withLongName("numClusters")
        .withShortName("k")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("numClusters").withMaximum(1).create())
        .withDescription("the number of clusters to cluster the vectors in")
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(outputFileOption)
        .withOption(actualCentroidsOption)
        .withOption(ballKMeansCentroidsOption)
        .withOption(streamingKMeansCentroidsOption)
        .withOption(numClustersOption)
        .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine = parser.parseAndHelp(args);

    if (cmdLine == null) {
      return false;
    }

    inputFile = (String) cmdLine.getValue(inputFileOption);
    outputFileBase = (String) cmdLine.getValue(outputFileOption);
    computeActualCentroids = cmdLine.hasOption(actualCentroidsOption);
    computeBallKMeansCentroids = cmdLine.hasOption(ballKMeansCentroidsOption);
    computeStreamingKMeansCentroids = cmdLine.hasOption(streamingKMeansCentroidsOption);
    numClusters = Integer.parseInt((String) cmdLine.getValue(numClustersOption));
    return true;
  }
}
