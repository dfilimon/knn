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

package org.apache.mahout.knn.tools;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.*;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.collect.*;
import com.google.common.io.Files;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.knn.experimental.CentroidWritable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TestNewsGroupsKMeanLogisticRegression {

  private String inputFile;
  private String modelFile;
  private String centroidsFile;
  private String labelFile;

  private TestNewsGroupsKMeanLogisticRegression() {
  }

  public static void main(String[] args) throws IOException {
    TestNewsGroupsKMeanLogisticRegression runner = new TestNewsGroupsKMeanLogisticRegression();
    if (runner.parseArgs(args)) {
      runner.run(new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
    }
  }

  public void run(PrintWriter output) throws IOException {

    // Contains the best model.
    OnlineLogisticRegression classifier =
        ModelSerializer.readBinary(new FileInputStream(modelFile), OnlineLogisticRegression.class);

    // Get the cluster labels.
    List<String> lines = Files.readLines(new File(labelFile), Charset.defaultCharset());
    Map<String, Integer> labels = Maps.newHashMap();
    for (String line : lines.subList(1, lines.size())) {
      String[] chunks = line.split(", ");
      Preconditions.checkArgument(chunks.length == 2, "Invalid labels line " + chunks.toString());
      labels.put(chunks[0], Integer.parseInt(chunks[1]));
      System.out.printf("%s: %s\n", chunks[0], chunks[1]);
    }
    List<String> reverseLabels = new ArrayList(Collections.nCopies(labels.size(), ""));
    for (Map.Entry<String, Integer> pair : labels.entrySet()) {
      reverseLabels.set(pair.getValue(), pair.getKey());
    }

    Configuration conf = new Configuration();
    // Get the centroids used for computing the distances for this model.
    SequenceFileDirValueIterable<CentroidWritable> centroidIterable = new
        SequenceFileDirValueIterable<CentroidWritable>(new Path(centroidsFile), PathType.LIST, conf);
    List<Centroid> centroids =
        Lists.newArrayList(
            CreateCentroids.getCentroidsFromCentroidWritableIterable(centroidIterable));
    // Get the encoded documents (the vectors from tf-idf).
    SequenceFileDirIterable<Text, VectorWritable> inputIterable = new
        SequenceFileDirIterable<Text, VectorWritable>(new Path(inputFile), PathType.LIST, conf);

    ResultAnalyzer ra = new ResultAnalyzer(labels.keySet(), "DEFAULT");
    for (Pair<Text, VectorWritable> pair : inputIterable) {
      int actual = labels.get(pair.getFirst().toString());
      Vector encodedInput = distancesFromCentroidsVector(pair.getSecond().get(), centroids);
      Vector result = classifier.classifyFull(encodedInput);
      int cat = result.maxValueIndex();
      double score = result.maxValue();
      double ll = classifier.logLikelihood(actual, encodedInput);
      ClassifierResult cr = new ClassifierResult(reverseLabels.get(cat), score, ll);
      ra.addInstance(pair.getFirst().toString(), cr);
    }
    output.println(ra);
  }

  private Vector distancesFromCentroidsVector(Vector input, List<Centroid> centroids) {
    Vector encodedInput = new DenseVector(centroids.size());
    for (int i = 0; i < centroids.size(); ++i) {
      encodedInput.setQuick(i, input.getDistanceSquared(centroids.get(i)));
    }
    return encodedInput;
  }

  boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder.withLongName("input")
        .withShortName("i")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
        .withDescription("where to get test data (encoded with tf-idf)")
        .create();

    Option modelFileOption = builder.withLongName("model")
        .withShortName("m")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("model").withMaximum(1).create())
        .withDescription("where to get a model")
        .create();

    Option centroidsFileOption = builder.withLongName("centroids")
        .withShortName("c")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("centroids").withMaximum(1).create())
        .withDescription("where to get the centroids seqfile")
        .create();

    Option labelFileOption = builder.withLongName("labels")
        .withShortName("l")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("labels").withMaximum(1).create())
        .withDescription("CSV file containing the cluster labels")
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(modelFileOption)
        .withOption(centroidsFileOption)
        .withOption(labelFileOption)
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
    modelFile = (String) cmdLine.getValue(modelFileOption);
    centroidsFile = (String) cmdLine.getValue(centroidsFileOption);
    labelFile = (String) cmdLine.getValue(labelFileOption);
    return true;
  }

}
