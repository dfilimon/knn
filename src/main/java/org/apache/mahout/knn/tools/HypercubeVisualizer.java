package org.apache.mahout.knn.tools;

import com.google.common.collect.Lists;
import org.apache.mahout.common.Pair;
import org.apache.mahout.knn.cluster.DataUtils;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.List;

public class HypercubeVisualizer {
  public static void main(String[] args) throws IOException {
    int numDimensions = Integer.parseInt(args[0]);
    int numSamples = Integer.parseInt(args[1]);

    Pair<List<Centroid>, List<Centroid>> dataset =
        DataUtils.sampleMultiNormalHypercube(numDimensions,  numSamples);

    List<Centroid> datapoints = dataset.getFirst();
    List<Centroid> medians = dataset.getSecond();

    if (numDimensions > 3) {
      Matrix basisMatrix = ProjectionSearch.generateBasis(3, numDimensions);
      List<Centroid> newDatapoins = Lists.newArrayList();
      for (int i = 0; i < datapoints.size(); ++i) {
        Centroid c = datapoints.get(i);
        Vector v = basisMatrix.times(c);
        newDatapoins.add(new Centroid(i, v, 1));
      }
      datapoints = newDatapoins;
      List<Centroid> newMedians = Lists.newArrayList();
      for (int i = 0; i < medians.size(); ++i) {
        Centroid c = medians.get(i);
        Vector v = basisMatrix.times(c);
        newMedians.add(new Centroid(i, v, 1));
      }
      medians = newMedians;
      numDimensions = 3;
    }
    System.out.printf("%d %d\n", datapoints.size(), medians.size());
    new File(args[2]).delete();
    PrintWriter writer = new PrintWriter(new FileWriter(args[2]));
    writer.printf("# Number of samples; Number of centroids; List of samples; List of " +
        "centroids\n");
    writer.printf("%d %d\n", numDimensions, numSamples);
    for (Centroid c : datapoints) {
      for (int i = 0; i < c.size(); ++i)
        writer.printf("%f ", c.get(i));
      writer.println();
    }
    for (Centroid c : medians) {
      for (int i = 0; i < c.size(); ++i)
        writer.printf("%f ", c.get(i));
      writer.println();
    }
    writer.close();
  }
}
