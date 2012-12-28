package org.apache.mahout.knn.tools;

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

public class Seqfile3DProjector {
  public static void main(String[] args) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(args[0]), conf);
    Text key = new Text();
    VectorWritable value = new VectorWritable();

    PrintWriter writer = new PrintWriter(args[1]);
    boolean initialized = false;
    List<Vector> basisVectors = null;
    Map<String, Centroid> actualClusters = Maps.newHashMap();
    System.out.println("Projecting all vectors");
    // Loop through each entry in the sequence file and project it in 3D.
    while (reader.next(key, value)) {
      // If we don't have a basis yet, generate one (only once).
      if (!initialized) {
        basisVectors = ProjectionSearch.generateBasis(value.get().size(), 3);
        initialized = true;
      }
      // Project the vector.
      Vector vectorValue = value.get().clone();
      Vector projectedVector = new DenseVector(3);
      for (int i = 0; i < 3; ++i) {
        projectedVector.set(i, vectorValue.dot(basisVectors.get(i)));
      }
      projectedVector = projectedVector.normalize();
      writer.printf("%f %f %f\n", projectedVector.get(0), projectedVector.get(1),
          projectedVector.get(2));
      // Update the real center.
      String stringKey = ""; // EvaluateClustering.getNameBase(key.toString());
      Centroid centroid = actualClusters.remove(stringKey);
      if (centroid == null) {
        centroid = new Centroid(1, vectorValue, 1);
      } else {
        centroid.update(vectorValue);
      }
      actualClusters.put(stringKey, centroid);
    }
    writer.close();
    System.out.println("Projecting centroids");
    writer = new PrintWriter(args[1] + "-centers");
    // Loop through each actual cluster and project it.
    for (Map.Entry<String, Centroid> actualClusterEntry : actualClusters.entrySet()) {
      Vector vectorValue = actualClusterEntry.getValue();
      writer.printf("%f %f %f\n", vectorValue.dot(basisVectors.get(0)),
          vectorValue.dot(basisVectors.get(1)), vectorValue.dot(basisVectors.get(2)));
    }
    writer.close();
    reader.close();
  }
}
