package org.apache.mahout.knn.experimental;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.knn.cluster.BallKMeans;
import org.apache.mahout.knn.cluster.DataUtils;
import org.apache.mahout.knn.cluster.StreamingKMeans;
import org.apache.mahout.knn.search.BruteSearch;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.List;

public class EvaluateClustering {
  public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    Path inFile = new Path(args[0]);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, inFile, conf);
    Text key = new Text();
    VectorWritable value = new VectorWritable();
    reader.next(key, value);
    int numVectors = 0;
    int vectorDimension = value.get().size();

    int reducedDimension = 50;
    List<Vector> basisVectors =  ProjectionSearch.generateBasis(vectorDimension, reducedDimension);
    List<Centroid> reducedVectors = Lists.newArrayList();
    do {
      Vector v = value.get();
      Vector reducedVector = new DenseVector(reducedDimension);
      for (int i = 0; i < reducedDimension; ++i) {
        reducedVector.setQuick(i, basisVectors.get(i).dot(value.get()));
      }
      reducedVectors.add(new Centroid(numVectors, reducedVector, 1));
      /* System.out.printf("%s => %s [%d] {%s}\n", key.toString(), v.getClass(), v.size(),
          reducedVector.toString()); */
      ++numVectors;
    } while (reader.next(key, value));

    System.out.println("Total number of vectors " + numVectors);

    StreamingKMeans clusterer = new StreamingKMeans(new ProjectionSearch(new
        EuclideanDistanceMeasure(), 10, 10), 20, DataUtils.estimateDistanceCutoff(reducedVectors));
    clusterer.cluster(reducedVectors);
    float totalWeight = 0;
    for (Centroid c : clusterer.getCentroidsIterable()) {
      System.out.printf("%d: %s : %f\n", c.getIndex(), c.getVector(), c.getWeight());
      totalWeight += c.getWeight();
    }
    System.out.printf("Detected %d centroids; total weight %f\n", clusterer.getCentroids().size()
        , totalWeight);

    BallKMeans finalClusterer = new BallKMeans(new BruteSearch(new EuclideanDistanceMeasure()),
        20, 10);
    List<Centroid> intermediateClusters = Lists.newArrayList(clusterer.getCentroidsIterable());
    finalClusterer.cluster(intermediateClusters);
    totalWeight = 0;
    for (Centroid c : finalClusterer) {
      System.out.printf("%d: %s : %f\n", c.getIndex(), c.getVector(), c.getWeight());
      totalWeight += c.getWeight();
    }
    System.out.printf("Detected %d centroids; total weight %f\n", clusterer.getCentroids().size()
        , totalWeight);
  }
}
