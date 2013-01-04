package org.apache.mahout.knn.tools;

import com.google.common.collect.Lists;
import org.apache.mahout.common.Pair;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.Normal;
import org.apache.mahout.math.stats.OnlineSummarizer;
import org.junit.Test;

import java.util.List;

public class OldProjectionBenchmark {
  private static final int VECTOR_SIZE = 1024;
  private static final int NUM_VECTORS = 2048;
  private static final int PROJECTED_VECTOR_SIZE = 256;
  private static final int NUM_TESTS = 10;

  @Test
  public void testProjectionAveraged() {
    OnlineSummarizer summarizer = new OnlineSummarizer();
    for (int i = 0; i < NUM_TESTS; ++i) {
      Pair<Double, Double> result = testProjection();
      System.out.printf("%d: Matrix multiplication projection time %f\n", i, result.getFirst());
      System.out.printf("%d: Component multiplication projection time %f\n", i,
          result.getSecond());
      summarizer.add(Math.abs(result.getFirst() - result.getSecond()));
    }
    System.out.printf("Mean absolute difference %f with deviation %f", summarizer.getMean(),
        summarizer.getSD());
  }

  /**
   * We're reducing the size down from VECTOR_SIZE to PROJECTED_VECTOR_SIZE by multiplying by a
   * matrix and by multiplying manually by a List<Vector> to compare the performance.
   */
  public static Pair<Double, Double> testProjection() {
    System.out.printf("Projecting %d vectors of size %d to size %d\n", NUM_VECTORS, VECTOR_SIZE,
        PROJECTED_VECTOR_SIZE);

    // This is the list of vectors we'll be projecting.
    Matrix testVectors = new DenseMatrix(NUM_VECTORS,  VECTOR_SIZE);
    testVectors.assign(new Normal());

    // To project down to PROJECTED_VECTOR_SIZE, we need PROJECTED_VECTOR_SIZE basis vectors of
    // size VECTOR_SIZE. The dot product of each test vector with each column of a projection
    // matrix gives the projected vectors.
    Matrix projectionMatrix = new DenseMatrix(VECTOR_SIZE, PROJECTED_VECTOR_SIZE);
    projectionMatrix.assign(new Normal());

    double start = System.currentTimeMillis();
    Matrix projectedVectorMatrix = testVectors.times(projectionMatrix);
    double end = System.currentTimeMillis();
    double firstTime = end - start;
    System.out.printf("Matrix projection done\n");

    List<Vector> projectionVectors = ProjectionSearch.generateVectorBasis(PROJECTED_VECTOR_SIZE,
        VECTOR_SIZE);
    List<Vector> projectedVectorList = Lists.newArrayListWithExpectedSize(NUM_VECTORS);
    start = System.currentTimeMillis();
    for (MatrixSlice ms : testVectors) {
      Vector v = ms.vector();
      Vector pv = new DenseVector(PROJECTED_VECTOR_SIZE);
      for (int i = 0; i < PROJECTED_VECTOR_SIZE; ++i) {
        pv.setQuick(i, v.dot(projectionVectors.get(i)));
      }
      projectedVectorList.add(pv);
    }
    end = System.currentTimeMillis();
    double secondTime = end - start;
    System.out.printf("Vector list projection done\n");
    return new Pair<Double, Double>(firstTime, secondTime);
  }
}
