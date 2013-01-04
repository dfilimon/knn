package org.apache.mahout.knn.tools;

import com.google.common.collect.Lists;
import org.apache.mahout.common.Pair;
import org.apache.mahout.knn.search.ProjectionSearch;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.random.Normal;
import org.apache.mahout.math.stats.OnlineSummarizer;
import org.junit.Test;

import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.OrderingComparisons.lessThan;

public class ProjectionBenchmark {
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
      summarizer.add(result.getFirst() - result.getSecond());
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
    Matrix testVectors = new DenseMatrix(VECTOR_SIZE, NUM_VECTORS);
    testVectors.assign(new Normal());

    // To project down to PROJECTED_VECTOR_SIZE, we need PROJECTED_VECTOR_SIZE basis vectors of
    // size VECTOR_SIZE. The dot product of each test vector with each column of a projection
    // matrix gives the projected vectors.
    Matrix projectionMatrix = ProjectionSearch.generateBasis(PROJECTED_VECTOR_SIZE, VECTOR_SIZE);

    double start = System.currentTimeMillis();
    Matrix projectedVectorMatrix = projectionMatrix.times(testVectors);
    double end = System.currentTimeMillis();
    double firstTime = end - start;
    System.out.printf("Matrix projection done\n");

    // Transpose the test vectors to iterate through the rows.
    testVectors = testVectors.transpose();

    List<Vector> projectionVectors = Lists.newArrayListWithExpectedSize(PROJECTED_VECTOR_SIZE);
    // generateVectorBasis(PROJECTED_VECTOR_SIZE, VECTOR_SIZE);
    for (Vector v : projectionMatrix) {
      projectionVectors.add(v);
    }

    List<Vector> projectedVectorList = Lists.newArrayListWithExpectedSize(NUM_VECTORS);
    start = System.currentTimeMillis();
    for (Vector v : testVectors) {
      Vector pv = new DenseVector(PROJECTED_VECTOR_SIZE);
      for (int i = 0; i < PROJECTED_VECTOR_SIZE; ++i) {
        pv.setQuick(i, projectionVectors.get(i).dot(v));
      }
      projectedVectorList.add(pv);
    }
    end = System.currentTimeMillis();
    double secondTime = end - start;
    System.out.printf("Vector list projection done\n");

    for (int i = 0; i < NUM_VECTORS; ++i) {
      assertThat(projectedVectorList.get(i).minus(projectedVectorMatrix.viewColumn(i))
          .getLengthSquared(), lessThan(0.5));
    }
    System.out.printf("\n");
    return new Pair<Double, Double>(firstTime, secondTime);
  }
}
