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

package org.apache.mahout.knn.search;

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.random.WeightedThing;

import java.util.*;

/**
 * Does approximate nearest neighbor dudes search by projecting the data.
 */
public class ProjectionSearch extends UpdatableSearcher implements Iterable<Vector> {

  /**
   * A lists of tree sets containing the scalar projections of each vector.
   * The elements in a TreeSet are WeightedThing<Integer>, where the weight is the scalar
   * projection of the vector at the index pointed to by the Integer from the referenceVectors list
   * on the basis vector whose index is the same as the index of the TreeSet in the List.
   */
  private List<TreeSet<WeightedThing<Vector>>> scalarProjections;

  /**
   * The list of random normalized projection vectors forming a basis.
   * The TreeSet of scalar projections at index i in scalarProjections corresponds to the vector
   * at index i from basisVectors.
   */
  private Matrix basisMatrix;

  /**
   * The number of elements to consider on both sides in the ball around the vector found by the
   * search in a TreeSet from scalarProjections.
   */
  private int searchSize;

  private int numProjections;
  private boolean initialized = false;

  /**
   * Generates a basis matrix of size projectedVectorSize x vectorSize. Multiplying a a vector by
   * this matrix results in the projected vector.
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a projection matrix
   */
  public static Matrix generateBasis(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    basisMatrix.assign(Functions.random());
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  /**
   * Generates a list of projectedVectorSize vectors, each of size vectorSize. This looks like a
   * matrix of size (projectedVectorSize, vectorSize).
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a list of projection vectors
   */
  public static List<Vector> generateVectorBasis(int projectedVectorSize, int vectorSize) {
    final DoubleFunction random = Functions.random();
    List<Vector> basisVectors = Lists.newArrayList();
    for (int i = 0; i < projectedVectorSize; ++i) {
      Vector basisVector = new DenseVector(vectorSize);
      basisVector.assign(random);
      basisVector.normalize();
      basisVectors.add(basisVector);
    }
    return basisVectors;
  }

  private void initialize(int numDimensions) {
    if (initialized)
      return;
    initialized = true;
    basisMatrix = generateBasis(numProjections, numDimensions);
    scalarProjections = Lists.newArrayList();
    for (int i = 0; i < numProjections; ++i) {
      scalarProjections.add(Sets.<WeightedThing<Vector>>newTreeSet());
    }
  }

  public ProjectionSearch(DistanceMeasure distanceMeasure, int numProjections,  int searchSize) {
    super(distanceMeasure);
    Preconditions.checkArgument(numProjections > 0 && numProjections < 100,
        "Unreasonable value for number of projections");

    this.searchSize = searchSize;
    this.numProjections = numProjections;
  }

  /**
   * Adds a WeightedVector into the set of projections for later searching.
   * @param v  The WeightedVector to add.
   */
  @Override
  public void add(Vector v) {
    initialize(v.size());
    Vector projection = basisMatrix.times(v);
    // Add the the new vector and the projected distance to each set separately.
    int i = 0;
    for (TreeSet<WeightedThing<Vector>> s : scalarProjections) {
      s.add(new WeightedThing<Vector>(v, projection.get(i++)));
    }
    int numVectors = scalarProjections.get(0).size();
    for (TreeSet<WeightedThing<Vector>> s : scalarProjections) {
      Preconditions.checkArgument(s.size() == numVectors, "Number of vectors in projection sets " +
          "differ");
      double firstWeight = s.first().getWeight();
      for (WeightedThing<Vector> w : s) {
        Preconditions.checkArgument(firstWeight <= w.getWeight(), "Weights not in non-decreasing " +
            "order");
        firstWeight = w.getWeight();
      }
    }
  }

  /**
   * Returns the number of scalarProjections that we can search
   * @return  The number of scalarProjections added to the search so far.
   */
  public int size() {
    if (scalarProjections == null)
      return 0;
    return scalarProjections.get(0).size();
  }

  /**
   * Searches for the query vector returning the closest limit referenceVectors.
   *
   * @param query the vector to search for.
   * @param limit the number of results to return.
   * @return a list of Vectors wrapped in WeightedThings where the "thing"'s weight is the
   * distance.
   */
  public List<WeightedThing<Vector>> search(final Vector query, int limit) {
    HashSet<Vector> candidates = Sets.newHashSet();

    Iterator<? extends Vector> projections = basisMatrix.iterator();
    for (TreeSet<WeightedThing<Vector>> v : scalarProjections) {
      Vector basisVector = projections.next();
      WeightedThing<Vector> projectedQuery = new WeightedThing<Vector>(query,
          query.dot(basisVector));
      for (WeightedThing<Vector> candidate : Iterables.concat(
          Iterables.limit(v.tailSet(projectedQuery, true), searchSize),
          Iterables.limit(v.headSet(projectedQuery, false).descendingSet(), searchSize))) {
        candidates.add(candidate.getValue());
      }
    }

    // If searchSize * scalarProjections.size() is small enough not to cause much memory pressure,
    // this is probably just as fast as a priority queue here.
    List<WeightedThing<Vector>> top = Lists.newArrayList();
    for (Vector candidate : candidates) {
      top.add(new WeightedThing<Vector>(candidate, distanceMeasure.distance(query, candidate)));
    }
    Collections.sort(top);
    return top.subList(0, Math.min(limit, top.size()));
  }

  public int getSearchSize() {
    return searchSize;
  }

  public void setSearchSize(int searchSize) {
    this.searchSize = searchSize;
  }

  @Override
  public Iterator<Vector> iterator() {
    return new AbstractIterator<Vector>() {
      private Iterator<WeightedThing<Vector>> projected = scalarProjections.get(0).iterator();
      @Override
      protected Vector computeNext() {
        if (!projected.hasNext()) {
          return endOfData();
        }
        return projected.next().getValue();
      }
    };
  }

  public boolean remove(Vector vector, double epsilon) {
    List<WeightedThing<Vector>> x = search(vector, 1);
    if (x.get(0).getWeight() < 1e-7) {
      Iterator<? extends Vector> basisVectors = basisMatrix.iterator();
      for (TreeSet<WeightedThing<Vector>> projection : scalarProjections) {
        if (!projection.remove(new WeightedThing<Vector>(null, vector.dot(basisVectors.next())))) {
          throw new RuntimeException("Internal inconsistency in ProjectionSearch");
        }
      }
      return true;
    } else {
      return false;
    }
  }

  @Override
  public void clear() {
    if (scalarProjections == null) {
      return;
    }
    for (TreeSet<WeightedThing<Vector>> set : scalarProjections) {
      set.clear();
    }
  }
}
