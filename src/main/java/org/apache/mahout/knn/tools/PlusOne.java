package org.apache.mahout.knn.tools;

import com.google.common.base.Function;

/**
 * Created with IntelliJ IDEA.
 * User: dan
 * Date: 12/7/12
 * Time: 5:38 PM
 * To change this template use File | Settings | File Templates.
 */
class PlusOne implements Function<Integer, Integer> {
  @Override
  public Integer apply(@javax.annotation.Nullable Integer input) {
    int n = input.intValue();
    return new Integer(n + 1);
  }
}
