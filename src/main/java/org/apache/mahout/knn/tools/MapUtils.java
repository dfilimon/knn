package org.apache.mahout.knn.tools;

import com.google.common.base.Function;

import java.util.Map;

public class MapUtils {
  /**
   * Find the given key and applies the function f to it replacing the corresponding value of the
   * key in the map with the value of the function or, if the key is not found,
   * creates a new element in the map for the given key whose value is initValue.
   * @param map to work on
   * @param f the function to apply to the old value
   * @param initValue the initial value to associate to the key
   * @param key the key we're looking for
   * @return true if a new element was added (the key wasn't found and a new pair (key,
   * initValue) was created)
   */
  public static <K, V> boolean findAndApplyFunctionOrInitialize(Map<K, V> map, Function<V, V> f,
                                                                V initValue, K key) {
    V oldValue = map.remove(key);
    boolean added = true;
    V newValue = initValue;
    if (oldValue != null) {
      newValue = f.apply(oldValue);
      added = false;
    }
    map.put(key, newValue);
    return added;
  }

  public static <K, V> V findOrInitialize(Map<K, V> map, Class<V> vClass, K key)
      throws IllegalAccessException, InstantiationException {
    V value = map.get(key);
    if (value == null) {
      value = vClass.newInstance();
      map.put(key, value);
    }
    return value;
  }

  static class PlusOne implements Function<Integer, Integer> {
    @Override
    public Integer apply(Integer input) {
      return input + 1;
    }
  }

}
