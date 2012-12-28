package org.apache.mahout.knn.tools;

import com.google.common.base.Function;

import javax.annotation.Nullable;

public class TFIDFScorer {
  static public class Tuple {
    double tf;
    double df;
    double n;

    public Tuple(int tf, int df, int n) {
      this.tf = tf;
      this.df = df;
      this.n = n;
    }
  }

  static public class Linear implements Function<Tuple, Double> {
    @Override
    public Double apply(Tuple input) {
      return input.tf * Math.log(input.n / input.df);
    }
  }

  static public class Const implements Function<Tuple, Double> {
    @Override
    public Double apply(Tuple input) {
      return Math.log(input.n / input.df);
    }
  }

  static public class Log implements Function<Tuple, Double> {
    @Override
    public Double apply(Tuple input) {
      return Math.log(input.tf) * Math.log(input.n / input.df);
    }
  }

  static public class Sqrt implements Function<Tuple, Double> {
    @Override
    public Double apply(Tuple input) {
      return Math.sqrt(input.tf) * Math.log(input.n / input.df);
    }
  }
}
