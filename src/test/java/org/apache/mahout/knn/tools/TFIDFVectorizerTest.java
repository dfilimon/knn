package org.apache.mahout.knn.tools;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

public class TFIDFVectorizerTest {
  @Test
  public void testSimpleFile() throws IOException {
    TFIDFVectorizer vectorizer = new TFIDFVectorizer(new TFIDFScorer.Linear());
    Map<String, Integer> wordCounts = Maps.newHashMap();
    wordCounts.put("alfa", 23);
    wordCounts.put("beta", 2);
    wordCounts.put("gamma", 40);
    wordCounts.put("delta", 1);
    File tempFile = File.createTempFile("tfidf-vectorizer-test", ".txt");
    FileWriter writer = new FileWriter(tempFile);
    for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
      for (int i = 0;  i < entry.getValue(); ++i) {
        writer.write(entry.getKey() + " ");
      }
      writer.write('\n');
    }
    writer.close();

    Map<String, Integer> actualCounts = vectorizer.buildWordTFDictionaryForPath(tempFile.getPath());
    assertThat(actualCounts, is(wordCounts));

    List<String> pathList = Lists.newArrayList();
    pathList.add(tempFile.getPath());

    Iterable<Vector> vectors = vectorizer.vectorize(pathList);
    for (Vector vector : vectors) {
      System.out.println(vector.toString());
    }

    FileUtils.forceDeleteOnExit(tempFile);
  }
}
