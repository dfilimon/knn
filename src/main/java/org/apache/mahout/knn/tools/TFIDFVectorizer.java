package org.apache.mahout.knn.tools;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Map;

public class TFIDFVectorizer {
  // The word dictionary from Lucene tokens to the document frequency of the word.
  Map<String, Integer> wordDFDictionary = Maps.newHashMap();
  // The scoring function that gets a (TF, DF) pair and computes the score.
  Function<TFIDFScorer.Tuple, Double> tfIdfScorer;

  public TFIDFVectorizer(Function<TFIDFScorer.Tuple, Double> tfIdfScorer) {
    this.tfIdfScorer = tfIdfScorer;
  }

  /**
   * Creates a sequence file of Text, VectorWritable for a set of documents trying different
   * approaches to TF/IDF scoring.
   * @param args the first argument is the folder/file where the documents are. It will be
   *             scanned recursively to get the list of all documents to be tokenized. The second
   *             argument is the name of output seqfile containing the vectors. The third
   *             argument is the canonical name of the scoring class to be used.
   * @throws IOException
   * @see TFIDFScorer
   */
  public static void main(String[] args) throws IOException, ClassNotFoundException, NoSuchMethodException, InvocationTargetException, IllegalAccessException, InstantiationException {
    // Create the list of documents to be vectorized.
    List<String> paths = Lists.newArrayList();
    FileContentsToSequenceFiles.getRecursiveFilePaths(args[0], paths);

    // Parse arguments and see what scorer to use.
    Function<TFIDFScorer.Tuple, Double> tfIdfScorer =
        (Function<TFIDFScorer.Tuple, Double>) Class.forName(args[2]).getConstructor().newInstance();

    // Vectorize the documents and write them out.
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    TFIDFVectorizer vectorizer = new TFIDFVectorizer(tfIdfScorer);
    vectorizer.vectorize(paths,  new Path(args[1]), fs, conf);
  }

  public void vectorize(List<String> paths, Path outputPath, FileSystem fs, Configuration conf) throws IOException {
    SequenceFile.Writer writer =
        SequenceFile.createWriter(fs, conf, outputPath, Text.class, VectorWritable.class);
    List<Map<String, Integer>> wordTFDictionaries = Lists.newArrayList();
    System.out.println("Building global document dictionary");
    // Build the dictionary for all the files.
    for (String path : paths) {
      Map<String, Integer> wordTFDictionary = buildWordTFDictionaryForPath(path);
      for (Map.Entry<String, Integer> wordEntry : wordTFDictionary.entrySet()) {
        findApplyFunctionOrInitialize(wordDFDictionary, wordEntry.getKey(), new PlusOne(), 1);
      }
      wordTFDictionaries.add(wordTFDictionary);
    }
    // Build the actual vectors
    int numWords = wordDFDictionary.size();
    for (int i = 0; i < paths.size(); ++i) {
      String path = paths.get(i);
      Map<String, Integer> wordTFDictionary = wordTFDictionaries.get(i);
      Vector documentVector = new RandomAccessSparseVector(numWords, numWords);
      // Build the vector, element by element.
      int wordIndex = 0;
      for (Map.Entry<String, Integer> dfEntry : wordDFDictionary.entrySet()) {
        Integer termFrequnecy = wordTFDictionary.get(dfEntry.getKey());
        if (termFrequnecy != null) {
          documentVector.set(wordIndex++,
              tfIdfScorer.apply(
                  new TFIDFScorer.Tuple(termFrequnecy.intValue(), dfEntry.getValue(), numWords)));
        }
      }
      writer.append(new Text(path), new VectorWritable(documentVector));
      System.out.println("Wrote vector for " + path);
    }
    writer.close();
  }

  public static void writeVectorizedDocumentsToFile(List<Pair<String, Vector>> vectorizedDocuments,
                                                    Path outputPath,
                                                    FileSystem fs,
                                                    Configuration conf) throws IOException {
    SequenceFile.Writer writer =
        SequenceFile.createWriter(fs, conf, outputPath, Text.class, VectorWritable.class);
    for (Pair<String, Vector> document : vectorizedDocuments) {
      writer.append(document.getFirst(), new VectorWritable(document.getSecond()));
    }
    writer.close();
  }

  /**
   * Builds a term frequency dictionary for the words in a file.
   * @param path the name of the file to be processed.
   * @return a map of words to frequence counts.
   */
  public Map<String, Integer> buildWordTFDictionaryForPath(String path) throws IOException {
    Tokenizer tokenizer = new StandardTokenizer(Version.LUCENE_36, new FileReader(path));
    CharTermAttribute cattr = tokenizer.addAttribute(CharTermAttribute.class);
    Map<String, Integer> wordTFDictionary = Maps.newHashMap();
    while (tokenizer.incrementToken()) {
      String word = cattr.toString();
      findApplyFunctionOrInitialize(wordTFDictionary, word, new PlusOne(), 1);
    }
    tokenizer.end();
    tokenizer.close();
    return wordTFDictionary;
  }

  public <K, V> boolean findApplyFunctionOrInitialize(Map<K, V> map, K key, Function<V, V> f,
                                                      V initValue) {
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
}
