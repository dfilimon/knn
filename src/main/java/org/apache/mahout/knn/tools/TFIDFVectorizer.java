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
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.encoders.AdaptiveWordValueEncoder;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class TFIDFVectorizer {
  // The word dictionary from Lucene tokens to the document frequency of the word.
  private Map<String, Integer> wordDFDictionary = Maps.newHashMap();
  // The scoring function that gets a (TF, DF) pair and computes the score.
  private Function<TFIDFScorer.Tuple, Double> tfIdfScorer;

  private AdaptiveWordValueEncoder encoder;

  public TFIDFVectorizer(Function<TFIDFScorer.Tuple, Double> tfIdfScorer) {
    this.tfIdfScorer = tfIdfScorer;
    this.encoder = new AdaptiveWordValueEncoder("encoder");
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

  public Iterable<Vector> vectorize(final Iterable<String> paths) throws IOException {
    final List<Map<String, Integer>> wordTFDictionaries = Lists.newArrayList();
    int numPaths = 0;
    // Build the dictionary for all the files.
    for (String path : paths) {
      Map<String, Integer> wordTFDictionary = buildWordTFDictionaryForPath(path);
      for (Map.Entry<String, Integer> wordEntry : wordTFDictionary.entrySet()) {
        MapUtils.findAndApplyFunctionOrInitialize(wordDFDictionary, new MapUtils.PlusOne(), 1, wordEntry.getKey());
      }
      wordTFDictionaries.add(wordTFDictionary);
      ++numPaths;
    }
    // Build the actual vectors.
    final int finalNumPaths = numPaths;
    return new Iterable<Vector>() {
      @Override
      public Iterator<Vector> iterator() {
        return new Iterator<Vector>() {
          private int i = 0;
          private int numWords = wordDFDictionary.size();

          @Override
          public boolean hasNext() {
            return i < finalNumPaths;
          }

          @Override
          public Vector next() {
            Map<String, Integer> wordTFDictionary = wordTFDictionaries.get(i++);
            Vector documentVector = new SequentialAccessSparseVector(numWords);
            int wordIndex = 0;
            for (Map.Entry<String, Integer> dfEntry : wordDFDictionary.entrySet()) {
              Integer termFrequency = wordTFDictionary.get(dfEntry.getKey());
              if (termFrequency != null) {
                documentVector.set(wordIndex++,
                    tfIdfScorer.apply(new TFIDFScorer.Tuple(termFrequency, dfEntry.getValue(),
                        finalNumPaths)));
              }
            }
            return documentVector;
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();
          }
        };
      }
    };
  }

  public void vectorize(List<String> paths, Path outputPath, FileSystem fs, Configuration conf) throws IOException {
    SequenceFile.Writer writer =
        SequenceFile.createWriter(fs, conf, outputPath, Text.class, VectorWritable.class);
    Iterable<Vector> documentVectors = vectorize(paths);
    int i = 0;
    for (Vector documentVector : documentVectors) {
      String path = paths.get(i++);
      writer.append(new Text(path), new VectorWritable(documentVector));
      if (i % 500 == 0) {
        float percent = (float) i / paths.size() * 100;
        System.out.println(percent + "%");
      }
    }
    System.out.println("Finished writing vectors to " + outputPath.getName());
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
   * Builds a term frequency dictionary for the words in a reader.
   * @param reader the reader to get the words from.
   * @return a map of words to frequency counts.
   */
  public Map<String, Integer> buildWordTFDictionary(Reader reader) throws IOException {
    Tokenizer tokenizer = new StandardTokenizer(Version.LUCENE_36, reader);
    CharTermAttribute cattr = tokenizer.addAttribute(CharTermAttribute.class);
    Map<String, Integer> wordTFDictionary = Maps.newHashMap();
    while (tokenizer.incrementToken()) {
      String word = cattr.toString();
      MapUtils.findAndApplyFunctionOrInitialize(wordTFDictionary, new MapUtils.PlusOne(), 1, word);
    }
    tokenizer.end();
    tokenizer.close();
    return wordTFDictionary;
  }

  /**
   * Builds a term frequency dictionary for the words in a file.
   * @param path the name of the file to be processed.
   * @return a map of words to frequency counts.
   */
  public Map<String, Integer> buildWordTFDictionaryForPath(String path) throws IOException {
    return buildWordTFDictionary(new FileReader(path));
  }

}

