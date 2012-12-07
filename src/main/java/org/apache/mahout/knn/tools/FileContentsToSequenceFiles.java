package org.apache.mahout.knn.tools;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import java.io.*;
import java.util.Iterator;
import java.util.List;

public class FileContentsToSequenceFiles {
  public static void main(String args[]) throws IOException {
    Preconditions.checkArgument(args.length == 2);

    List<String> filePaths = Lists.newArrayList();
    getRecursiveFilePaths(args[0], filePaths);
    System.out.printf("Total number of files to convert: %d\n", filePaths.size());

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    Path outPath = new Path(args[1]);
    writeFilesToSequenceFile(filePaths, outPath, fs, conf);
    System.out.println("Finished writing sequence file");

    SequenceFile.Reader reader = new SequenceFile.Reader(fs, outPath, conf);
    System.out.printf("K [%s] V[%s]\n", reader.getKeyClassName(), reader.getValueClassName());
    Text filePath = new Text();
    Text fileContents = new Text();
    long i = 0;
    Iterator<String> pathIterator = filePaths.iterator();
    while (reader.next(filePath, fileContents)) {
      String itPath = pathIterator.next();
      String seqPath = filePath.toString();
      Preconditions.checkArgument(itPath.equals(seqPath), "Expected " + itPath + " got " + seqPath);
      ++i;
    }
    System.out.printf("%d\n", i);
  }

  public static void writeFilesToSequenceFile(List<String> filePaths, Path outFile,
                                               FileSystem fs, Configuration conf) throws IOException {
    SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, outFile,
        Text.class, Text.class);
    for (String filePath : filePaths) {
      Text text = new Text();
      text.set(FileUtils.readFileToString(new File(filePath)));
      writer.append(new Text(filePath), text);
    }
    writer.close();
  }

  public static void getRecursiveFilePaths(String base, List<String> filePaths) throws
      IOException {
    File baseFile = new File(base);
    for (File child : baseFile.listFiles()) {
      String childPath = child.getCanonicalPath();
      if (child.isFile()) {
        filePaths.add(childPath);
      } else {
        getRecursiveFilePaths(childPath, filePaths);
      }
    }
  }
}
