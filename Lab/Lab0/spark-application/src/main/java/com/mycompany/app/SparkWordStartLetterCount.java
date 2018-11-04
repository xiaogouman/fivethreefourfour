import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;

import java.util.Arrays;
import java.util.List;
import java.util.Collection;

public final class SparkWordStartLetterCount {

  public static void main(String[] args) throws Exception {

    //create Spark context with Spark configuration
    JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("wordstartlettercount")); 

    //set the input file
    JavaRDD<String> textFile = sc.textFile("in.txt");

    /**
    *   word start letter count process
    *   filter out empty words
    *   map the first letter
    *   sort in alphabetical order
    **/
    JavaPairRDD<Character, Integer> counts = textFile
    .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
    .filter(word -> !word.isEmpty())
    .mapToPair(word -> new Tuple2<>(word.charAt(0), 1))
    .reduceByKey((a, b) -> a + b)
    .sortByKey();

    //set the output folder
    counts.saveAsTextFile("outfile");
    //stop spark
  }
}
