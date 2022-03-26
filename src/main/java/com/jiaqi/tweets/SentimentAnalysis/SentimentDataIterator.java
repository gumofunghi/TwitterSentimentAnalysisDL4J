package com.jiaqi.tweets.SentimentAnalysis;

import lombok.SneakyThrows;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


//modified from SentimentExampleIterator.java https://github.com/breandan/deep-learning-samples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment/SentimentExampleIterator.java
public class SentimentDataIterator implements DataSetIterator {

    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
//    private final int truncateLength;

    private int cursor = 0;
    private int positiveFileLength;
    private int negativeFileLength;
    private final File positiveFile;
    private final File negativeFile;
    private final String positivePath;
    private final String negativePath;
    private final TokenizerFactory tokenizerFactory;
    private boolean training;

    private Scanner sPos;
    private Scanner sNeg;

    public  SentimentDataIterator(WordVectors wordVectors, int batchSize, boolean training) throws IOException {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
//        System.out.println(this.vectorSize);

        this.wordVectors = wordVectors;
//        this.truncateLength = truncateLength;

        this.training = training;

        if(this.training){
            positiveFile = new File("src/main/resources/labeled/positive_small.txt");
            positivePath = "src/main/resources/labeled/positive_small.txt";
            negativeFile = new File("src/main/resources/labeled/negative_small.txt");
            negativePath = "src/main/resources/labeled/negative_small.txt";
//            System.out.println("---------------------TRAIN______________________________");
        }else{
            positiveFile = new File("src/main/resources/test/positive.txt");
            positivePath = "src/main/resources/test/positive.txt";
            negativeFile = new File("src/main/resources/test/negative.txt");
            negativePath = "src/main/resources/test/negative.txt";

//            System.out.println("---------------------TEST______________________________");
        }

        totalExamples();
        sPos = new Scanner(positiveFile);
        sNeg = new Scanner(negativeFile);
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    @Override
    public DataSet next(int num) throws NoSuchElementException {
//        System.out.println("cursor: " + cursor + " dataset here ");
        if (cursor >= positiveFileLength + negativeFileLength || cursor/2 >= positiveFileLength || cursor/2 >= negativeFileLength) throw new NoSuchElementException();
        else{
            try{
//                System.out.println(num);
                return  nextDataSet(num);
            }catch (IOException e){
//                System.out.println(e);
                throw new RuntimeException(e);
            }
        }

    }

    private DataSet nextDataSet(int num) throws IOException{
        //load data to string
        //0: positive 1:negative

        List<String> tweets = new ArrayList<>(num);
        boolean[] positive = new boolean[num];
        for(int i=0; i<num && (cursor/2 < positiveFileLength && cursor/2 < negativeFileLength); i++){
            if(cursor % 2 == 0){
                //load positive tweets
                if (sPos.hasNext()){
                    String tweet = sPos.nextLine();
                    tweets.add(tweet);
                    positive[i] = true;
//                    System.out.println("POS: " + tweet);
                }
            }
            else{
                //load negative tweets
                if (sNeg.hasNext()){
                    String tweet = sNeg.nextLine();
                    tweets.add(tweet);
                    positive[i] = false;
//                    System.out.println("NEG: " + tweet);
                }
            }
//            System.out.println("pfile length: " + positiveFileLength);
//            System.out.println("nfile length: " + negativeFileLength);
//            System.out.println("CUrsor: " + cursor);
            cursor++;
        }
        //Second: tokenize tweets and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(tweets.size());
//        System.out.println(tweets.size());
        int maxLength = 0;
        for(String s : tweets){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());

        }

        //Create data for training
        //Here: we have reviews.size() examples of varying lengths
//        System.out.println(tweets.size() + " " + vectorSize + " " + maxLength);
        INDArray features = Nd4j.create(tweets.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(tweets.size(), 2, maxLength);    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(tweets.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(tweets.size(), maxLength);


        int[] temp = new int[2];
        for( int i=0; i<tweets.size(); i++ ){
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.size() && j<maxLength; j++ ){
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }

            int idx = (positive[i] ? 0 : 1);
//            System.out.println(maxLength);
            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
    }


    public int totalExamples() throws IOException {
        Path path = Paths.get(positivePath);
        long positiveLength = Files.lines(path).count();
        positiveFileLength = (int) positiveLength;

        path = Paths.get(negativePath);
        long negativeLength = Files.lines(path).count();
        negativeFileLength = (int) negativeLength;

        return (int)(positiveLength + negativeLength);
//        return 0;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @SneakyThrows
    @Override
    public void reset() {
        cursor = 0;

        this.sNeg.close();
        this.sPos.close();
        sPos = new Scanner(positiveFile);
        sNeg = new Scanner(negativeFile);



    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    public int cursor() {
        return cursor;
    }

    public int numExamples() throws IOException {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("positive","negative");
    }

    @SneakyThrows
    @Override
    public boolean hasNext() {

            if(cursor < numExamples() && (cursor/2 < positiveFileLength && cursor/2 < negativeFileLength)){
                return true;
            }
            else
                return false;

//        return false;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }
    @Override
    public  DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public INDArray loadFeaturesFromString(String reviewContents, int maxLength){
        List<String> tokens = tokenizerFactory.create(reviewContents).getTokens();
        List<String> tokensFiltered = new ArrayList<>();
        for(String t : tokens ){
            if(wordVectors.hasWord(t)) tokensFiltered.add(t);
        }
        int outputLength = Math.max(maxLength,tokensFiltered.size());


        INDArray features = Nd4j.create(1, vectorSize, outputLength);

        for( int j=0; j<tokensFiltered.size() && j<maxLength; j++ ){
            String token = tokensFiltered.get(j);
//            System.out.println(token + "    ==== 1 ====");
            INDArray vector = wordVectors.getWordVectorMatrix(token);
//            System.out.println(vector + "    ==== 2 ====");
            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
        }

        return features;
    }

}
