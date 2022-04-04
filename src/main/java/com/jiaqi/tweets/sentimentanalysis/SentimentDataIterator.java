package com.jiaqi.tweets.sentimentanalysis;

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
        this.wordVectors = wordVectors;
        this.training = training;

        if(this.training){ //training files and path
            positiveFile = new File("src/main/resources/labeled/positive.txt");
            positivePath = "src/main/resources/labeled/positive.txt";
            negativeFile = new File("src/main/resources/labeled/negative.txt");
            negativePath = "src/main/resources/labeled/negative.txt";
        }else{ //testing files and path
            positiveFile = new File("src/main/resources/test/positive.txt");
            positivePath = "src/main/resources/test/positive.txt";
            negativeFile = new File("src/main/resources/test/negative.txt");
            negativePath = "src/main/resources/test/negative.txt";
        }
        totalExamples();
        sPos = new Scanner(positiveFile);
        sNeg = new Scanner(negativeFile);
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    @Override
    public DataSet next(int num) throws NoSuchElementException {
        if (cursor >= positiveFileLength + negativeFileLength || cursor/2 >= positiveFileLength || cursor/2 >= negativeFileLength) throw new NoSuchElementException();
        else{
            try{
                return  nextDataSet(num);
            }catch (IOException e){
                throw new RuntimeException(e);
            }
        }

    }

    private DataSet nextDataSet(int num) throws IOException{
        //load data to string
        //0: positive 1:negative

        List<String> tweets = new ArrayList<>(num);
        boolean[] positive = new boolean[num];
        String tweet = "";
        for(int i=0; i<num && (cursor/2 < positiveFileLength && cursor/2 < negativeFileLength); i++){
            if(cursor % 2 == 0){
                //load positive tweets
                if (sPos.hasNext()){
                    tweet = sPos.nextLine();
                    positive[i] = true;
                }
            }
            else{
                //load negative tweets
                if (sNeg.hasNext()){
                    tweet = sNeg.nextLine();
                    positive[i] = false;
                }
            }
            tweet.replaceAll("http.*?[\\S]+", "")// remove links
                    .replaceAll("@[\\S]+", "")// remove usernames
                    .replaceAll("#", "")// replace hashtags by just words
                    .replaceAll("[\\s]+", " ");// correct all multiple white spaces to a single white space
            tweets.add(tweet);
            cursor++;
        }

        //Second: tokenize tweets and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(tweets.size());
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
        INDArray features = Nd4j.create(tweets.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(tweets.size(), 2, maxLength);    //Two labels: positive or negative
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
            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
    }


    public int totalExamples() throws IOException { //get total number of lines in text file
        Path path = Paths.get(positivePath);
        long positiveLength = Files.lines(path).count();
        positiveFileLength = (int) positiveLength;

        path = Paths.get(negativePath);
        long negativeLength = Files.lines(path).count();
        negativeFileLength = (int) negativeLength;

        return (int)(positiveLength + negativeLength);
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
        cursor = 0; //reset cursor to 0
        this.sNeg.close(); //close negative file scanner
        this.sPos.close(); //close positive file scanner
        sPos = new Scanner(positiveFile); //declare scanner again
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

        INDArray features = Nd4j.create(outputLength, vectorSize, tokensFiltered.size());

        for( int j=0; j<tokensFiltered.size() && j<maxLength; j++ ){
            String token = tokensFiltered.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
        }
        return features;
    }

}
