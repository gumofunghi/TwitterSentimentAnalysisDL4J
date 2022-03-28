package com.jiaqi.tweets.controller;

import com.jiaqi.tweets.SentimentAnalysis.SentimentDataIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;

public class SentimentAnalysisController {

    private String trainedModelPath;
    private MultiLayerNetwork model;
    private WordVectors wordVectors;
    private SentimentDataIterator test;

    public SentimentAnalysisController() throws IOException {

        trainedModelPath = "src/main/resources/LstmModel.zip";
        model = ModelSerializer.restoreMultiLayerNetwork(trainedModelPath);
        wordVectors = WordVectorSerializer.loadStaticModel(new File("src/main/resources/word2vec.dat"));
        test = new SentimentDataIterator(wordVectors, 100, false);

    }

    //to do sentiment analysis on tweet
    public void evaluateTweet(SentimentDataIterator test, MultiLayerNetwork model, String tweet) throws IOException
    {
        INDArray features = test.loadFeaturesFromString(tweet, 100);
        INDArray networkOutput = model.output(features);
        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("Tweet: \n" + tweet);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Evaluate complete -----");
    }
}
