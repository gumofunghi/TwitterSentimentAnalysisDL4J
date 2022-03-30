package com.jiaqi.tweets.SentimentAnalysis;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;

import static com.jiaqi.tweets.SentimentAnalysis.CnnTrain.DATA_PATH;

public class LoadAndTestModel {

    public static void main(String[] args) throws IOException {
        System.out.println("----- Evaluation initializing -----");

        String trainedModelPath = "src/main/resources/LstmModel.zip";

        System.out.println("----- Evaluation starting -----");

        LoadAndTestModel deepLearner = new LoadAndTestModel();

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(trainedModelPath);

        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File("src/main/resources/word2vec.dat"));

        int batchSize = 50;

        SentimentDataIterator test = new SentimentDataIterator(wordVectors, batchSize, false);
        System.out.println(test.next().getFeatures().shape()[0]);
        System.out.println(test.next().getFeatures().shape()[1]);
        System.out.println(test.next().getFeatures().shape()[2]);
        // expected bad review
        String shortNegativeReview = "@AyekKamal yer lah sbb sombong mmg lah ";
        deepLearner.evaluate(test, model, shortNegativeReview);

        // another expected bad review
        String secondBadReview = "Batok kelapa menjadi bara, Terbakar semua tidak tersisa. Wahai saudara seiman senegara, Saya ucapakan selamat puas";
        deepLearner.evaluate(test, model, secondBadReview);

        // a good review follows (hopefully)
        String goodReview = "Sahur tengah malam kaya nya enak ya... ";
        deepLearner.evaluate(test, model, goodReview);

        System.out.println("----- Evaluation complete -----");
    }

    private void evaluate(SentimentDataIterator test, MultiLayerNetwork model, String review) throws IOException
    {
        INDArray features = test.loadFeaturesFromString(review, 100);
        System.out.println(features);
        INDArray networkOutput = model.output(features);
        System.out.println(networkOutput);
        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("Review: \n" + review);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Evaluate complete -----");
    }

}
