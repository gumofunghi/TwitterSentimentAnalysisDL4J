package com.jiaqi.tweets.SentimentAnalysis;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class LstmTrain {

    public static void main(String[] args) throws IOException, InterruptedException {

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        ClassPathResource resource = new ClassPathResource("/data.txt");

        File file = resource.getFile();
        SentenceIterator iterator = new BasicLineIterator(file);
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        LabelsSource labelFormat = new LabelsSource("LINE_");
        
        File modelFile = new File("src/main/resources/word2vec.dat");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(modelFile);;

        int inputNeurons = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
//        System.out.println(inputNeurons);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .l2(1e-4)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .list()
                .layer(new LSTM.Builder()
                        .nIn(inputNeurons)
                        .nOut(100)
//                        .dropOut(0.5)
                        .activation(Activation.TANH)
                        .build())
                .layer(new RnnOutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(100)
                        .nOut(2)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        int batchSize = 100;

        SentimentDataIterator train = new SentimentDataIterator(wordVectors, batchSize, true);
        SentimentDataIterator test = new SentimentDataIterator(wordVectors, batchSize, false);

        int epoch = 5;

        //train with number of epoch
        for (int i = 0; i < epoch; i++) {
            model.fit(train);
            train.reset();
            System.out.println("Epoch " + (i+1) + " complete. Starting evaluation:");

            //Run evaluation

            Evaluation trainevaluation = new Evaluation();
            while (train.hasNext()) {
                DataSet t = train.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = model.output(features, true, inMask, outMask);

                trainevaluation.evalTimeSeries(labels, predicted, outMask);
            }
            train.reset();

            System.out.println(trainevaluation.stats());


            Evaluation evaluation = new Evaluation();
            while (test.hasNext()) {
                DataSet t = test.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = model.output(features, false, inMask, outMask);

                evaluation.evalTimeSeries(labels, predicted, outMask);
            }

            test.reset();

            System.out.println(evaluation.stats());



        }

//        model.save(new File("src/main/resources/LstmModel.zip"), true);

        //load sample to generate prediction
        List<String> sampleTweets = Arrays.asList("Bantulah mereka yang kesusahan, berikan sumbangan pada yang memerlukan. Semoga Allah permudahkan urusan kita.",
                "Sahur tengah malam kaya nya enak ya... ",
                "cakap dgn org bodoh sampai bila pon tak habis",
                "Penat la kepala otak ni",
                "Terima kasih, happy la, suka ini sangat",
                "Comel anak patung",
                "Rezeki duduk tepi tingkap. Ahakkkks",
                "Kan best kalau ada peti sejuk mini dalam bilik",
                "Tukarlah nama jadi Periuk babi tentu tak ada org nak ambil dah hajgsjagajahagaha");

        for(int i=0; i<sampleTweets.size();i++){

//            INDArray result = classifier.outputSingle(features);
//
//            String[] labels = new String[]{"Negative", "Positive"};
//
//            List<Double> scores = new ArrayList<>();
//
//            for (int i = 0; i < labels.length; i++) {
//                System.out.println(labels[i] + ": " + result.getDouble(i));
//                scores.add(result.getDouble(i));
//            }
//
//            return scores.get(1) - scores.get(0);

            INDArray features = test.loadFeaturesFromString(sampleTweets.get(i), 0);
            INDArray networkOutput = model.output(features);//result
            System.out.println("NetworkOutput");
            System.out.println(networkOutput);

            //---
            int timeSeriesLength = (int)networkOutput.size(2);
            System.out.println("TimeSeriesLength");
            System.out.println(timeSeriesLength);

            INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0));
            //---

            System.out.println("ProbabilitiesAtLastWord");
            System.out.println(probabilitiesAtLastWord);
            System.out.println("\n\n-------------------------------");
            System.out.println("Tweet: \n" + sampleTweets.get(i));
            System.out.println("\n\nProbabilities at last time step:");
            System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
            System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));
            System.out.println("\n=========================\n\n");

        }



    }

}
