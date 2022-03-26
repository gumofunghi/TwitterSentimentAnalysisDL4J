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

public class LstmTrain {

    public static void main(String[] args) throws IOException, InterruptedException {

//        File model = new File("word2vec.txt");
//        Word2Vec vec = WordVectorSerializer.readWord2VecModel(model);
//        ParagraphVectors paragraphVectors;
//        LabelAwareIterator labelAwareIterator;

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        ClassPathResource resource = new ClassPathResource("/data.txt");

        File file = resource.getFile();
        SentenceIterator iterator = new BasicLineIterator(file);
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        LabelsSource labelFormat = new LabelsSource("LINE_");

//        labelAwareIterator = new FileLabelAwareIterator.Builder()
//                .addSourceFolder(resource.getFile())
//                .build();


//        ParagraphVectors vec = new ParagraphVectors.Builder()
//                .minWordFrequency(1)
//                .iterations(5)
//                .epochs(1)
//                .layerSize(100)
//                .learningRate(0.025)
//                .labelsSource(labelFormat)
//                .windowSize(5)
//                .iterate(iterator)
//                .trainWordVectors(false)
//                .tokenizerFactory(tokenizerFactory)
//                .sampling(0)
//                .build();
//
//        vec.fit();
//        double similar1 = vec.similarity("LINE_28", "LINE_37");
//        System.out.println("Comparing lines 98 & 124, Similarity = " + similar1);

//
        File modelFile = new File("src/main/resources/word2vec.dat");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(modelFile);;

        int inputNeurons = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
//        System.out.println(inputNeurons);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.002))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .list()
                .layer(new LSTM.Builder()
                        .nIn(inputNeurons)
                        .nOut(300)
                        .activation(Activation.TANH)
                        .build())
                .layer(new RnnOutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(300)
                        .nOut(2)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        int batchSize = 50;

        SentimentDataIterator train = new SentimentDataIterator(wordVectors, batchSize, true);
        SentimentDataIterator test = new SentimentDataIterator(wordVectors, batchSize, true);

        int epoch = 10;

        //train with number of epoch
        for (int i = 0; i < epoch; i++) {
            model.fit(train);
            train.reset();
            System.out.println("Epoch " + (i+1) + " complete. Starting evaluation:");

            //Run evaluation
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

        //load sample to generate prediction
//        String sampleTweet1 = "Bantulah mereka yang kesusahan, berikan sumbangan pada yang memerlukan. Semoga Allah permudahkan urusan kita.";
        String sampleTweet2 = "cakap dgn org bodoh sampai bila pon tak habis";
//        String sampleTweet3 = "Yeyeah... Mendung dah sampai... Semoga terus bawa hujan";
//        String sampleTweet4 = "Penat la kepala otak ni";

//        String sampleTweet5 = "Terima kasih, happy la";


        INDArray features = test.loadFeaturesFromString(sampleTweet2, 100);
        INDArray networkOutput = model.output(features);
        int timeSeriesLength = (int)networkOutput.size(2);

        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("Tweet: \n" + sampleTweet2);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Example complete -----");

    }

}
