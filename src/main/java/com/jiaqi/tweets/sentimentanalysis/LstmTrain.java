package com.jiaqi.tweets.sentimentanalysis;

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
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

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

            //Run evaluation for train and test dataset iterator respectively
            evaluateModel(train, model);
            evaluateModel(test, model);
        }
//        model.save(new File("src/main/resources/LstmModel.zip"), true);
        }

        public static void evaluateModel(SentimentDataIterator dataset, MultiLayerNetwork model){
            Evaluation evaluation = new Evaluation();
            while (dataset.hasNext()) {
                DataSet t = dataset.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = model.output(features, true, inMask, outMask);

                evaluation.evalTimeSeries(labels, predicted, outMask);
            }
            dataset.reset();

            System.out.println(evaluation.stats());

        }

}
