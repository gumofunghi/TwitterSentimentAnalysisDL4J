package com.jiaqi.tweets.SentimentAnalysis;


import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.util.SerializationUtils;

import java.io.File;
import java.io.IOException;

public class Word2Vec {

    public static void main(String[] args) throws IOException {

        SentenceIterator iterator = new LineSentenceIterator(new File("src/main/resources/data.txt"));

        iterator.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String s) {
                return s.toLowerCase();
            }
        });

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        org.deeplearning4j.models.word2vec.Word2Vec vec =  new org.deeplearning4j.models.word2vec.Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(5)
                .layerSize(300)
                .seed(123)
                .windowSize(5)
                .learningRate(0.025)
                .iterate(iterator)
                .tokenizerFactory(tokenizerFactory)
                .build();
        vec.setSentenceIterator(iterator);
        vec.fit();

//        WordVectorSerializer.writeWordVectors(vec, "src/main/resources/word2vec.txt");
        WordVectorSerializer.writeWord2VecModel(vec, "src/main/resources/word2vec.dat");



    }


}
