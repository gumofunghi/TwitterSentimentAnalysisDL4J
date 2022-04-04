package com.jiaqi.tweets.sentimentanalysis;


import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Word2Vec {

    public static void main(String[] args) throws IOException {

        SentenceIterator iterator = new LineSentenceIterator(new File("src/main/resources/data.txt"));
        List<String> stopwords = Files.readAllLines(Paths.get("src/main/resources/malaysian_stopwords.txt"), Charset.defaultCharset() );

        iterator.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String s) {
                s.replaceAll("http.*?[\\S]+", "")// remove links
                        .replaceAll("@[\\S]+", "")// remove usernames
                        .replaceAll("#", "")// replace hashtags by just words
                        .replaceAll("[\\s]+", " ");// correct all multiple white spaces to a single white space

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
                .stopWords(stopwords)
                .build();
        vec.setSentenceIterator(iterator);
        vec.fit();

//        WordVectorSerializer.writeWordVectors(vec, "src/main/resources/word2vec.txt");
        WordVectorSerializer.writeWord2VecModel(vec, "src/main/resources/word2vec.dat");



    }


}
