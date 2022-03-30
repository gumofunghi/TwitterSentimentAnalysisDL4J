package com.jiaqi.tweets.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.jiaqi.tweets.Tweet;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.io.*;

@RestController
public class TweetController {

    private final Logger logger = LoggerFactory.getLogger(TweetController.class);
    @Autowired
    private WebClient webClient;
    @Autowired
    private Sinks.Many<Tweet> sink;

    private SentimentAnalysisController sentimentAnalysisController;

    @GetMapping(value = "tweet", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> publish() throws IOException {
        logger.info("flux");
        return this.webClient.get()
                .accept(MediaType.APPLICATION_JSON)
                .retrieve()
                .bodyToFlux(String.class)
                .map(message -> {

                    logger.info(message);

//                    if(message != null)
//                        sentimentAnalysisController.evaluateTweet(message);

                    return message;
                });

    }

}
