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

    @GetMapping(value = "tweet", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> publish() throws IOException {
        logger.info("Test Test B");

        return this.webClient.get()
                .accept(MediaType.APPLICATION_JSON)
                .retrieve()
                .bodyToFlux(String.class)
                .map(message -> {
                    logger.info(message);
//                    if(!message.isEmpty()){
//                        try {
//                            Tweet tweet = new ObjectMapper()
//                                    .readerFor(Tweet.class)
//                                    .readValue(message);
//
////                        logger.info(text);
////
//                            FileWriter fw = new FileWriter("data.txt",  true);
//                            PrintWriter out = new PrintWriter(fw);
//                            out.println(tweet.getText());
//
//                            System.out.println(tweet.getText());
//
//                        } catch (IOException e) {
//                            System.out.println(e.getMessage());
//                            e.printStackTrace();
//                        }
//                    }
                    return message;
                });

    }

}
