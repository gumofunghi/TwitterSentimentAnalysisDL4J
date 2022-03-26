package com.jiaqi.tweets;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;


@Configuration
public class AppConfiguration {

    private static final String TWITTER_API = "https://api.twitter.com/2/tweets/search/stream";
    private static final String HEADER = "";

    private final Logger logger = LoggerFactory.getLogger(AppConfiguration.class);

    @Bean
    public WebClient webClient(){

        logger.info("Test Test A");

        return WebClient.builder()
                .defaultHeader("Authorization", "Bearer AAAAAAAAAAAAAAAAAAAAAAfPaAEAAAAActJcntpFgAQIhZPy1Egug5SCuHo%3DqqsqYPI0P67WqpUCKiXUGdMBCjJlXwg4zNTMnENDFHaaTPCVZ1")
                .baseUrl(TWITTER_API)
                .build();
    }

    @Bean
    public Sinks.Many<Tweet> sink(){
        return Sinks.many().replay().latest();
    }

    @Bean
    public Flux<Tweet> flux(Sinks.Many<Tweet> sink){
        return sink.asFlux().cache();
    }
}
