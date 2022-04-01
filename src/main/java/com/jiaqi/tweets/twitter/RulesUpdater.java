package com.jiaqi.tweets.twitter;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.config.CookieSpecs;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.net.URISyntaxException;
import java.util.*;

public class RulesUpdater {

    //this class is to check, update and delete the rules in Twitter API Filter Stream
    //in a very manual method
    public static void main(String args[]) throws IOException, URISyntaxException {
        String bearerToken = "AAAAAAAAAAAAAAAAAAAAAAfPaAEAAAAActJcntpFgAQIhZPy1Egug5SCuHo%3DqqsqYPI0P67WqpUCKiXUGdMBCjJlXwg4zNTMnENDFHaaTPCVZ1";

        //delete rule //uncomment when needed
//        String ruleID = "1509723459852828674";//raya
//        deleteRule(bearerToken, ruleID);
//        ruleID = "1509723459852828675";//makan
//        deleteRule(bearerToken, ruleID);
//        ruleID = "1509723459852828676";//uitm
//        deleteRule(bearerToken, ruleID);

//
//        //add rule //uncomment when needed
//        Map<String, String> rule = new HashMap<>();
//        rule.put("-is:retweet -has:mentions -has:media -has:images -has:videos UiTM", "UiTM");
//        rule.put("-is:retweet -has:mentions -has:media -has:images -has:videos makan", "Makan");
//        rule.put("-is:retweet -has:mentions -has:media -has:images -has:videos raya ", "Raya");
//
//        setRule(bearerToken, rule);

        //list of rules
        List<String> rules = getRulesList(bearerToken);
        for (int i=0; i<rules.size(); i++)
        {
            System.out.println(rules.get(i));
        }
    }

    public static List<String> getRulesList(String bearerToken) throws URISyntaxException, IOException {
        List<String> rules = new ArrayList<>();

        HttpClient httpClient = HttpClients.custom()
                .setDefaultRequestConfig(RequestConfig.custom()
                        .setCookieSpec(CookieSpecs.STANDARD).build())
                .build();

        URIBuilder uriBuilder = new URIBuilder("https://api.twitter.com/2/tweets/search/stream/rules");

        HttpGet httpGet = new HttpGet(uriBuilder.build());
        httpGet.setHeader("Authorization", String.format("Bearer %s", bearerToken));
        httpGet.setHeader("content-type", "application/json");
        HttpResponse response = httpClient.execute(httpGet);
        HttpEntity entity = response.getEntity();
        if (null != entity) {
            JSONObject json = new JSONObject(EntityUtils.toString(entity, "UTF-8"));
            if (json.length() > 1) {
                JSONArray array = (JSONArray) json.get("data");
                for (int i = 0; i < array.length(); i++) {

                    JSONObject jsonObject = (JSONObject) array.get(i);
                    //System.out.println(jsonObject);
                    rules.add(jsonObject.getString("id") + " ; " + jsonObject.getString("tag")+ " ; " + jsonObject.getString("value"));
                }
            }
        }

        return rules;
    }

    public static void setRule(String bearerToken, Map<String, String> rules ) throws URISyntaxException, IOException {
        HttpClient httpClient = HttpClients.custom()
                .setDefaultRequestConfig(RequestConfig.custom()
                        .setCookieSpec(CookieSpecs.STANDARD).build())
                .build();

        URIBuilder uriBuilder = new URIBuilder("https://api.twitter.com/2/tweets/search/stream/rules");

        HttpPost httpPost = new HttpPost(uriBuilder.build());
        httpPost.setHeader("Authorization", String.format("Bearer %s", bearerToken));
        httpPost.setHeader("content-type", "application/json");
        StringEntity body = new StringEntity(getFormattedString("{\"add\": [%s]}", rules));
        httpPost.setEntity(body);
        HttpResponse response = httpClient.execute(httpPost);
        HttpEntity entity = response.getEntity();
        if (null != entity) {
            System.out.println(EntityUtils.toString(entity, "UTF-8"));
        }
    }

    public static void deleteRule(String bearerToken, String rule_id) throws URISyntaxException, IOException {
        HttpClient httpClient = HttpClients.custom()
                .setDefaultRequestConfig(RequestConfig.custom()
                        .setCookieSpec(CookieSpecs.STANDARD).build())
                .build();

        URIBuilder uriBuilder = new URIBuilder("https://api.twitter.com/2/tweets/search/stream/rules");

        HttpPost httpPost = new HttpPost(uriBuilder.build());
        httpPost.setHeader("Authorization", String.format("Bearer %s", bearerToken));
        httpPost.setHeader("content-type", "application/json");
        StringEntity body = new StringEntity("{ \"delete\": { \"ids\": [" + "\"" + rule_id + "\"" +  "]}}");
        System.out.println(body);
        httpPost.setEntity(body);
        HttpResponse response = httpClient.execute(httpPost);
        HttpEntity entity = response.getEntity();
        if (null != entity) {
            System.out.println(EntityUtils.toString(entity, "UTF-8"));
        }
    }

    private static String getFormattedString(String string, Map<String, String> rules) {
        StringBuilder sb = new StringBuilder();
        if (rules.size() == 1) {
            String key = rules.keySet().iterator().next();
            return String.format(string, "{\"value\": \"" + key + "\", \"tag\": \"" + rules.get(key) + "\"}");
        } else {
            for (Map.Entry<String, String> entry : rules.entrySet()) {
                String value = entry.getKey();
                String tag = entry.getValue();
                sb.append("{\"value\": \"" + value + "\", \"tag\": \"" + tag + "\"}" + ",");
            }
            String result = sb.toString();
            return String.format(string, result.substring(0, result.length() - 1));
        }
    }

}

