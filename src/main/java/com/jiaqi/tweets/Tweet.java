package com.jiaqi.tweets;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.io.Serializable;
import java.util.List;
import java.util.Map;

@Data
@JsonIgnoreProperties("matching_rules")
public class Tweet implements Serializable{

    private String id;
    private String text;
//    private List<String> rule_id;
//    private List<String> rule_tag;

    public String getId() {
        return id;
    }

    public String getText() {
        return text;
    }

//    public List<String> getRule_tag() {
//        return rule_tag;
//    }
//
//    public List<String> getRule_id() {
//        return rule_id;
//    }

    @SuppressWarnings("unchecked")
    @JsonProperty("data")
    private void unpackedNested(Map<String, Object> data){
        this.id = (String)data.get("id");
        this.text = (String)data.get("text");
    }
}
