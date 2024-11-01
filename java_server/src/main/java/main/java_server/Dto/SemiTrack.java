package main.java_server.Dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class SemiTrack {
    @JsonProperty("track_id")
    private String trackId;

    private double similarity;
}
