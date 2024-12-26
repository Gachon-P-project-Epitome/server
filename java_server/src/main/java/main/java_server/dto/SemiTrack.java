package main.java_server.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class SemiTrack {
    @JsonProperty("track_id")
    private String trackId;

    private double similarity;
    private String genre;

}
