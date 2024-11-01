package main.java_server.Dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
public class Results {
    @JsonProperty("original_file_id")
    private String originalFileId;
    @JsonProperty("similarity_score")
    private double similarityScore;
    @JsonProperty("similar_tracks")
    private List<SemiTrack> similarTracks;
}
