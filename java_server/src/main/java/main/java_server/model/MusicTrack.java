package main.java_server.model;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class MusicTrack {
    private String trackId;
    private String name;
    private String artist;
    private String album;
    private String albumImageUrl;
    private String previewUrl;
    private double similarity;
    private String genre;
}
