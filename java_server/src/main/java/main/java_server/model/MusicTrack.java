package main.java_server.model;


import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;

@Data
@NoArgsConstructor
@RequiredArgsConstructor
public class MusicTrack {
    private String name;
    private String artist;
    private String album;
    private String albumImageUrl;
    private String previewUrl;
    private double similarity;
}
