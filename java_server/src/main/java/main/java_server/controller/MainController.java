package main.java_server.controller;

import lombok.extern.slf4j.Slf4j;
import main.java_server.dto.Results;
import main.java_server.dto.SemiTrack;
import main.java_server.model.MusicTrack;
import main.java_server.service.FindTrackService;
import main.java_server.service.SendMusic;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Controller
@CrossOrigin("*")
public class MainController {

    private final SendMusic sendMusic;
    private final FindTrackService findTrackService;

    @Autowired
    public MainController(SendMusic sendMusic, FindTrackService findTrackService) {
        this.sendMusic = sendMusic;
        this.findTrackService = findTrackService;
    }

    @GetMapping("/")
    public ResponseEntity<String> home() {
        return ResponseEntity.ok("Hello World");
    }

    @PostMapping("/upload")
    public ResponseEntity<?> upload(@RequestParam("file") MultipartFile file) throws IOException {
        

        File tempFile = File.createTempFile("music_", ".mp3");
        file.transferTo(tempFile); // MultipartFile을 임시 파일로 전송


        ResponseEntity<Results> response = sendMusic.sendToFlask(tempFile);

        tempFile.delete();

        if (response.getStatusCode().is2xxSuccessful()) {
            Results result = response.getBody();
            List<SemiTrack> similarTracks = result.getSimilarTracks();

            if (similarTracks.isEmpty()) {
                // 유사한 트랙이 없는 경우에 대한 처리
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("No similar tracks found.");
            }

            String[] trackIds = new String[similarTracks.size()];

            Map<String, Double> similarityMap = new HashMap<>();
            for (SemiTrack semiTrack : similarTracks) {
                trackIds[similarTracks.indexOf(semiTrack)] = semiTrack.getTrackId();
                similarityMap.put(semiTrack.getTrackId(), semiTrack.getSimilarity());
            }

            List<MusicTrack> tracks = findTrackService.search(trackIds);

            // ID에 따라 유사도 값을 설정
            for (MusicTrack track : tracks) {
                Double similarity = similarityMap.get(track.getTrackId());
                if (similarity != null) {
                    track.setSimilarity(similarity); // 유사도 값을 설정
                }
                track.setGenre(result.getGenre());
            }

            return ResponseEntity.ok(tracks);
        } else {
            return ResponseEntity.status(response.getStatusCode()).body("Error: " + response.getBody());
        }
    }

}
