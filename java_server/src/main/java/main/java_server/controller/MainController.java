package main.java_server.controller;

import main.java_server.Dto.results;
import main.java_server.service.SendMusic;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@Controller
@CrossOrigin("*")
public class MainController {

    private final SendMusic sendMusic;

    @Autowired
    public MainController(SendMusic sendMusic) {
        this.sendMusic = sendMusic;
    }

    @GetMapping("/")
    public ResponseEntity<String> home() {
        return ResponseEntity.ok("Hello World");
    }

    @PostMapping("/upload")
    public ResponseEntity<?> upload(@RequestParam("file") MultipartFile file) throws IOException {

        File tempFile = File.createTempFile("music_", ".mp3");
        file.transferTo(tempFile); // MultipartFile을 임시 파일로 전송

        ResponseEntity<results> response = sendMusic.sendToFlask(tempFile);

        tempFile.delete();


        if (response.getStatusCode().is2xxSuccessful()) {
            results result = response.getBody();

            if (result != null) {
                return ResponseEntity.ok(result);
            } else {
                return ResponseEntity.status(500).body("Error: Response body is null"); // null 처리
            }
        } else {
            return ResponseEntity.status(response.getStatusCode()).body("Error: " + response.getBody());
        }
    }

}
