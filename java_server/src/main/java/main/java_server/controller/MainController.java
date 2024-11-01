package main.java_server.controller;

import main.java_server.Dto.results;
import main.java_server.service.SendMusic;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
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
            return ResponseEntity.ok(response.getBody()); // 성공적인 응답 반환
        } else {
            return ResponseEntity.status(response.getStatusCode()).body("Error: " + response.getBody()); // 에러 처리
        }
    }

}
