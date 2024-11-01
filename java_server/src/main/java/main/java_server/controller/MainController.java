package main.java_server.controller;

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

    private final RestTemplate restTemplate;

    public MainController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @GetMapping("/")
    public ResponseEntity<String> home() {
        return ResponseEntity.ok("Hello World");
    }

    @PostMapping("/upload")
    public ResponseEntity<?> upload(@RequestParam("file") MultipartFile file) throws IOException {

        File tempFile = File.createTempFile("music_", ".mp3");
        file.transferTo(tempFile); // MultipartFile을 임시 파일로 전송

        // FileSystemResource 생성
        FileSystemResource fileResource = new FileSystemResource(tempFile);

        // 파일을 Multipart 요청으로 설정
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);

        String flaskUrl = "http://localhost:5001/process_music";

        // 헤더 생성
        HttpHeaders headers = new HttpHeaders();

        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        // 헤더와 본문 결합
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        ResponseEntity<String> response = restTemplate.postForEntity(flaskUrl, requestEntity, String.class);

        tempFile.delete();

        if (response.getStatusCode().is2xxSuccessful()) {
            return ResponseEntity.ok(response.getBody()); // 성공적인 응답 반환
        } else {
            return ResponseEntity.status(response.getStatusCode()).body("Error: " + response.getBody()); // 에러 처리
        }
    }

}
