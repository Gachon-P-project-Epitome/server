package main.java_server.service;

import main.java_server.Dto.Results;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.io.File;

@Service
public class SendMusic {

    private final RestTemplate restTemplate;
    String flaskUrl = "http://localhost:5001/process_music";
    public SendMusic(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public ResponseEntity<Results> sendToFlask(File file){
        FileSystemResource fileResource = new FileSystemResource(file);
        // 파일을 Multipart 요청으로 설정
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);
        // 헤더 생성
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        // 헤더와 본문 결합
        HttpEntity<MultiValueMap<String,Object>> requestEntity = new HttpEntity<>(body, headers);
        ResponseEntity<Results> responseEntity = restTemplate.postForEntity(flaskUrl, requestEntity, Results.class);
        return responseEntity;
    }


}
