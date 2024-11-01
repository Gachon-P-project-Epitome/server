package main.java_server.service;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class sendMusic {

    private final RestTemplate restTemplate;

    public sendMusic(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

}
