package main.java_server.service;

import main.java_server.model.MusicTrack;
import main.java_server.spotify.CreateToken;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class FindTrackService {

    private final CreateToken createToken;

    public FindTrackService(CreateToken createToken) {
        this.createToken = createToken;
    }

    public List<MusicTrack> search(List<String> trackIds) {

    }

}
