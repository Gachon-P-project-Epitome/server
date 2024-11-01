package main.java_server.service;

import com.wrapper.spotify.SpotifyApi;
import com.wrapper.spotify.requests.data.tracks.GetSeveralTracksRequest;
import com.wrapper.spotify.model_objects.specification.Track;
import main.java_server.model.MusicTrack;
import main.java_server.spotify.CreateToken;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
public class FindTrackService {

    private final CreateToken createToken;

    public FindTrackService(CreateToken createToken) {
        this.createToken = createToken;
    }

    public List<MusicTrack> search(String[] trackIds) {
        String accessToken = createToken.accesstoken();
        if (accessToken == null) {
            System.out.println("Failed to get access token.");
            return Collections.emptyList();
        }

        List<MusicTrack> validTracks = new ArrayList<>();

        // Spotify API 설정
        SpotifyApi spotifyApi = new SpotifyApi.Builder()
                .setAccessToken(accessToken)
                .build();

        GetSeveralTracksRequest getSeveralTracksRequest = spotifyApi.getSeveralTracks(trackIds)
                .build();

        try {
            Track[] tracks = getSeveralTracksRequest.execute();

            for (Track track : tracks) {
                MusicTrack musicTrack = new MusicTrack();

                String previewUrl = track.getPreviewUrl();
                String albumImageUrl = track.getAlbum().getImages() != null && track.getAlbum().getImages().length > 0
                        ? track.getAlbum().getImages()[0].getUrl() : null;

                // 둘 다 유효한 경우에만 추가
                if (previewUrl != null && albumImageUrl != null) {
                    musicTrack.setTrackId(track.getId());
                    musicTrack.setName(track.getName());
                    musicTrack.setArtist(track.getArtists().length > 0 ? track.getArtists()[0].getName() : "Unknown Artist");
                    musicTrack.setAlbum(track.getAlbum().getName());
                    musicTrack.setAlbumImageUrl(albumImageUrl);
                    musicTrack.setPreviewUrl(previewUrl);
                    musicTrack.setSimilarity(0.0); // 초기 유사도 값 설정

                    // 리스트에 추가
                    validTracks.add(musicTrack);
                } else {
                    System.out.println("Track skipped due to missing preview or album image URL: " + track.getName());
                }
            }
        } catch (Exception e) {
            System.err.println("Error fetching tracks: " + e.getMessage());
            return Collections.emptyList();
        }

        // 유효한 트랙 리스트를 반환
        return validTracks;
    }
}
