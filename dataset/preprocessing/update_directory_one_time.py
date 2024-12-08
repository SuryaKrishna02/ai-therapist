import os
import subprocess


def get_video_fps(video_file):
    """
    Return the FPS of a video or None if there's no video track.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        fps = result.stdout.strip()
        if fps:
            num, denom = fps.split('/')
            if denom == '0':
                return None
            return float(num) / float(denom)
        return None
    except Exception:
        # Any exception here means we failed to read a valid video track
        return None


def is_corrupted_clip(clip_path):
    """
    Determine if a given clip is corrupted.
    We define 'corrupted' as having no valid video track.
    """
    fps = get_video_fps(clip_path)
    return fps is None


def rename_playlists_and_videos(data_dir):
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        return

    # Get all playlist directories (old names)
    playlist_dirs = [d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))]

    # Sort them to maintain a consistent order
    playlist_dirs.sort()

    # Rename playlists to playlist_1, playlist_2, ...
    for p_idx, old_playlist_name in enumerate(playlist_dirs, start=1):
        old_playlist_path = os.path.join(data_dir, old_playlist_name)
        new_playlist_name = f"playlist_{p_idx}"
        new_playlist_path = os.path.join(data_dir, new_playlist_name)

        os.rename(old_playlist_path, new_playlist_path)
        print(f"Renamed playlist '{
              old_playlist_name}' -> '{new_playlist_name}'")

        # Get all video directories in the newly renamed playlist directory
        video_dirs = [d for d in os.listdir(new_playlist_path)
                      if os.path.isdir(os.path.join(new_playlist_path, d))]

        # Sort video directories to ensure consistent ordering
        video_dirs.sort()

        # Rename videos to video_1, video_2, ...
        for v_idx, old_video_name in enumerate(video_dirs, start=1):
            old_video_path = os.path.join(new_playlist_path, old_video_name)
            new_video_name = f"video_{v_idx}"
            new_video_path = os.path.join(new_playlist_path, new_video_name)

            os.rename(old_video_path, new_video_path)
            print(f"  Renamed video '{old_video_name}' -> '{new_video_name}'")


def remove_corrupted_clips(data_dir):
    """
    Walk through all directories under data_dir and remove any corrupted clips found.
    A clip is considered corrupted if it does not have a valid video track.
    """
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.startswith("clip_") and f.endswith(".mp4"):
                clip_path = os.path.join(root, f)
                if is_corrupted_clip(clip_path):
                    print(f"Corrupted clip found: {clip_path}. Removing...")
                    os.remove(clip_path)
                else:
                    print(f"Clip is fine: {clip_path}")


def main():
    data_dir = 'data'
    # Step 1: Rename directories
    rename_playlists_and_videos(data_dir)

    # Step 2: Check and remove corrupted clips
    remove_corrupted_clips(data_dir)

    print("\nAll directories successfully renamed, and corrupted clips removed if any were found.")


if __name__ == "__main__":
    main()
