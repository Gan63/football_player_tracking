from utils import *
from trackers import *
from team_assigner import *
from player_ball_assigner import *
from camera_movement import *
from view_transformation import *
from speed_and_distance import *
import numpy as np
import cv2
import os
import gc

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def process_video_optimized(input_path, output_path):
    """
    Process a football video with player tracking, team assignment, and analysis in a single pass.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path where processed video will be saved
        
    Returns:
        str: Path to the processed video file
        
    Raises:
        ValueError: If video cannot be read
        Exception: If processing fails at any stage
    """
    try:
        print("Initializing tracker...")
        tracker = Tracker('models/yolov5su.pt')

        print("Tracking objects...")
        tracks = tracker.get_object_tracks(
            read_video_frames(input_path),
            read_from_stub=False,
            stub_path='stubs/track_stubs.pkl'
        )
        tracker.add_position_to_tracks(tracks)

        print("Interpolating ball positions...")
        tracks["ball"] = tracker.ball_interpolation(tracks["ball"])

        # Initialize modules
        frame_generator = read_video_frames(input_path)
        first_frame = next(frame_generator)

        print("Initializing Camera Movement detector...")
        camera_movement = CameraMovement(first_frame)

        print("Initializing Speed and Distance estimator...")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        speed_dist = SpeedAndDistance_Estimator(frame_rate=fps)

        print("Initializing Team Assigner...")
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(first_frame, tracks["players"][0])

        print("Initializing Player-Ball Assigner...")
        player_assigner = PlayerBallAssigner()
        team_ball_possession = []
        output_frames = []

        # Reset generator to process from the beginning
        frame_generator = read_video_frames(input_path)

        print("Processing frames in a single pass...")
        for frame_num, frame in enumerate(frame_generator):
            # Camera Movement
            camera_movement_per_frame = camera_movement.get_camera_movement([frame], read_from_stub=False)
            camera_movement.adjust_single_frame_tracks(tracks, frame_num, camera_movement_per_frame[0])

            # Team Assignment
            player_track = tracks["players"][frame_num]
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(frame, track["bbox"], player_id)
                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team].tolist()

            # Ball Possession
            if 1 in tracks["ball"][frame_num]:
                ball_bbox = tracks["ball"][frame_num][1]["bbox"]
                closest_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
                if closest_player != -1:
                    tracks["players"][frame_num][closest_player]["ball_possession"] = True
                    team_ball_possession.append(tracks["players"][frame_num][closest_player]["team"])
                else:
                    team_ball_possession.append(team_ball_possession[-1] if team_ball_possession else 0)
            else:
                team_ball_possession.append(team_ball_possession[-1] if team_ball_possession else 0)
        
        # Speed and Distance calculation (should be done once after all positions are adjusted)
        print("Calculating speed and distance...")
        speed_dist.add_speed_and_distance(tracks)

        # Drawing final annotations
        print("Drawing final annotations...")
        output_frames = []
        frame_generator = read_video_frames(input_path) # Reset generator for drawing
        team_ball_possession_np = np.array(team_ball_possession)
        for frame_num, frame in enumerate(frame_generator):
            annotated_frame = tracker.draw_annotations([frame], tracks, team_ball_possession_np, frame_num)[0]
            annotated_frame = speed_dist.draw_speed_and_distance([annotated_frame], tracks, frame_num)[0]
            output_frames.append(annotated_frame)

        print("Saving output video...")
        save_video(output_frames, output_path, fps)
        
        print("Done!")
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        
        raise

    finally:
        # Final cleanup
        gc.collect()

# Only run this if the script is executed directly (not imported)
if __name__ == '__main__':
    # Example usage - uncomment to test directly
    input_file = "input_videos/video2.mp4"
    output_file = f"output/{os.path.splitext(os.path.basename(input_file))[0]}_processed.mp4"
    process_video_optimized(input_file, output_file)