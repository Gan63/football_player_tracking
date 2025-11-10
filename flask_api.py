import sys
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import json
import os
import pickle
import threading
import time
import traceback
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing modules with error handling
modules_available = {}

try:
    from utils import *
    modules_available['utils'] = True
    logger.info("Successfully imported utils")
except ImportError as e:
    logger.warning(f"Could not import utils: {e}")
    modules_available['utils'] = False

try:
    from trackers import *
    modules_available['trackers'] = True
    logger.info("Successfully imported trackers")
except ImportError as e:
    logger.warning(f"Could not import trackers: {e}")
    modules_available['trackers'] = False

try:
    from team_assigner import *
    modules_available['team_assigner'] = True
    logger.info("Successfully imported team_assigner")
except ImportError as e:
    logger.warning(f"Could not import team_assigner: {e}")
    modules_available['team_assigner'] = False

try:
    from player_ball_assigner import *
    modules_available['player_ball_assigner'] = True
    logger.info("Successfully imported player_ball_assigner")
except ImportError as e:
    logger.warning(f"Could not import player_ball_assigner: {e}")
    modules_available['player_ball_assigner'] = False

try:
    from camera_movement import *
    modules_available['camera_movement'] = True
    logger.info("Successfully imported camera_movement")
except ImportError as e:
    logger.warning(f"Could not import camera_movement: {e}")
    modules_available['camera_movement'] = False

try:
    from view_transformation import *
    modules_available['view_transformation'] = True
    logger.info("Successfully imported view_transformation")
except ImportError as e:
    logger.warning(f"Could not import view_transformation: {e}")
    modules_available['view_transformation'] = False

try:
    from speed_and_distance import SpeedAndDistance_Estimator
    modules_available['speed_and_distance'] = True
    logger.info("Successfully imported speed_and_distance")
except ImportError as e:
    logger.warning(f"Could not import speed_and_distance: {e}")
    modules_available['speed_and_distance'] = False

app = Flask(__name__, template_folder='frontend', static_folder='frontend', static_url_path='/')
CORS(app)

# Configuration with error handling
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# Create necessary directories with error handling
def create_directories():
    """Create necessary directories with proper error handling"""
    directories = [UPLOAD_FOLDER, OUTPUT_FOLDER, 'static', 'templates', 'stubs']
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory created/verified: {directory}")
        except PermissionError:
            logger.error(f"Permission denied creating directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")

create_directories()

# Flask configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

# Global variables for tracking
current_tracking_data = None
processing_status = {"status": "idle", "progress": 0, "message": "Ready"}
processing_lock = threading.Lock()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_video_file(filepath):
    """Validate video file integrity"""
    try:
        logger.info(f"Validating video file at path: {filepath}")
        logger.info(f"Absolute path: {os.path.abspath(filepath)}")
        logger.info(f"File exists: {os.path.exists(filepath)}")
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            logger.error(f"cv2.VideoCapture failed to open: {filepath}")
            return False, "Cannot open video file"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        if frame_count <= 0:
            return False, "Video has no frames"
        if fps <= 0:
            return False, "Invalid video frame rate"

        return True, f"Video validated: {frame_count} frames at {fps} FPS"

    except Exception as e:
        return False, f"Video validation error: {str(e)}"

class FootballTrackerAPI:
    """Enhanced Football Tracker API with error handling"""

    def __init__(self):
        self.tracker = None
        self.tracking_data = None
        self.video_frames = None
        self.model_path = None

    def find_model_path(self):
        """Find available YOLO model file"""
        possible_paths = [
            'models/yolov5su.pt',
            'models/yolov5lu/best.pt',
            'models/yolov5lu/best.pt',
            'models/best.pt',
            'yolov5lu/best.pt',
            'best.pt',
            'model.pt',
            'yolo.pt'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found model at: {path}")
                return path

        logger.warning("No YOLO model found in standard locations")
        return None

    def initialize_tracker(self):
        """Initialize the YOLO tracker with comprehensive error handling"""
        global processing_status

        try:
            if not modules_available.get('trackers', False):
                raise ImportError("Tracker module not available")

            if not self.model_path:
                self.model_path = self.find_model_path()

            if not self.model_path:
                processing_status = {
                    "status": "error", 
                    "progress": 0, 
                    "message": "YOLO model file not found. Please ensure model file is in models/ directory"
                }
                return False

            processing_status = {"status": "processing", "progress": 5, "message": "Loading YOLO model..."}

            # Try to initialize tracker with timeout
            self.tracker = Tracker(self.model_path)

            processing_status = {"status": "processing", "progress": 15, "message": "YOLO model loaded successfully"}
            logger.info("Tracker initialized successfully")
            return True

        except Exception as e:
            error_msg = f"Error initializing tracker: {str(e)}"
            logger.error(error_msg)
            processing_status = {"status": "error", "progress": 0, "message": error_msg}
            return False

    def safe_video_read(self, video_path):
        """Safely read video with error handling"""
        try:
            # Validate file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                raise ValueError("Video file is empty")

            logger.info(f"Reading video: {video_path} (Size: {file_size / (1024*1024):.1f} MB)")

            # Use custom video reading if utils module is available
            if modules_available.get('utils', False):
                frames = read_video_frames(video_path) # Keep as generator
            else:
                # Fallback video reading
                frames = self.fallback_read_video(video_path)

            # For generators, we need to check if it yields at least one frame.
            # We can't know the length beforehand. We'll peek at the first frame.
            try:
                first_frame = next(frames)
                # Re-create the generator including the first frame
                import itertools
                reconstructed_generator = itertools.chain([first_frame], frames)
                logger.info("Successfully started reading frames from video.")
                return reconstructed_generator
            except StopIteration:
                # This means the generator was empty
                raise ValueError("No frames could be read from the video.")

        except Exception as e:
            logger.error(f"Video reading error: {e}")
            raise e

    def fallback_read_video(self, video_path, max_frames=1000):
        """Fallback video reading method"""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Cannot open video file with OpenCV")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            frames.append(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"Read {frame_count} frames...")

        cap.release()
        return frames

    def process_video_safe(self, video_path):
        """Process video with comprehensive error handling"""
        global processing_status, current_tracking_data

        try:
            with processing_lock:
                logger.info(f"Starting video processing: {video_path}")

                # Step 1: Validate video
                processing_status = {"status": "processing", "progress": 5, "message": "Validating video file..."}
                is_valid, validation_msg = validate_video_file(video_path)
                if not is_valid:
                    raise ValueError(validation_msg)

                logger.info(validation_msg)

                # Step 2: Read video
                processing_status = {"status": "processing", "progress": 10, "message": "Reading video frames..."}
                # We will pass the path and read frame-by-frame inside the functions
                self.video_path = video_path

                # Step 3: Initialize tracker
                processing_status = {"status": "processing", "progress": 20, "message": "Initializing tracker..."}
                if not self.tracker:
                    if not self.initialize_tracker():
                        return None
                
                # Step 4: Track objects
                processing_status = {"status": "processing", "progress": 30, "message": "Tracking objects..."}
                if not self.tracker:
                    raise ValueError("Tracker not initialized")
                
                tracks = self.tracker.get_object_tracks(read_video_frames(video_path), read_from_stub=False)
                self.tracker.add_position_to_tracks(tracks)
                processing_status = {"status": "processing", "progress": 35, "message": "Interpolating ball position..."}
                tracks["ball"] = self.tracker.ball_interpolation(tracks["ball"])
                # Log the tracks object for inspection
                logger.info(f"Tracks object structure: {tracks}")

                # Initialize modules for single-pass processing
                frame_generator = read_video_frames(video_path)
                first_frame = next(frame_generator, None)
                if first_frame is None:
                    raise ValueError("Could not read the first frame of the video.")

                camera_movement = CameraMovement(first_frame) if modules_available.get('camera_movement') else None
                speed_dist = SpeedAndDistance_Estimator() if modules_available.get('speed_and_distance') else None
                team_assigner = TeamAssigner() if modules_available.get('team_assigner') else None
                player_assigner = PlayerBallAssigner() if modules_available.get('player_ball_assigner') else None

                if team_assigner and tracks.get("players"):
                    team_assigner.assign_team_color(first_frame, tracks["players"][0])

                team_ball_possession = []
                
                # Reset generator for the main loop
                total_frames = len(tracks['players'])

                for frame_num in range(total_frames):
                    progress = 30 + int(60 * (frame_num / total_frames))
                    processing_status = {"status": "processing", "progress": progress, "message": f"Processing frame {frame_num+1}/{total_frames}"}

                    # We need the frame for team assignment, so we'll read it once per loop
                    # This is less efficient than a single pass, but necessary for the current structure
                    # A better long-term solution would be to refactor to a single loop.
                    frame = next(read_video_frames(video_path, start_frame=frame_num), None)
                    if frame is None:
                        logger.warning(f"Could not read frame {frame_num}. Skipping.")
                        continue

                    if camera_movement:
                        cam_move_per_frame = camera_movement.get_camera_movement([frame], read_from_stub=False)
                        camera_movement.adjust_single_frame_tracks(tracks, frame_num, cam_move_per_frame[0])

                    player_track = tracks["players"][frame_num]
                    if team_assigner and frame is not None:
                        for player_id, track in player_track.items():
                            team = team_assigner.get_player_team(frame, track["bbox"], player_id)
                            tracks["players"][frame_num][player_id]["team"] = team
                            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team].tolist()

                    # Ball Possession
                    if player_assigner and 1 in tracks["ball"][frame_num]:
                        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
                        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
                        if assigned_player != -1:
                            player_team = tracks['players'][frame_num][assigned_player]['team']
                            tracks['players'][frame_num][assigned_player]['ball_possession'] = True
                            team_ball_possession.append(int(player_team))
                        else:
                            # No player has the ball, keep last known possession
                            team_ball_possession.append(team_ball_possession[-1] if team_ball_possession else 0)
                    else:
                        # Ball not detected, keep last known possession
                        team_ball_possession.append(team_ball_possession[-1] if team_ball_possession else 0)

                # Step 5: Process tracking results
                processing_status = {"status": "processing", "progress": 90, "message": "Finalizing analysis..."}
                if speed_dist:
                    speed_dist.add_speed_and_distance(tracks)
                
                output_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_processed.mp4"
                output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

                # Draw final annotations after all calculations are done
                processing_status = {"status": "processing", "progress": 95, "message": "Drawing annotations..."}
                output_frames = []
                frame_generator = read_video_frames(video_path)

                # Get video FPS
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                # Draw ball trace
                def draw_ball_trace(frame, ball_positions, frame_num, trace_length=20):
                    """Draws a fading trail for the ball."""
                    start_frame = max(0, frame_num - trace_length)
                    for i in range(start_frame, frame_num):
                        if 1 in ball_positions.get(i, {}) and 1 in ball_positions.get(i + 1, {}):
                            pos1 = tuple(map(int, ball_positions[i][1]['position']))
                            pos2 = tuple(map(int, ball_positions[i+1][1]['position']))
                            cv2.line(frame, pos1, pos2, (255, 255, 0), 2)

                for frame_num, frame in enumerate(frame_generator):
                    annotated_frame = self.tracker.draw_annotations([frame], tracks, team_ball_possession, specific_frame_num=frame_num)[0]
                    if speed_dist:
                        annotated_frame = speed_dist.draw_speed_and_distance([annotated_frame], tracks, frame_num)[0]
                    output_frames.append(annotated_frame)
                save_video(output_frames, output_filepath, fps)


                tracking_data = self.convert_tracks_to_json(tracks, team_ball_possession, output_filename, fps)

                # Step 6: Finalize
                current_tracking_data = tracking_data
                processing_status = {"status": "completed", "progress": 100, "message": "Processing completed successfully"}

                logger.info("Video processing completed successfully")
                return tracking_data

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            processing_status = {"status": "error", "progress": 0, "message": error_msg}
            return None

    def safe_track_objects(self, video_path):
        """Safely track objects with error handling"""
        try:
            if not modules_available.get('trackers', False):
                raise ImportError("Trackers module not available")

            # Get tracks with error handling
            tracks = self.tracker.get_object_tracks(
                read_video_frames(video_path), # Pass a new generator
                read_from_stub=False,
                stub_path='stubs/track_stubs.pkl'
            )

            processing_status = {"status": "processing", "progress": 40, "message": "Adding position data..."}

            # Add positions to tracks
            self.tracker.add_position_to_tracks(tracks)

            return tracks

        except Exception as e:
            logger.error(f"Object tracking error: {e}")
            # Can't generate fallback with frames, so return empty
            return {'players': [], 'ball': [], 'referees': []}

    def convert_tracks_to_json(self, tracks, team_ball_control, output_filename, fps=30.0):
        """Convert tracking data to JSON format with error handling"""
        try:
            tracking_data = []
            team_stats = {1: {"possession_time": 0, "players": []}, 2: {"possession_time": 0, "players": []}}

            total_frames = len(tracks.get('players', []))

            for frame_num in range(total_frames):
                frame_data = {
                    "frame": frame_num,
                    "timestamp": round(frame_num / fps, 2),
                    "players": {},
                    "ball": {}
                }

                # Add player data with error handling
                if frame_num < len(tracks.get('players', [])):
                    for player_id, player_data in tracks['players'][frame_num].items():
                        try:
                            frame_data["players"][str(player_id)] = {
                                "name": f"Player {str(player_id)}",
                                "position": [
                                    float(player_data.get('position', [0, 0])[0]),
                                    float(player_data.get('position', [0, 0])[1])
                                ],
                                "bbox": [float(x) for x in player_data.get('bbox', [0, 0, 50, 100])],
                                "team": int(player_data.get('team', 1)) if player_data.get('team') is not None else 1,
                                "team_color": [int(x) for x in player_data.get('team_color', [255, 0, 0])],
                                "speed": float(player_data.get('speed', 0)) if player_data.get('speed') is not None else 0,
                                "distance_covered": float(player_data.get('distance', 0)) if player_data.get('distance') is not None else 0,
                                "ball_possession": bool(player_data.get('ball_possession', False))
                            }
                        except (KeyError, IndexError, TypeError, ValueError) as e:
                            logger.warning(f"Error processing player {player_id} data: {type(e).__name__}: {e}")

                # Add ball data with error handling
                try:
                    if frame_num < len(tracks.get('ball', [])) and tracks['ball'][frame_num]:
                        ball_data = list(tracks['ball'][frame_num].values())[0]
                        frame_data["ball"] = {
                            "position": [
                                float(ball_data.get('position', [320, 240])[0]),
                                float(ball_data.get('position', [320, 240])[1])
                            ]
                        }
                    else:
                        frame_data["ball"] = {"position": [320, 240]}
                except:
                    frame_data["ball"] = {"position": [320, 240]}

                tracking_data.append(frame_data)

            # Calculate team statistics with error handling
            try:
                if team_ball_control:
                    team1_possession = sum(1 for team in team_ball_control if team == 1)
                    team2_possession = sum(1 for team in team_ball_control if team == 2)

                    team_stats[1]["possession_time"] = round((team1_possession / len(team_ball_control)) * 100, 1)
                    team_stats[2]["possession_time"] = round((team2_possession / len(team_ball_control)) * 100, 1)
                else:
                    team_stats[1]["possession_time"] = 50.0
                    team_stats[2]["possession_time"] = 50.0

                player_ids = set(pid for frame in tracking_data for pid in frame["players"].keys())
                for player_id in player_ids:
                    player_speeds = [frame["players"][player_id]["speed"] for frame in tracking_data if player_id in frame["players"]]
                    player_team = [frame["players"][player_id]["team"] for frame in tracking_data if player_id in frame["players"]][0]
                    player_name = [frame["players"][player_id]["name"] for frame in tracking_data if player_id in frame["players"]][0]
                    if player_speeds:
                        avg_speed = round(sum(player_speeds) / len(player_speeds), 1)
                        max_speed = round(max(player_speeds), 1)
                    else:
                        avg_speed = 0
                        max_speed = 0

                    # Correctly get the final total distance for the player
                    total_distance = 0
                    # Find the last frame this player appears in to get their final distance
                    for frame in reversed(tracking_data):
                        if player_id in frame["players"]:
                            total_distance = round(frame["players"][player_id]["distance_covered"], 1)
                            break

                    player_stats = {
                        "id": player_id,
                        "name": player_name,
                        "avg_speed": avg_speed,
                        "max_speed": max_speed,
                        "total_distance": total_distance
                    }
                    team_stats[player_team]["players"].append(player_stats)

            except:
                team_stats[1]["possession_time"] = 50.0
                team_stats[2]["possession_time"] = 50.0

            return {
                "tracking_data": tracking_data,
                "team_statistics": {
                    1: {"name": "Team Red", "color": "#FF4444", **team_stats[1]},
                    2: {"name": "Team Blue", "color": "#4444FF", **team_stats[2]}
                },
                "possession_history": team_ball_control,
                "match_info": {
                    "duration": f"{len(tracking_data)} frames ({len(tracking_data)/fps:.2f}s at {fps:.2f}fps)" if tracking_data else "0 frames (0.00s at 30fps)",
                    "total_players": len(set(pid for frame in tracking_data for pid in frame["players"].keys())),
                    "teams": 2,
                    "field_dimensions": {"width": 640, "height": 480}
                },
                "processed_video_url": f"/download/{output_filename}"
            }

        except Exception as e:
            logger.error(f"Error converting tracks to JSON: {e}")
            return self.generate_basic_tracking_data()

    def generate_basic_tracking_data(self):
        """Generate basic tracking data as fallback"""
        return {
            "tracking_data": [],
            "team_statistics": {
                1: {"name": "Team Red", "color": "#FF4444", "possession_time": 50.0, "players": []},
                2: {"name": "Team Blue", "color": "#4444FF", "possession_time": 50.0, "players": []}
            },
            "possession_history": [],
            "match_info": {
                "duration": "0 frames (0.00s at 30fps)",
                "total_players": 0,
                "teams": 2,
                "field_dimensions": {"width": 640, "height": 480}
            }
        }

# Initialize tracker API
tracker_api = FootballTrackerAPI()

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors"""
    return jsonify({"error": "File too large. Maximum size is 500MB."} ), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error. Please check the logs."}), 500

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return f"Error loading page: {e}", 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    """Handle video upload and processing with comprehensive error handling"""
    if request.method == 'POST':
        logger.info("Received POST request to /upload")
        try:
            # Check if file is present
            if 'video' not in request.files:
                logger.warning("No 'video' part in request.files")
                return jsonify({"error": "No video file provided"}), 400

            file = request.files['video']
            logger.info(f"Received file: {file.filename}")

            if file.filename == '':
                logger.warning("Received an empty filename")
                return jsonify({"error": "No file selected"}), 400

            # Validate file type
            if not allowed_file(file.filename):
                logger.warning(f"File type not allowed: {file.filename}")
                return jsonify({
                    "error": f"Invalid file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400

            # Secure filename
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to: {filepath}")

            # Save file
            file.save(filepath)
            logger.info("File saved.")

            # Validate saved file
            if not os.path.exists(filepath):
                logger.error("File not found after saving.")
                return jsonify({"error": "Failed to save uploaded file"}), 500

            file_size = os.path.getsize(filepath)
            logger.info(f"File saved successfully: {filename} ({file_size / (1024*1024):.1f} MB)")

            # Start processing in background thread
            logger.info("Starting processing thread.")
            processing_thread = threading.Thread(
                target=tracker_api.process_video_safe,
                args=(filepath,),
                daemon=True
            )
            processing_thread.start()
            logger.info("Processing thread started.")

            return jsonify({
                "message": "Video uploaded successfully, processing started",
                "filename": filename,
                "size_mb": round(file_size / (1024*1024), 1)
            })

        except Exception as e:
            logger.error(f"Upload error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Upload failed: {str(e)}"}), 500

    logger.info("Received GET request to /upload")
    try:
        return render_template('upload.html')
    except Exception as e:
        logger.error(f"Error rendering upload page: {e}")
        return f"Error loading upload page: {e}", 500

@app.route('/api/status')
def get_status():
    """Get current processing status"""
    try:
        # Add module availability info
        status_info = processing_status.copy()
        status_info['modules_available'] = modules_available
        status_info['server_time'] = datetime.now().isoformat()
        return jsonify(status_info)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": "Failed to get status"}), 500

@app.route('/api/tracking-data')
def get_tracking_data():
    """Get current tracking data"""
    try:
        global current_tracking_data
        if current_tracking_data:
            return jsonify(current_tracking_data)
        else:
            return jsonify({"error": "No tracking data available"}), 404
    except Exception as e:
        logger.error(f"Error getting tracking data: {e}")
        return jsonify({"error": "Failed to get tracking data"}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Serve processed video for download"""
    try:
        # Security: Ensure the filename is safe and points to the output directory
        safe_filename = secure_filename(filename)
        if ".." in safe_filename or "/" in safe_filename:
            return jsonify({"error": "Invalid filename"}), 400

        logger.info(f"Download requested for: {safe_filename}")
        return send_from_directory(
            app.config['OUTPUT_FOLDER'], safe_filename,
            as_attachment=True
        )
    except FileNotFoundError:
        logger.error(f"Download failed: File not found - {filename}")
        return jsonify({"error": "File not found"}), 404

@app.route('/api/process-sample', methods=['GET', 'POST'])
def process_sample():
    """Process sample video or generate sample data for demo"""
    global current_tracking_data, processing_status

    try:
        # Check for sample video files
        sample_videos = [
            'input_videos/video4.mp4',
            'sample_video.mp4', 
            'demo.mp4',
            'test.mp4'
        ]

        sample_path = None
        for path in sample_videos:
            if os.path.exists(path):
                sample_path = path
                break

        if sample_path:
            # Process sample video in background
            processing_thread = threading.Thread(
                target=tracker_api.process_video_safe,
                args=(sample_path,),
                daemon=True
            )
            processing_thread.start()

            return jsonify({
                "message": f"Sample video processing started: {sample_path}"
            })
        else:
            # Generate sample data
            logger.info("No sample video found, generating sample data...")
            processing_status = {
                "status": "processing", 
                "progress": 50, 
                "message": "Creating sample tracking data..."
            }

            sample_data = generate_sample_tracking_data()
            current_tracking_data = sample_data

            processing_status = {
                "status": "completed", 
                "progress": 100, 
                "message": "Sample data ready"
            }

            return jsonify({
                "message": "Sample tracking data generated successfully"
            })

    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        processing_status = {
            "status": "error", 
            "progress": 0, 
            "message": f"Error: {str(e)}"
        }
        return jsonify({"error": str(e)}), 500

@app.route('/api/system-info')
def system_info():
    """Get system information for debugging"""
    try:
        import platform
        import sys

        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "opencv_available": True,
            "modules_status": modules_available,
            "directories": {
                "upload_folder": os.path.exists(UPLOAD_FOLDER),
                "output_folder": os.path.exists(OUTPUT_FOLDER),
                "templates": os.path.exists('templates'),
                "static": os.path.exists('static')
            }
        }

        # Check OpenCV
        try:
            import cv2
            info["opencv_version"] = cv2.__version__
        except:
            info["opencv_available"] = False

        # Check for model files
        model_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.pt'):
                    model_files.append(os.path.join(root, file))
        info["model_files"] = model_files

        return jsonify(info)

    except Exception as e:
        return jsonify({"error": f"Failed to get system info: {str(e)}"}), 500

def generate_sample_tracking_data():
    """Generate comprehensive sample tracking data for demo"""
    import random

    tracking_data = []
    team_ball_control = []

    # Sample players with more realistic data
    players = {
        "player_1": {"team": 1, "pos": [140, 210], "color": [255, 68, 68], "name": "John Doe"},
        "player_2": {"team": 2, "pos": [242, 242], "color": [68, 68, 255], "name": "Jane Smith"},
        "player_3": {"team": 1, "pos": [339, 159], "color": [255, 68, 68], "name": "Mike Johnson"},
        "player_4": {"team": 2, "pos": [491, 311], "color": [68, 68, 255], "name": "Sarah Wilson"},
        "player_5": {"team": 1, "pos": [180, 350], "color": [255, 68, 68], "name": "Tom Brown"},
        "player_6": {"team": 2, "pos": [520, 180], "color": [68, 68, 255], "name": "Lisa Davis"}
    }

    ball_pos = [327, 247]

    for frame in range(150):  # 150 frames = 5 seconds at 30fps
        frame_data = {
            "frame": frame,
            "timestamp": round(frame / 30.0, 2),
            "players": {},
            "ball": {"position": ball_pos.copy()}
        }

        # Update player positions with more realistic movement
        for player_id, player_info in players.items():
            # More realistic movement patterns
            if frame % 60 < 30:  # First half of cycle - move towards ball
                dx = (ball_pos[0] - player_info["pos"][0]) * 0.02 + random.uniform(-1, 1)
                dy = (ball_pos[1] - player_info["pos"][1]) * 0.02 + random.uniform(-1, 1)
            else:  # Second half - more random movement
                dx = random.uniform(-2, 2)
                dy = random.uniform(-2, 2)

            # Keep players on field
            player_info["pos"][0] = max(30, min(610, player_info["pos"][0] + dx))
            player_info["pos"][1] = max(30, min(450, player_info["pos"][1] + dy))

            # Calculate realistic speeds
            if frame > 0:
                prev_frame = tracking_data[frame-1]["players"].get(player_id)
                if prev_frame:
                    dist = ((player_info["pos"][0] - prev_frame["position"][0])**2 + 
                           (player_info["pos"][1] - prev_frame["position"][1])**2)**0.5
                    speed = (dist * 30) / 100  # Convert to m/s approximation
                else:
                    speed = 0
            else:
                speed = 0

            # Ball possession logic
            ball_distance = ((player_info["pos"][0] - ball_pos[0])**2 + 
                           (player_info["pos"][1] - ball_pos[1])**2)**0.5

            frame_data["players"][player_id] = {
                "name": player_info["name"],
                "position": [round(player_info["pos"][0], 1), round(player_info["pos"][1], 1)],
                "bbox": [
                    round(player_info["pos"][0]-25, 1), 
                    round(player_info["pos"][1]-50, 1), 
                    50, 100
                ],
                "team": player_info["team"],
                "team_color": player_info["color"],
                "speed": round(speed, 1),
                "distance_covered": round(frame * 0.05, 1),
                "ball_possession": ball_distance < 25,
                "ball_distance": round(ball_distance, 1)
            }

        # Find closest player to ball
        closest_player = None
        min_distance = float('inf')
        for player_id, player_data in frame_data["players"].items():
            if player_data["ball_distance"] < min_distance:
                min_distance = player_data["ball_distance"]
                closest_player = player_id

        # Update ball possession
        for player_id in frame_data["players"]:
            frame_data["players"][player_id]["ball_possession"] = (player_id == closest_player and min_distance < 25)

        if closest_player and min_distance < 25:
            team_ball_control.append(frame_data["players"][closest_player]["team"])
            # Move ball towards player with possession
            target_pos = frame_data["players"][closest_player]["position"]
            ball_pos[0] += (target_pos[0] - ball_pos[0]) * 0.3
            ball_pos[1] += (target_pos[1] - ball_pos[1]) * 0.3
        else:
            # Ball moves freely
            ball_pos[0] += random.uniform(-1, 1)
            ball_pos[1] += random.uniform(-1, 1)
            ball_pos[0] = max(20, min(620, ball_pos[0]))
            ball_pos[1] = max(20, min(460, ball_pos[1]))
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        frame_data["ball"]["position"] = [round(ball_pos[0], 1), round(ball_pos[1], 1)]
        tracking_data.append(frame_data)

    # Calculate comprehensive team statistics
    team1_possession = sum(1 for team in team_ball_control if team == 1)
    team2_possession = sum(1 for team in team_ball_control if team == 2)
    total_frames = len(team_ball_control)

    # Calculate player statistics
    team1_players = []
    team2_players = []

    for player_id, player_info in players.items():
        player_speeds = [frame["players"][player_id]["speed"] for frame in tracking_data]
        avg_speed = sum(player_speeds) / len(player_speeds) if player_speeds else 0
        max_speed = max(player_speeds) if player_speeds else 0
        total_distance = tracking_data[-1]["players"][player_id]["distance_covered"]

        player_stat = {
            "id": player_id,
            "name": player_info["name"],
            "avg_speed": round(avg_speed, 1),
            "max_speed": round(max_speed, 1),
            "total_distance": round(total_distance, 1)
        }

        if player_info["team"] == 1:
            team1_players.append(player_stat)
        else:
            team2_players.append(player_stat)

    return {
        "tracking_data": tracking_data,
        "team_statistics": {
            1: {
                "name": "Team Red",
                "color": "#FF4444",
                "possession_time": round((team1_possession / total_frames) * 100, 1),
                "players": team1_players,
                "avg_speed": round(sum([p["avg_speed"] for p in team1_players]) / len(team1_players), 1) if team1_players else 0,
                "total_distance": round(sum([p["total_distance"] for p in team1_players]), 1)
            },
            2: {
                "name": "Team Blue",
                "color": "#4444FF", 
                "possession_time": round((team2_possession / total_frames) * 100, 1),
                "players": team2_players,
                "avg_speed": round(sum([p["avg_speed"] for p in team2_players]) / len(team2_players), 1) if team2_players else 0,
                "total_distance": round(sum([p["total_distance"] for p in team2_players]), 1)
            }
        },
        "possession_history": team_ball_control,
        "match_info": {
            "duration": f"{len(tracking_data)} frames ({len(tracking_data)/30:.2f}s at 30fps)",
            "total_players": 6,
            "teams": 2,
            "field_dimensions": {"width": 640, "height": 480},
            "fps": 30
        }
    }

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Football Player Tracker Flask Application")
    print("=" * 60)
    print()
    print("📊 System Status:")
    print(f"   • Python: {os.sys.version.split()[0]}")
    print(f"   • OpenCV: Available")
    print(f"   • Modules Status:")
    for module, available in modules_available.items():
        status = "✅" if available else "❌"
        print(f"     - {module}: {status}")
    print()
    print("🌐 Server Information:")
    print("   • URL: http://localhost:5000")
    print("   • Upload limit: 500MB")
    print("   • Debug mode: Enabled")
    print()
    print("📁 Directories:")
    print(f"   • Upload folder: {UPLOAD_FOLDER}")
    print(f"   • Output folder: {OUTPUT_FOLDER}")
    print()
    print("🎯 Available Endpoints:")
    print("   • / : Main dashboard")
    print("   • /upload : Upload page") 
    print("   • /api/status : Processing status")
    print("   • /api/tracking-data : Get tracking results")
    print("   • /api/process-sample : Load sample data")
    print("   • /api/system-info : System information")
    print()
    print("⚠️  Note: If modules are missing, the app will use fallback methods")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    try:
        
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")