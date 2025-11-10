document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const filePreview = document.getElementById('file-preview');
    const fileName = document.getElementById('file-name');
    const fileDetails = document.getElementById('file-details');
    const startProcessing = document.getElementById('start-processing');
    const processingSection = document.getElementById('processing-section');
    const resultsSection = document.getElementById('results-section');
    const progressFill = document.getElementById('progress-fill');
    const progressPercent = document.getElementById('progress-percent');
    const currentStage = document.getElementById('current-stage');
    const stagesList = document.getElementById('stages-list');
    const elapsedTime = document.getElementById('elapsed-time');
    const downloadVideo = document.getElementById('download-video');
    const clearAll = document.getElementById('clear-all');
    const originalVideo = document.getElementById('original-video');
    const processedVideoContainer = document.getElementById('processed-video-container');
    const processedVideoPlaceholder = document.getElementById('processed-video-placeholder');
    const team1Bar = document.getElementById('team1-bar');
    const team1Percent = document.getElementById('team1-percent');
    const team2Bar = document.getElementById('team2-bar');
    const team2Percent = document.getElementById('team2-percent');
    const totalPlayers = document.getElementById('total-players');
    const avgSpeed = document.getElementById('avg-speed');
    const totalDistance = document.getElementById('total-distance');
    const processingTime = document.getElementById('processing-time');
    const videoDuration = document.getElementById('video-duration');
    const detectionAccuracy = document.getElementById('detection-accuracy');
    const sampleButton = document.getElementById('sample-button');

    let file;
    let processingInterval;
    let elapsedSeconds = 0;

    const stages = [
        "Initializing...",
        "Uploading video...",
        "Analyzing frames...",
        "Detecting players and ball...",
        "Assigning teams...",
        "Tracking ",
        "movement...",
        "Calculating speed and distance...",
        "Generating analytics...",
        "Finalizing video..."
    ];

    function handleFiles(files) {
        if (files.length === 0) return;
        file = files[0];

        if (file.size > 500 * 1024 * 1024) {
            alert("File is too large. Maximum size is 500MB.");
            return;
        }

        fileName.textContent = file.name;
        fileDetails.textContent = `Size: ${(file.size / 1024 / 1024).toFixed(2)} MB | Type: ${file.type}`;
        
        uploadArea.classList.add('hidden');
        filePreview.classList.remove('hidden');
        startProcessing.disabled = false;

        const reader = new FileReader();
        reader.onload = (e) => {
            originalVideo.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    uploadButton.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => handleFiles(fileInput.files));

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('highlight'), false);
    });

    uploadArea.addEventListener('drop', (e) => {
        handleFiles(e.dataTransfer.files);
    });

    startProcessing.addEventListener('click', () => {
        if (!file) return;
        startProcessingFlow();
    });

    sampleButton.addEventListener('click', () => {
        startProcessingFlow(true);
    });

    function startProcessingFlow(isSample = false) {
        filePreview.classList.add('hidden');
        uploadArea.classList.add('hidden');
        processingSection.classList.remove('hidden');
        
        elapsedSeconds = 0;
        elapsedTime.textContent = '0:00';
        processingInterval = setInterval(() => {
            elapsedSeconds++;
            const minutes = Math.floor(elapsedSeconds / 60);
            const seconds = elapsedSeconds % 60;
            elapsedTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);

        if (isSample) {
            // Simulate processing for a sample video
            mockProcessing();
        } else {
            uploadAndProcess();
        }
    }

    function uploadAndProcess() {
        const formData = new FormData();
        formData.append('video', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload', true);

        let stageIndex = 0;
        updateStage(stageIndex);

        xhr.upload.onprogress = (event) => {
            if (event.lengthComputable) {
                const percentComplete = (event.loaded / event.total) * 100;
                if (stageIndex === 1) { // Uploading stage
                    updateProgress(percentComplete, stageIndex);
                }
            }
        };
        
        xhr.onloadstart = () => {
            stageIndex = 1; // Move to uploading stage
            updateStage(stageIndex);
        };

        xhr.onload = () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                mockProcessing(stageIndex + 1, response.processed_video_url, response.analytics);
            } else {
                alert('Processing failed. Please try again.');
                resetToInitialState();
            }
        };

        xhr.onerror = () => {
            alert('An error occurred during the upload. Please try again.');
            resetToInitialState();
        };

        xhr.send(formData);
    }

    function mockProcessing(startStage = 0, processedVideoUrl, analyticsData) {
        let currentProgress = 0;
        let stageIndex = startStage;

        updateStage(stageIndex);

        const interval = setInterval(() => {
            currentProgress += Math.random() * 5;
            if (currentProgress > 100) {
                currentProgress = 100;
            }
            
            updateProgress(currentProgress, stageIndex);

            if (currentProgress === 100) {
                stageIndex++;
                if (stageIndex >= stages.length) {
                    clearInterval(interval);
                    clearInterval(processingInterval);
                    displayResults(processedVideoUrl, analyticsData);
                } else {
                    currentProgress = 0;
                    updateStage(stageIndex);
                }
            }
        }, 200 + Math.random() * 300);
    }

    function updateStage(index) {
        currentStage.textContent = stages[index];
        
        stagesList.innerHTML = '';
        for (let i = 0; i < stages.length; i++) {
            const stageEl = document.createElement('div');
            stageEl.className = 'stage';
            if (i < index) {
                stageEl.classList.add('completed');
                stageEl.innerHTML = `✓ <span>${stages[i]}</span>`;
            } else if (i === index) {
                stageEl.classList.add('active');
                stageEl.innerHTML = `» <span>${stages[i]}</span>`;
            } else {
                stageEl.innerHTML = `&nbsp;&nbsp;<span>${stages[i]}</span>`;
            }
            stagesList.appendChild(stageEl);
        }
    }

    function updateProgress(percent, stageIndex) {
        const overallPercent = ((stageIndex / stages.length) + (percent / 100 / stages.length)) * 100;
        progressFill.style.width = `${overallPercent}%`;
        progressPercent.textContent = `${Math.round(overallPercent)}%`;
    }

    function displayResults(processedVideoUrl, analytics) {
        processingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        if (processedVideoUrl) {
            const videoElement = document.createElement('video');
            videoElement.controls = true;
            videoElement.src = processedVideoUrl;
            processedVideoPlaceholder.replaceWith(videoElement);
            processedVideoContainer.replaceChild(videoElement, processedVideoContainer.firstChild);
            processedVideoContainer.insertBefore(document.createElement('h4'), videoElement).textContent = 'Processed Video';
            downloadVideo.disabled = false;
            downloadVideo.onclick = () => window.open(processedVideoUrl, '_blank');
        }

        if (analytics) {
            team1Bar.style.width = `${analytics.team_possession.team1}%`;
            team1Percent.textContent = `${analytics.team_possession.team1}%`;
            team2Bar.style.width = `${analytics.team_possession.team2}%`;
            team2Percent.textContent = `${analytics.team_possession.team2}%`;
            totalPlayers.textContent = analytics.key_metrics.total_players;
            avgSpeed.textContent = `${analytics.key_metrics.avg_speed} km/h`;
            totalDistance.textContent = `${analytics.key_metrics.total_distance} m`;
            processingTime.textContent = `${analytics.key_metrics.processing_time}s`;
            videoDuration.textContent = analytics.key_metrics.video_duration;
            detectionAccuracy.textContent = `${analytics.key_metrics.detection_accuracy}%`;
        }
    }

    clearAll.addEventListener('click', resetToInitialState);

    function resetToInitialState() {
        file = null;
        clearInterval(processingInterval);
        
        resultsSection.classList.add('hidden');
        processingSection.classList.add('hidden');
        filePreview.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        
        startProcessing.disabled = true;
        downloadVideo.disabled = true;
        
        progressFill.style.width = '0%';
        progressPercent.textContent = '0%';
        
        originalVideo.src = '';
        
        const currentProcessedVideo = processedVideoContainer.querySelector('video');
        if (currentProcessedVideo) {
            currentProcessedVideo.replaceWith(processedVideoPlaceholder);
        }

        // Reset analytics
        team1Bar.style.width = '0%';
        team1Percent.textContent = '0%';
        team2Bar.style.width = '0%';
        team2Percent.textContent = '0%';
        totalPlayers.textContent = '0';
        avgSpeed.textContent = '0 km/h';
        totalDistance.textContent = '0 m';
        processingTime.textContent = '0s';
        videoDuration.textContent = '0:00';
        detectionAccuracy.textContent = '0%';
    }
});
