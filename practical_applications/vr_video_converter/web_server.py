#!/usr/bin/env python3
"""
Web interface for VR video conversion.
Simple Flask server for uploading and processing videos.
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import os
import cv2
import cupy as cp
import threading
from pathlib import Path
from vr_converter import VRConverter
from gpu_video_encoder import GPUVideoEncoder

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global state
processing_status = {
    'status': 'idle',  # idle, processing, complete, error
    'progress': 0,
    'message': '',
    'current_file': None
}

# Initialize converter
vr_converter = VRConverter(use_gpu=True)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>VR Video Converter</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-box.dragover { border-color: #4CAF50; background: #f0f0f0; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .progress { width: 100%; height: 30px; background: #f0f0f0; margin: 20px 0; }
        .progress-bar { height: 100%; background: #4CAF50; transition: width 0.3s; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status.processing { background: #fff3cd; }
        .status.complete { background: #d4edda; }
        .status.error { background: #f8d7da; }
        .controls { margin: 20px 0; }
        .controls label { display: inline-block; width: 150px; }
        .controls input { width: 200px; padding: 5px; }
    </style>
</head>
<body>
    <h1>🎥 VR Video Converter</h1>
    <p>Convert 2D videos to VR 180° stereoscopic format</p>
    
    <div class="upload-box" id="dropZone">
        <p>Drag & drop video here or click to select</p>
        <input type="file" id="fileInput" accept="video/*" style="display:none">
        <button onclick="document.getElementById('fileInput').click()">Select Video</button>
    </div>
    
    <div class="controls">
        <div><label>IPD (mm):</label><input type="number" id="ipd" value="64" min="55" max="75"></div>
        <div><label>Depth Scale:</label><input type="number" id="depthScale" value="0.2" min="0" max="1" step="0.05"></div>
        <div><label>Resolution:</label>
            <select id="resolution">
                <option value="1920x960">1920×960 (Fast)</option>
                <option value="3840x1920" selected>3840×1920 (Standard)</option>
                <option value="7680x3840">7680×3840 (8K)</option>
            </select>
        </div>
        <div><label>Bitrate:</label>
            <select id="bitrate">
                <option value="5M">5 Mbps (Low)</option>
                <option value="10M" selected>10 Mbps (Standard)</option>
                <option value="20M">20 Mbps (High)</option>
            </select>
        </div>
    </div>
    
    <button id="processBtn" onclick="processVideo()" disabled>Process Video</button>
    
    <div id="statusBox" style="display:none"></div>
    
    <div id="progressBox" style="display:none">
        <div class="progress">
            <div class="progress-bar" id="progressBar" style="width:0%"></div>
        </div>
        <p id="progressText">0%</p>
    </div>
    
    <div id="downloadBox" style="display:none">
        <h3>✅ Conversion Complete!</h3>
        <button onclick="downloadVideo()">Download VR Video</button>
    </div>
    
    <script>
        let uploadedFile = null;
        let currentFilename = null;
        
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            uploadedFile = file;
            currentFilename = file.name;
            document.getElementById('processBtn').disabled = false;
            dropZone.innerHTML = `<p>✓ ${file.name}</p><p>Ready to process</p>`;
        }
        
        async function processVideo() {
            if (!uploadedFile) return;
            
            // Upload file
            const formData = new FormData();
            formData.append('video', uploadedFile);
            
            document.getElementById('processBtn').disabled = true;
            document.getElementById('statusBox').style.display = 'block';
            document.getElementById('statusBox').className = 'status processing';
            document.getElementById('statusBox').textContent = 'Uploading...';
            
            const uploadResp = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const uploadData = await uploadResp.json();
            
            if (uploadData.error) {
                showError(uploadData.error);
                return;
            }
            
            // Start processing
            const processResp = await fetch('/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    filename: uploadData.filename,
                    ipd: parseFloat(document.getElementById('ipd').value),
                    depth_scale: parseFloat(document.getElementById('depthScale').value),
                    resolution: document.getElementById('resolution').value,
                    bitrate: document.getElementById('bitrate').value
                })
            });
            
            // Poll for status
            document.getElementById('progressBox').style.display = 'block';
            pollStatus();
        }
        
        async function pollStatus() {
            const resp = await fetch('/status');
            const data = await resp.json();
            
            document.getElementById('progressBar').style.width = data.progress + '%';
            document.getElementById('progressText').textContent = 
                data.progress + '% - ' + data.message;
            document.getElementById('statusBox').textContent = data.message;
            
            if (data.status === 'complete') {
                document.getElementById('statusBox').className = 'status complete';
                document.getElementById('downloadBox').style.display = 'block';
                document.getElementById('processBtn').disabled = false;
            } else if (data.status === 'error') {
                showError(data.message);
            } else {
                setTimeout(pollStatus, 500);
            }
        }
        
        function showError(msg) {
            document.getElementById('statusBox').className = 'status error';
            document.getElementById('statusBox').textContent = 'Error: ' + msg;
            document.getElementById('processBtn').disabled = false;
        }
        
        function downloadVideo() {
            window.location.href = '/download';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    return jsonify({'filename': filename})

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    filename = data['filename']
    ipd = data.get('ipd', 64.0)
    depth_scale = data.get('depth_scale', 0.2)
    resolution = data.get('resolution', '3840x1920')
    bitrate = data.get('bitrate', '10M')
    
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_filename = f"vr_{filename}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    # Start processing in background
    thread = threading.Thread(
        target=process_video_background,
        args=(input_path, output_path, ipd, depth_scale, resolution, bitrate)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

def process_video_background(input_path, output_path, ipd, depth_scale, resolution, bitrate):
    global processing_status
    
    try:
        processing_status['status'] = 'processing'
        processing_status['progress'] = 0
        processing_status['message'] = 'Starting...'
        processing_status['current_file'] = output_path
        
        # Parse resolution
        width, height = map(int, resolution.split('x'))
        output_height = height
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        encoder = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            left, right = vr_converter.process_frame(
                frame,
                ipd_mm=ipd,
                depth_scale=depth_scale,
                output_height=output_height,
                return_gpu=True
            )
            
            # Combine
            output = VRConverter.apply_center_separation_gpu(left, right, 0.0)
            output = cp.clip(output, 0, 255).astype(cp.uint8)
            
            # Initialize encoder
            if encoder is None:
                h, w = output.shape[:2]
                encoder = GPUVideoEncoder(output_path, w, h, fps, bitrate=bitrate)
            
            # Encode
            encoder.encode_frame_gpu(output)
            
            frame_count += 1
            processing_status['progress'] = int((frame_count / total_frames) * 95)
            processing_status['message'] = f'Processing frame {frame_count}/{total_frames}'
        
        # Finalize
        if encoder:
            processing_status['message'] = 'Finalizing...'
            encoder.finalize()
        
        cap.release()
        
        processing_status['status'] = 'complete'
        processing_status['progress'] = 100
        processing_status['message'] = 'Complete!'
        
    except Exception as e:
        processing_status['status'] = 'error'
        processing_status['message'] = str(e)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

@app.route('/status')
def status():
    return jsonify(processing_status)

@app.route('/download')
def download():
    if processing_status['current_file'] and os.path.exists(processing_status['current_file']):
        return send_file(processing_status['current_file'], as_attachment=True)
    return jsonify({'error': 'No file available'}), 404

if __name__ == '__main__':
    print("="*70)
    print("VR Video Converter Web Server")
    print("="*70)
    print("Starting server on http://localhost:5000")
    print("Open your browser and navigate to the URL above")
    print("="*70)
    app.run(host='0.0.0.0', port=5000, debug=True)
