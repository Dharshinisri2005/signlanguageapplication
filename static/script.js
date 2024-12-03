const socket = io.connect('http://localhost:5000');
const localVideo = document.getElementById('localVideo');
const actionDisplay = document.getElementById('action');

let localStream;

// Initialize video stream
async function startVideo() {
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    localVideo.srcObject = localStream;

    // Start sending frames to the server
    setInterval(sendFrame, 100); // Send a frame every 100ms
}

function sendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = 640; canvas.height = 480;
    const context = canvas.getContext('2d');
    context.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64data = reader.result;
            socket.emit('video_frame', { frame: base64data });
        };
        reader.readAsDataURL(blob);
    });
}

// Handle server predictions
socket.on('prediction', (data) => {
    console.log("Prediction received:", data.action);
    actionDisplay.innerText = data.action;
});

// Start the video and WebRTC connection
startVideo();
