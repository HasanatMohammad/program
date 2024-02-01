const ATS = ()=> {
    // Check if browser supports the SpeechRecognition API
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        var startButton = document.getElementById('startButton');
        var stopButton = document.getElementById('stopButton');

        recognition.onstart = function () {
            console.log('Speech recognition started');
            startButton.disabled = true;
            stopButton.disabled = false;
        };

        recognition.onresult = function (event) {
            var result = event.results[0][0].transcript;
            document.getElementById('output').innerText = 'You said: ' + result;

            // Send the recognized text to the server
            document.getElementById('textInput').value = result;
            document.getElementById('textForm').submit();
        };

        recognition.onend = function () {
            console.log('Speech recognition ended');
            startButton.disabled = false;
            stopButton.disabled = true;
        };

        recognition.onerror = function (event) {
            console.error('Speech recognition error:', event.error);
            startButton.disabled = false;
            stopButton.disabled = true;
        };

        // Start speech recognition on start button click
        startButton.addEventListener('click', function () {
           
        });

        // Stop speech recognition on stop button click
        stopButton.addEventListener('click', function () {
            recognition.stop();
        });

        // // Process the form submission
        // document.getElementById('textForm').addEventListener('submit', function (event) {
        //     event.preventDefault(); // Prevent the form from submitting normally

        //     fetch('/process_text', {
        //         method: 'POST',
        //         body: new FormData(document.getElementById('textForm'))
        //     })
        //     .then(response => response.json())
        //     .then(data => {
        //         // Remove any existing video element
        //         var existingVideo = document.getElementById('videoPlayer');
        //         if (existingVideo) {
        //             existingVideo.parentNode.removeChild(existingVideo);
        //         }

        //         // Create a new video element
        //         var videoPlayer = document.createElement('video');
        //         videoPlayer.id = 'videoPlayer';
        //         videoPlayer.width = 320;
        //         videoPlayer.height = 240;
        //         videoPlayer.controls = true;

        //         // Create a source element
        //         var sourceElement = document.createElement('source');

        //         // Set the src attribute using the combined MP4 path received from the server
        //         sourceElement.src = data.combined_mp4_path;
        //         sourceElement.type = 'video/mp4';

        //         // Append the source element to the video element
        //         videoPlayer.appendChild(sourceElement);

        //         // Append the video element to the body or a specific container
        //         document.body.appendChild(videoPlayer);

        //         // Display a message indicating that the combined video is ready
        //         document.getElementById('output').innerText = data.message;
        //     })
        //     .catch(error => console.error('Error:', error));
        // });

    } else {
        alert('Speech recognition is not supported in your browser. Please use a modern browser.');
    }
};

export default ATS;