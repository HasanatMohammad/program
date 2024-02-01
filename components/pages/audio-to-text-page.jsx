import React, { useEffect } from 'react'
import './../../functions/audio-to-text'
import ATS from './../../functions/audio-to-text'

const AudioToText = () => {
    useEffect(() => {
        ATS();
    }, [])
    return (
        <div>
            <h1>Speech Recognition Example</h1>
            <p id="output">Speech recognition output will appear here. <br /></p>

            {/*  Add start and stop buttons  */}
            <button id="startButton"
                onClick={(e) => {
                    ATS.recognition.start();
                }}
            >Start Recognition</button>
            <button id="stopButton" disabled
                onClick={(e) => {
                    ATS.recognition.stop();
                }}
            >Stop Recognition</button>

        </div>
    )
}

export default AudioToText