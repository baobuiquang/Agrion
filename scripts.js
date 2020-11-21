// Classifier Variable
let classifier;
// Model URL
let imageModelURL = './model/';
// Video
let video;
let flippedVideo;
// To store the classification
let label = "Waiting for capture device";
let label1 = "Waiting for capture device";
let label2 = "Waiting for capture device";
let confidence = 0;
let confidence1 = 0.0;
let confidence2 = 0.0;
// Some variables
var w = window.innerWidth;
var h = window.innerHeight;
var videoWidth = w * 0.3;
var videoHeight = videoWidth / 4 * 3;
var lineHeight = 32;
// Load the model first
function preload() {
    classifier = ml5.imageClassifier(imageModelURL + 'model.json');
}

var canvas;
function setup() {
    canvas = createCanvas(w, h);
    canvas.position(0, 0);
    canvas.style('z-index', '-1');

    // Create the video
    video = createCapture(VIDEO);
    video.size(videoWidth, videoHeight);
    video.hide();

    // flippedVideo = ml5.flipImage(video);
    // Start classifying
    classifyVideo();
}

function draw() {
    background(255);
    // Draw the video

    textStyle(NORMAL);
    fill(
        label=="In Good Condition"?0:label=="In Bad Condition"?255:255, 
        label=="In Good Condition"?255:label=="In Bad Condition"?0:255, 
        label=="In Good Condition"?0:label=="In Bad Condition"?0:255,);
    
    rect(w / 2 - videoWidth - 57, h / 2 - videoHeight / 2 - 7, videoWidth + 13, videoHeight + 13, 4);
    image(flippedVideo, w / 2 - videoWidth - 50, h / 2 - videoHeight / 2);


    stroke('transparent');
    // Vertically
    rect(w / 2 - videoWidth / 8 * 1 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 2 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 3 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 4 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 5 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 6 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    rect(w / 2 - videoWidth / 8 * 7 - 50, h / 2 - videoHeight / 2, 0, videoHeight);
    // Horizontally
    rect(w / 2 - videoWidth - 50, h / 2 + videoHeight / 6 * 1, videoWidth - 1, 0);
    rect(w / 2 - videoWidth - 50, h / 2 + videoHeight / 6 * 2, videoWidth - 1, 0);
    rect(w / 2 - videoWidth - 50, h / 2 + videoHeight / 6 * 0, videoWidth - 1, 0);
    rect(w / 2 - videoWidth - 50, h / 2 - videoHeight / 6 * 1, videoWidth - 1, 0);
    rect(w / 2 - videoWidth - 50, h / 2 - videoHeight / 6 * 2, videoWidth - 1, 0);

    // Draw the label
    fill(0);
    textSize(22);
    textAlign(LEFT);

    // fill('#59C9A5');
    // rect(w / 2 - 14, h / 2 - videoHeight / 2 + lineHeight * 0, 10, 108, 10);
    // fill(0);
    textStyle(BOLD);
    text(label, w / 2, h / 2 - videoHeight / 2 + lineHeight);
    textStyle(NORMAL);
    text(confidence.toFixed(5) + "%", w / 2, h / 2 - videoHeight / 2 + lineHeight * 2);
    rect(w / 2, h / 2 - videoHeight / 2 + lineHeight * 2.5, confidence * videoWidth / 100, lineHeight / 2, 100);
    
    textStyle(BOLD);
    text(label1, w / 2, h / 2 - videoHeight / 2 + lineHeight * 4.5);
    textStyle(NORMAL);
    text(confidence1.toFixed(5) + "%", w / 2, h / 2 - videoHeight / 2 + lineHeight * 5.5);
    rect(w / 2, h / 2 - videoHeight / 2 + lineHeight * 6, confidence1 * videoWidth / 100, lineHeight / 2, 100);

    textStyle(BOLD);
    text(label2, w / 2, h / 2 - videoHeight / 2 + lineHeight * 8);
    textStyle(NORMAL);
    text(confidence2.toFixed(5) + "%", w / 2, h / 2 - videoHeight / 2 + lineHeight * 9);
    rect(w / 2, h / 2 - videoHeight / 2 + lineHeight * 9.5, confidence2 * videoWidth / 100, lineHeight / 2, 100);

    textSize(14);
    textAlign(LEFT);
    text("AGRION VISION demo on mangoes", w / 2 - videoWidth - 50, h / 2 - videoHeight / 2 - 20);

    textSize(18);
    textAlign(CENTER);
    text("Product Status Diagnosis", width / 2, height - lineHeight * 4);
    textSize(36);
    fill(
        label=="In Good Condition"?0:label=="In Bad Condition"?255:255, 
        label=="In Good Condition"?255:label=="In Bad Condition"?0:255, 
        label=="In Good Condition"?0:label=="In Bad Condition"?0:255,);
    textStyle(BOLD);
    text(label, width / 2, height - lineHeight * 2.5);
    fill(255);

}

// Get a prediction for the current video frame
function classifyVideo() {
    flippedVideo = ml5.flipImage(video);
    classifier.classify(flippedVideo, gotResult);
    flippedVideo.remove();
}

// When we get a result
function gotResult(error, results) {
    // If there is an error
    if (error) {
        console.error(error);
        return;
    }
    // The results are in an array ordered by confidence.
    // console.log(results[0]);
    label = results[0].label;
    label1 = results[1].label;
    label2 = results[2].label;
    // label3 = results[3].label;
    confidence = results[0].confidence * 100;
    confidence1 = results[1].confidence * 100;
    confidence2 = results[2].confidence * 100;
    // Classifiy again!
    classifyVideo();
}