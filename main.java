// Importing the k−Nearest Neighbors Algorithm
import {
  KNNImageClassifier
} from 'deeplearn-knn-image-classifier';
import * as dl from 'deeplearn';

// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN. 10 means that we will take votes from 10 data points to classify each
// tensor.
const TOPK = 10;

// Percent confidence above which prediction needs to be to return a prediction.
const confidenceThreshold = 0.98;

// Initial Gestures that need to be trained.
// The start gesture is for signalling when to start prediction
// The stop gesture is for signalling when to stop prediction
var words = ["start", "stop"];

/*
The Main class is responsible for the training and prediction of words.
It controls the webcam, user interface, as well as initiates the output of predicted words.
*/
class Main {

  constructor() {
    // Initialize variables for display as well as prediction purposes
    this.exampleCountDisplay = [];
    this.checkMarks = [];
    this.gestureCards = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;
    this.previousPrediction = -1;
    this.currentPredictedWords = [];

    // Variables to restrict prediction rate
    this.now;
    this.then = Date.now();
    this.startTime = this.then;

    this.fps = 5; // framerate - number of prediction per second
    this.fpsInterval = 1000 / this.fps;
    this.elapsed = 0;

    // Initalizing kNNmodel to none.
    this.knn = null;

    /* Initalizing previous kNNmodel that we trained when training of the current model
    is stopped or prediction has begun. */
    this.previousKnn = this.knn;

    // Storing all elements that from the User Interface that need to be altered into variables.
    this.welcomeContainer = document.getElementById("welcomeContainer");
    this.proceedBtn = document.getElementById("proceedButton");
    this.proceedBtn.style.display = "block";
    this.proceedBtn.classList.add("animated");
    this.proceedBtn.classList.add("flash");
    this.proceedBtn.addEventListener('click', () => {
      this.welcomeContainer.classList.add("slideOutUp");
    });

    this.stageTitle = document.getElementById("stage");
    this.stageInstruction = document.getElementById("steps");
    this.predButton = document.getElementById("predictButton");
    this.backToTrainButton = document.getElementById("backButton");
    this.nextButton = document.getElementById('nextButton');
    this.statusContainer = document.getElementById("status");
    this.statusText = document.getElementById("status-text");
    this.translationHolder = document.getElementById("translationHolder");
    this.translationText = document.getElementById("translationText");
    this.translatedCard = document.getElementById("translatedCard");
    this.initialTrainingHolder = document.getElementById('initialTrainingHolder');
    this.videoContainer = document.getElementById("videoHolder");
    this.video = document.getElementById("video");
    this.trainingContainer = document.getElementById("trainingHolder");
    this.addGestureTitle = document.getElementById("add-gesture");
    this.plusImage = document.getElementById("plus sign");
    this.addWordForm = document.getElementById("add-word");
    this.newWordInput = document.getElementById("new-word");
    this.doneRetrain = document.getElementById("doneRetrain");
    this.trainingCommands = document.getElementById("trainingCommands");
    this.videoCallBtn = document.getElementById("videoCallBtn");
    this.videoCall = document.getElementById("videoCall");
    this.trainedCardsHolder = document.getElementById("trainedCardsHolder");

    // Start Translator function is called
    this.initializeTranslator();

    // Instantiate Prediction Output  (Assuming PredictionOutput is defined elsewhere)
    this.predictionOutput = new PredictionOutput();  // Replace with actual instantiation

  }

  /*This function starts the webcam and initial training process. It also loads the kNN
  classifier */
  initializeTranslator() {
    this.startWebcam();
    this.initialTraining();
    this.loadKNN();
  }

  //This function sets up the webcam
  startWebcam() {
    navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user'
        },
        audio: false
      })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;
        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      });
  }

  /*This function initializes the training for Start and Stop Gestures. It also
  sets a click listener for the next button. */
  initialTraining() {
    // if next button on initial training page is pressed, setup the custom gesture training UI.
    this.nextButton.addEventListener('click', () => {
      const exampleCount = this.knn.getClassExampleCount();
      if (Math.max(...exampleCount) > 0) {
        // if start gesture has not been trained
        if (exampleCount[0] == 0) {
          alert('You haven\'t added examples for the Start Gesture');
          return;
        }
        // if stop gesture has not been trained
        if (exampleCount[1] == 0) {
          alert('You haven\'t added examples for the Stop Gesture.\n\nCapture yourself in idle states e.g hands by your side, empty background etc.');
          return;
        }

        this.nextButton.style.display = "none";
        this.stageTitle.innerText = "Continue Training";
        this.stageInstruction.innerText = "Add Gesture Name and Train.";

        //Start custom gesture training process
        this.setupTrainingUI();
      }
    });

    //Create initial training buttons
    this.initialGestures(0, "startButton");
    this.initialGestures(1, "stopButton");
  }

  //This function loads the kNN classifier
  loadKNN() {
    this.knn = new KNNImageClassifier(words.length, TOPK);
    // Load knn model
    this.knn.load().then(() => this.initializeTraining()); // Double check this initializeTraining call.  Is it needed?  Potentially a bug.
  }

  /*This creates the training and clear buttons for the initial Start and Stop gesture.
  It also creates the Gesture Card. */
  initialGestures(i, btnType) {
    // Get specified training button
    var trainBtn = document.getElementById(btnType);

    // Call training function for this gesture on click  (Assuming train function is defined later in the class)
    trainBtn.addEventListener('click', () => {
      this.train(i);  // Ensure this.train(i) exists
    });

    // Clear button to remove training examples on click
    var clearBtn = document.getElementById('clear' + btnType);
    clearBtn.addEventListener('click', () => {
      this.knn.clearClass(i);
      this.exampleCountDisplay[i].innerText = " 0 examples";
      this.gestureCards[i].removeChild(this.gestureCards[i].childNodes[1]);
      this.checkMarks[i].src = "Images\\loader.gif";
    });

    // Variables for training information for the user
    var exampleCountDisplay = document.getElementById('counter' + btnType);
    var checkMark = document.getElementById('checkmark' + btnType);

    // Create Gesture Card
    var gestureCard = document.createElement("div");
    gestureCard.className = "trained-gestures";
    var gestName = "";
    if (i == 0) {
      gestName = "Start";
    } else {
      gestName = "Stop";
    }
    var gestureName = document.createElement("h5");
    gestureName.innerText = gestName;
    gestureCard.appendChild(gestureName);
    this.trainedCardsHolder.appendChild(gestureCard);
    exampleCountDisplay.innerText = " 0 examples";
    checkMark.src = 'Images\\loader.gif';

    this.exampleCountDisplay.push(exampleCountDisplay);
    this.checkMarks.push(checkMark);
    this.gestureCards.push(gestureCard);
  }

  /*This function sets up the custom gesture training UI. */
  setupTrainingUI() {
    const exampleCount = this.knn.getClassExampleCount();
    // check if training is complete
    if (Math.max(...exampleCount) > 0) {
      // if start gesture has not been trained
      if (exampleCount[0] == 0) {
        alert('You haven\'t added examples for the wake word');
        return;
      }
      // if stop gesture has not been trained
      if (exampleCount[1] == 0) {
        alert('You haven\'t added examples for the Stop Gesture.\n\nCapture yourself in idle states e.g hands by your side, empty background etc.');
        return;
      }

      // Remove Initial Training Screen
      this.initialTrainingHolder.style.display = "none";
      // Add the Custom Gesture Training UI
      this.trainingContainer.style.display = "block";
      this.trainedCardsHolder.style.display = "block";

      // Add Gesture on Submission of new gesture form
      this.addWordForm.addEventListener('submit', (e) => {
        e.preventDefault(); // Prevent form submission

        const newWord = this.newWordInput.value.trim();

        if (newWord !== "" && words.indexOf(newWord) === -1) { // Check for empty string and duplicate
          words.push(newWord);
          this.knn.numClasses = words.length; // Update KNN class count
          this.createTrainingBtns(words.length - 1); // Pass the index of the new word
          this.newWordInput.value = ""; // Clear the input
          this.addWordForm.style.display = "none";  // Hide the form after submission.
          this.addGestureTitle.innerText = "Add Gesture"; // Reset the title
          this.plusImage.src = "Images/plus sign.svg";
          this.plusImage.classList.add("rotateInLeft");

        } else {
          alert("Duplicate word or no word entered");
        }
        return;
      });
    } else {
      alert('You haven\'t added any examples yet.\n\nAdd a Gesture, then perform the sign in front of the webcam.');
    }
  }

  /*This creates the training and clear buttons for the new gesture. It also creates the
  Gesture Card. */
  createTrainingBtns(i) { // i is the index of the newword
    // Create Train and Clear Buttons
    var trainBtn = document.createElement('button');
    trainBtn.className = "trainBtn";
    trainBtn.innerText = "Train";
    this.trainingCommands.appendChild(trainBtn);

    var clearBtn = document.createElement('button');
    clearBtn.className = "clearButton";
    clearBtn.innerText = "Clear";
    this.trainingCommands.appendChild(clearBtn);

    // Change training class from none to specified class if training button is pressed
    trainBtn.addEventListener('mousedown', () => {
      this.train(i);
    });

    // Create clear button to remove training examples on click
    clearBtn.addEventListener('click', () => {
      this.knn.clearClass(i);
      this.exampleCountDisplay[i].innerText = " 0 examples";
      this.gestureCards[i].removeChild(this.gestureCards[i].childNodes[1]);
      this.checkMarks[i].src = 'Images\\loader.gif';
    });

    // Create elements to display training information for the user
    var exampleCountDisplay = document.createElement('h3');
    exampleCountDisplay.style.color = "black";
    this.trainingCommands.appendChild(exampleCountDisplay);

    var checkMark = document.createElement('img');
    checkMark.className = "checkMark";
    this.trainingCommands.appendChild(checkMark);

    //Create Gesture Card
    var gestureCard = document.createElement("div");
    gestureCard.className = "trained-gestures";
    var gestName = words[i];
    var gestureName = document.createElement("h5");
    gestureName.innerText = gestName;
    gestureCard.appendChild(gestureName);
    this.trainedCardsHolder.appendChild(gestureCard);
    exampleCountDisplay.innerText = " 0 examples";
    checkMark.src = 'Images\\loader.gif';
    this.exampleCountDisplay.push(exampleCountDisplay);
    this.checkMarks.push(checkMark);
    this.gestureCards.push(gestureCard);

    // Retrain/Continue Training gesture on click of the gesture card
    gestureCard.addEventListener('click', () => { // create btn
      /* If gesture card was not already pressed display the specific gesture card’s
      training buttons to train it */
      if (gestureCard.style.marginTop == "17px" || gestureCard.style.marginTop == "") {
        // Display done retraining button and the training buttons for the specific gesture
        this.doneRetrain.style.display = "block";
        this.trainingCommands.innerHTML = "";
        this.trainingCommands.appendChild(trainBtn);
        this.trainingCommands.appendChild(clearBtn);
        this.trainingCommands.appendChild(exampleCountDisplay);
        this.trainingCommands.appendChild(checkMark);
        gestureCard.style.marginTop = "-10px";
        this.addWordForm.style.display = "none"; // Hide form.
        this.addGestureTitle.innerText = gestName;
        this.plusImage.src = "Images/retrain.svg";
        this.plusImage.classList.add("rotateIn");


      }
      // if gesture card is pressed again, change the add gesture card back to add gesture mode
      // instead of retrain mode
      else {
        this.addGestureTitle.innerText = "Add Gesture";
        this.addWordForm.style.display = "block";
        gestureCard.style.marginTop = "17px";
        this.trainingCommands.innerHTML = "";
        this.addWordForm.style.display = "block";
        this.doneRetrain.style.display = "none";
        this.plusImage.src = "Images/plus sign.svg";
        this.plusImage.classList.add("rotateInLeft");
      }
    });

    // if done retrain button is pressed again, change the add gesture card back to add gesture
    // mode instead of retrain mode
    this.doneRetrain.addEventListener('click', () => {
      this.addGestureTitle.innerText = "Add Gesture";
      this.addWordForm.style.display = "block";
      gestureCard.style.marginTop = "17px";

      this.video.className = "videoTrain";
      this.videoContainer.className = "videoContainerTrain";
      this.videoCallBtn.style.display = "none";
      this.translationHolder.style.display = "none";
      this.statusContainer.style.display = "none";
      // Show elements from training mode
      this.trainingContainer.style.display = "block";
      this.trainedCardsHolder.style.marginTop = "0px";
      this.trainedCardsHolder.style.display = "block";
    });
  }

  train(classId) {
    if (!this.videoPlaying) {
      return;
    }
    this.training = classId;
    let total = this.knn.getClassExampleCount();
    let gestureType = document.getElementsByClassName("trained-gestures")[classId];
    let gestureImg = document.createElement('canvas');
    gestureImg.className = "trained image";
    gestureImg.getContext('2d').drawImage(this.video, 0, 0, 400, 180);
    if (total[classId] == 0) {
      gestureType.appendChild(gestureImg);
    }
    this.video.className = "videoTrain";
    this.videoContainer.className = "videoContainerTrain";
    this.videoCallBtn.style.display = "none";
    this.translationHolder.style.display = "none";
    this.statusContainer.style.display = "none";

    // Show elements from training mode
    this.trainingContainer.style.display = "block";
    this.trainedCardsHolder.style.marginTop = "0px";
    this.trainedCardsHolder.style.display = "block";

    dl.tidy(() => {
      const image = dl.fromPixels(this.video);
      this.knn.addExample(image, this.training);
    });
    let exampleCount = this.knn.getClassExampleCount();
    this.exampleCountDisplay[classId].innerText = exampleCount[classId] + " examples";
    this.checkMarks[classId].src = 'Images\\check.svg';

  }

  //Assuming a PredictionOutput class exists.  Placeholder implementation.
  setStatusText(text, mode) {
    console.log("Status: " + text + " Mode:" + mode);
  }

  speak(text) {  // Requires browser support for SpeechSynthesis API
    console.log("Speaking: " + text);
  }

  textOutput(word, gestureCard, gestureAccuracy) {
    // If the word is start, clear translated text content
    if (word == 'start') {
      this.clearPara();
      setTimeout(() => {
        this.currentPredictedWords.push(word);
        // Depending on the word, display the text output
        if (word == "start") {
          this.translationText.innerText += ' ';
        } else if (word == "stop") {
          this.translationText.innerText += '. ';
        } else {
          this.translationText.innerText += ' ' + word;
        }

        // Clone Gesture Card
        this.translatedCard.innerHTML = "";
        var clonedCard = document.createElement("div");
        clonedCard.className = "trained-gestures";
        var gestName = gestureCard.childNodes[0].innerText;
        var gestureName = document.createElement("h5");
        gestureName.innerText = gestName;
        clonedCard.appendChild(gestureName);
        var gestureImg = document.createElement("canvas");
        gestureImg.className = "trained image";
        gestureImg.getContext('2d').drawImage(gestureCard.childNodes[1], 0, 0, 400, 180);
        clonedCard.appendChild(gestureImg);
        var gestAccuracy = document.createElement("h7");
        gestAccuracy.innerText = "Confidence : " + gestureAccuracy + "%";
        clonedCard.appendChild(gestAccuracy);
        this.translatedCard.appendChild(clonedCard);

        // If its not video call mode, speak out the user's word
        if (word != "start" && word != "stop") {
          this.speak(word);
        }
      }, 0); // setTimeout is needed to avoid blocking execution
    }
  }

  /* This functions clears translation text and cards. Sets the previous predicted words to
  null */
  clearPara() {
    this.translationText.innerText = '';
    this.previousPrediction = -1;
    this.currentPredictedWords = []; // empty words in this query
    this.translatedCard.innerHTML = '';
  }

  copyTranslation() {
    this.translationHolder.addEventListener('mousedown', () => {
      this.setStatusText("Text Copied!", "copy");

      // const el = document.createElement('textarea');   Missing code from original snippet.

      // Add the rest of the implementation here.
      // For example:
      // el.value = this.translationText.innerText;
      // document.body.appendChild(el);
      // el.select();
      // document.execCommand('copy');
      // document.body.removeChild(el);

    });
  }
}