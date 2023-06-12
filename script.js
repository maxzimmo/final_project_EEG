// Add drag and drop functionality
function addDragAndDrop() {
  // Get the drop area element
  const dropArea = document.getElementById("dropzone");

  // Prevent default drag behaviors
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  // Highlight the drop area when a draggable item is dragged over it
  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false);
  });

  // Remove the highlight when a draggable item is dragged out of the drop area
  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false);
  });

  // Handle the dropped file
  dropArea.addEventListener("drop", handleDrop, false);

  // Prevent default drag behaviors
  function preventDefaults(event) {
    event.preventDefault();
    event.stopPropagation();
  }

  // Highlight the drop area
  function highlight() {
    dropArea.classList.add("highlight");
  }

  // Remove the highlight from the drop area
  function unhighlight() {
    dropArea.classList.remove("highlight");
  }

  // Handle the dropped file
  function handleDrop(event) {
    const files = event.dataTransfer.files;
    // Handle the dropped files here
    // You can access the file(s) using the `files` object
    uploadFiles(files);
  }
}

// Set up drag and drop functionality
addDragAndDrop();

// File selection and upload
document.getElementById("uploadButton").addEventListener("click", function () {
  document.getElementById("uploadInput").click();
});

document.getElementById("uploadInput").addEventListener("change", function () {
  const files = this.files;
  // Handle the selected files here
  // You can access the file(s) using the `files` object
  uploadFiles(files);
});

function uploadFiles(files) {
  // Create a new FormData object
  const formData = new FormData();

  // Append the files to the FormData object
  for (let i = 0; i < files.length; i++) {
    formData.append("file", files[i]);
  }

  // Create a new XMLHttpRequest object
  const xhr = new XMLHttpRequest();

  // Set up the event listeners for the XMLHttpRequest object
  xhr.upload.addEventListener("progress", handleUploadProgress);
  xhr.addEventListener("load", handleUploadComplete);
  xhr.addEventListener("error", handleUploadError);

  // Send the file using the XMLHttpRequest object
  xhr.open("POST", "/upload", true);
  xhr.send(formData);
}

function handleUploadProgress(event) {
  if (event.lengthComputable) {
    // Calculate the progress percentage
    const progressPercentage = (event.loaded / event.total) * 100;

    // Update the value of the progress bar
    const progressBar = document.getElementById("progress-bar");
    progressBar.value = progressPercentage;
  }
}

function handleUploadComplete(event) {
  // Handle the upload completion
  console.log("Upload completed");
}

function handleUploadError(event) {
  // Handle the upload error
  console.error("Upload error occurred");
}
