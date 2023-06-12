function addDragAndDrop() {
  // Get the drop area element
  var dropArea = document.getElementById("drop-area");

  // Prevent default drag behaviors
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  // Highlight the drop area when a draggable item is dragged over it
  ['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
  });

  // Remove the highlight when a draggable item is dragged out of the drop area
  ['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
  });

  // Handle the dropped file
  dropArea.addEventListener('drop', handleDrop, false);

  // Prevent default drag behaviors
  function preventDefaults(event) {
    event.preventDefault();
    event.stopPropagation();
  }

  // Highlight the drop area
  function highlight() {
    dropArea.classList.add('highlight');
  }

  // Remove the highlight from the drop area
  function unhighlight() {
    dropArea.classList.remove('highlight');
  }

  // Handle the dropped file
  function handleDrop(event) {
    var files = event.dataTransfer.files;
    // Handle the dropped files here
    // You can access the file(s) using the `files` object
  }
}

document.getElementById("uploadButton").addEventListener("click", function () {
  document.getElementById("uploadInput").click();
});

document.getElementById("uploadInput").addEventListener("change", function () {
  var files = this.files;
  // Handle the selected files here
  // You can access the file(s) using the `files` object
});


// Call the function at the bottom of your scripts.js file
addDragAndDrop();

