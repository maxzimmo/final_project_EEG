const express = require('express');
const multer = require('multer');

// Create an instance of Express
const app = express();

// Set up multer storage and file upload settings
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // Set the destination folder where the uploaded file will be saved
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    // Set the filename for the uploaded file
    cb(null, file.originalname);
  }
});

// Create an instance of the multer middleware
const upload = multer({ storage });

// Define a route to handle the file upload
app.post('/upload', upload.single('file'), (req, res) => {
  // Access the uploaded file details
  const file = req.file;
  console.log(file);

  // Send a response to the client
  res.send('File uploaded successfully');
});

// Start the server
app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
