# Face Detection

# Goal:
- To identify and locate faces in a video using eigenfaces.

# Procedure
- Train using dataset of faces
- I used the AT&T face dataset
- Includes 40 subjects with 10 photos each

- Compute mean of all images

- Compute the covariance matrix
  - Subtract the average from each image to get difference
  
  
- Compute the eigenvectors of covariance matrix
  - Detect important locations by getting largest eigenvalues
- Projecting training data onto sorted eigenvectors
- Then projecting test data onto sorted eigenvectors and computing difference
- If difference is large: 
  - Then it is not a face
- If small:
  - Then it’s a face!

# Complications
- Easy if image is of face only.
- Can send image as is and compute the difference
- Will let you know if the entire image is of a face or not

- For images with other objects in the view:
  - Read sections via sliding windows of varying size
  - See where was the best match for a face
  
![alt text](https://github.com/muhammadahmad2/FaceDetection/raw/master/Picture1.png)
![alt text](https://github.com/muhammadahmad2/FaceDetection/raw/master/Picture2.png)

