# Cartoonization and Artistic Style Transfer of Images

## Team Name: ML Maniacs

### Team Members and Contributions
1. **Yenneti Nitin Sree Venkat** (12242060)  
   - Worked on **Image Cartoonization** ,using image processing techniques in order to cartoonify the content image.  
   
2. **B Hari Charan Goud** (12240360)  
   - Worked on **Style Transfer**, leveraging neural networks to apply artistic styles to the cartoonized content image.  
   
3. **Yeldandi Suchethan Reddy** (12242050)  
   - Handled **Website Deployment**, ensuring the application is accessible via a web interface.
   - Created dataset required for training style transfer model.

---
## Project Overview
This project combines image processing and deep learning techniques to achieve a seamless **style transfer** effect. It involves:
1. **Cartoonization**:
   - Cartoonizing the content image to enhance its compatibility for style transfer.
2. **Style Transfer**:
   - Applying artistic styles to the cartoonized image using pre-trained neural network models.
3. **Web Application**:
   - Deploying the system on a website for user interaction and accessibility.
     
---
## Workflow

### 1. Cartoonization:
   - Apply bilateral filtering to smoothen the image while preserving edges.
   - Use Canny edge detection for extracting edges.

### 2. Style Transfer:
   - Use the cartoonized image as the content image.
   - Apply a pre-trained style transfer model to stylize the cartoonized image.

---

## Cartoonization Workflow

1. **Image Upload**:  
   The program allows you to upload an image using Google Colab's `files.upload()` method.  

2. **Resize Image**:  
   Resizes the uploaded image to a target width (default 600 px) while maintaining its aspect ratio.  

3. **Convert to Grayscale**:  
   Converts the image to grayscale and applies a median blur to reduce noise.  

4. **Sharpening**:  
   Enhances the details of the image using a custom sharpening kernel.  

5. **Edge Detection**:  
   Detects the edges in the grayscale image using the Canny edge detection algorithm. The edges are then inverted for blending.  

6. **Bilateral Filtering**:  
   Applies multiple iterations of bilateral filtering to smooth the image while preserving strong edges for a cartoon-like effect.  

7. **Cartoonization**:  
   Combines the smoothed image with the detected edges to produce the cartoon effect.  

8. **Blending**:  
   Blends the cartoonized image with the original colors for a more realistic cartoon look. Additionally, edges are softly blended with the final image.  

9. **Gamma Correction**:  
   Enhances the brightness and contrast of the final cartoonized image for a polished look.  

---
In this project, the **content image** used for style transfer undergoes a preprocessing step to achieve a cartoonized effect. This is done by applying **edge detection techniques**.

## Edge Detection Techniques Used

1. **Adaptive Thresholding**  
   Initially, **adaptive thresholding** was used for edge detection. This method determines the threshold for a small region of the image, which helps in preserving edges in areas with varying illumination. The steps included:  
   - Converting the image to grayscale.
   - Applying Gaussian blur to reduce noise.
   - Using adaptive thresholding to highlight edges.
  
![Original Image](C:\Users\Yenneti Nitin Sree\OneDrive\Pictures\Cartoonization\c2.jpg)


2. **Canny Edge Detection**  
   Currently, the **Canny edge detection** technique is being utilized for cartoonization. This method is effective in detecting edges with a clean and sharp appearance. The steps include:  
   - Converting the image to grayscale.
   - Applying Gaussian blur to smoothen the image.
   - Using the Canny algorithm to detect edges.

---

## Why Switch to Canny Edge Detection?

- **Sharpness**: Canny edge detection provides cleaner and more defined edges compared to adaptive thresholding.  
- **Consistency**: It works better for creating cartoonized images that maintain a consistent edge style.  
- **Compatibility**: The resulting edges blend well as a content image for the **style transfer project**.

---


