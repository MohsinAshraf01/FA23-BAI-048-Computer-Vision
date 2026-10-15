# Landmark Recognition Using SIFT and Feature Matching

A Computer Vision lab project that recognizes famous landmarks in query images using **Scale-Invariant Feature Transform (SIFT)**, descriptor matching, and geometric verification with **RANSAC**. The notebook also tests how recognition behaves when query images are rotated, resized, or made darker.

## Project Overview

The notebook builds a small reference database from landmark images, extracts SIFT keypoints and descriptors, compares query-image features with the reference images, and ranks candidate landmarks by their feature matches. A geometric verification stage uses a homography and RANSAC inliers to decide whether a candidate should be accepted or rejected as `Unknown`.

## Features

- Loads and summarizes images organized in landmark-specific folders.
- Extracts SIFT keypoints and 128-dimensional descriptors.
- Matches descriptors using a brute-force matcher with the L2 distance and a ratio test.
- Ranks landmark candidates based on reliable feature matches.
- Uses homography estimation and RANSAC to verify geometric consistency.
- Visualizes keypoints, feature matches, and landmark localization where applicable.
- Evaluates transformed queries, including rotation, resizing, and reduced brightness.
- Reports predicted landmark, match counts, RANSAC inliers, decision, correctness, and processing time.

## Technologies Used

- Python
- OpenCV (`opencv-contrib-python`, for SIFT support)
- NumPy
- Matplotlib
- Pandas
- Google Colab / Google Drive

## Dataset Structure

The notebook expects the image dataset at `drive/MyDrive/image_dataset` in Google Drive. Place each landmark's reference images in its own folder, for example:

```text
image_dataset/
├── colossem/
│   ├── image1.jpg
│   └── image2.jpg
├── eifel/
│   ├── image1.jpg
│   └── image2.jpg
├── statue/
├── taj_mahal/
└── wall/
```

Folder names are used as landmark labels. The folder names above reflect the current notebook's dataset naming; you can rename them if you also update the dataset folders and rerun the notebook. Use your own images or a dataset you have permission to use, and include any dataset attribution required by its license.

## How to Run

1. Upload `CV_lab_6.ipynb` to Google Colab.
2. Put the dataset in Google Drive under `MyDrive/image_dataset`, following the folder structure above.
3. Mount Google Drive when prompted, or add the appropriate Drive-mount cell if it is not already present in your Colab session.
4. Install the required packages if needed:

   ```python
   !pip install opencv-contrib-python numpy matplotlib pandas
   ```

5. Run the notebook cells from top to bottom.
6. Set the query image path to an image you want to recognize. Ensure the path points to an existing image file.
7. Review the candidate matches, localization visualization, and evaluation table.

> **Note:** The notebook uses a Google Drive path. If you run it locally, update `DATASET_PATH` and the query-image path to match your local folders.

## Recognition Method

1. **Reference preparation:** Read images from each landmark folder.
2. **Feature extraction:** Convert images to grayscale and compute SIFT keypoints and descriptors.
3. **Descriptor matching:** Compare query descriptors against reference descriptors using L2 distance and a ratio test.
4. **Candidate selection:** Rank candidate landmarks by the number of reliable matches.
5. **Geometric verification:** Estimate a homography with RANSAC and count inliers to reject inconsistent matches.
6. **Evaluation:** Test original and transformed query images and record the recognition decision and runtime.

## Limitations

- Recognition quality depends on the number, variety, and quality of reference images.
- Major viewpoint changes, occlusion, blur, or very different lighting can reduce reliable matches.
- A small dataset is useful for a lab demonstration but is not enough to establish real-world performance across all conditions.
- The notebook's evaluation results depend on the supplied dataset and query image; results may differ when the data changes.

## Repository Contents

```text
.
├── CV_lab_6.ipynb
└── README.md
```

## Author

**Mohsin Ashraf**  
Computer Vision Lab Project

## License

Add a license if you intend to distribute or reuse this project publicly. Check the licenses of all dataset images before publishing them.
