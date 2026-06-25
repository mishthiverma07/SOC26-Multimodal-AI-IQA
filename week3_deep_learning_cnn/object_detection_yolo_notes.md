# Object Detection and YOLO Notes

- Image classification classifies the whole image.
- Object detection finds objects inside an image using bounding boxes.
- Image segmentation classifies each pixel.
- Sliding window is an older object detection method.
- YOLO stands for You Only Look Once.
- YOLO divides an image into grid cells.
- Each grid cell predicts object probability, bounding box coordinates, and class probabilities.
- YOLO is fast because it detects objects in one forward pass.
- Intersection over Union measures overlap between bounding boxes.
- Non-Max Suppression removes duplicate bounding boxes.
- Anchor boxes help detect multiple objects in the same grid cell.
- YOLO can use pre-trained weights trained on COCO dataset.