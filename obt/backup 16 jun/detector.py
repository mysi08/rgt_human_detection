#!/usr/bin/env python3
import cv2
import numpy as np
import openvino as ov

class YoloDetector:
    def __init__(self, model_path, num_threads, conf_thresh, nms_thresh, max_candidates):
        self.conf_threshold = conf_thresh
        self.nms_threshold = nms_thresh
        self.max_candidates = max_candidates

        print("Loading YOLO OpenVINO model...")
        print(f"Model Path: {model_path}")
        self.ie = ov.Core()
        self.model = self.ie.read_model(model_path)
        self.compiled_model = self.ie.compile_model(self.model, "CPU", config={
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": str(num_threads)
        })
        self.output_layer = self.compiled_model.outputs[0]
        # Warmup execution
        self.compiled_model([np.zeros((1, 3, 640, 640), dtype=np.float32)])
        print("YOLO initialization complete.")

    def run(self, cv_image):
        orig_h, orig_w = cv_image.shape[:2]
        
        # 1. C++ OPTIMIZED PREPROCESSING
        inp = cv2.dnn.blobFromImage(cv_image, 1.0/255.0, (640, 640), swapRB=False, crop=False)
        
        # 2. INFERENCE
        output = self.compiled_model([inp])[self.output_layer].squeeze() # Shape: (84, 8400)
        
        # 3. POST-PROCESSING OPTIMIZATION
        scores = output[4, :] 
        mask = scores > self.conf_threshold
        
        boxes = output[:4, mask].T # Shape: (N, 4)
        confs = scores[mask]
        
        if len(confs) > self.max_candidates:
            keep = np.argpartition(confs, -self.max_candidates)[-self.max_candidates:]
            boxes = boxes[keep]
            confs = confs[keep]

        detections = []
        if len(boxes) > 0:
            sx, sy = orig_w / 640.0, orig_h / 640.0
            
            # Vectorized scaling operations
            x1 = ((boxes[:, 0] - boxes[:, 2] / 2) * sx).astype(np.int32)
            y1 = ((boxes[:, 1] - boxes[:, 3] / 2) * sy).astype(np.int32)
            w  = (boxes[:, 2] * sx).astype(np.int32)
            h  = (boxes[:, 3] * sy).astype(np.int32)
            
            pred_boxes = np.column_stack((x1, y1, w, h)).tolist()
            pred_confs = confs.astype(float).tolist()
            
            idxs = cv2.dnn.NMSBoxes(pred_boxes, pred_confs, self.conf_threshold, self.nms_threshold)
            if len(idxs) > 0:
                # Direct array indexing for NMS output flattening
                flat_idxs = np.array(idxs).flatten()
                for i in flat_idxs:
                    x, y, bw, bh = pred_boxes[i]
                    detections.append({
                        "bbox": [x, y, x + bw, y + bh],
                        "conf": pred_confs[i]
                    })
        return detections