from ultralytics import YOLO
import cv2
import onnxruntime as ort
import numpy as np

# onnx_model = ort.InferenceSession(
#     "./runs/detect/train8/weights/best.onnx", 
#     providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"],
# )

# def predict(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (640,640), interpolation = cv2.INTER_LINEAR)
#     image = np.expand_dims(image, axis=0).astype('float32') / 255.
#     image = np.transpose(image, [0, 3, 1, 2])
#     outputs = onnx_model.run(None, {'input': image})
#     return outputs

# model = predict

if __name__ == "__main__":
    # print(ort.__version__)
    # print(ort.get_device() ) # 如果得到的输出结果是GPU，所以按理说是找到了GPU的
    # session = ort.InferenceSession(
    #     "./runs/detect/train8/weights/best.onnx", 
    #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"],
    # )

    model = YOLO("./runs/detect/train8/weights/best.onnx")
    source = "F:/YOLO/multicell/multicell1.avi"
    cap = cv2.VideoCapture(source)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, device="0")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()