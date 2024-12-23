import kagglehub

# Download the EfficientDet TFLite model
model_path = kagglehub.model_download("tensorflow/efficientdet/tfLite/lite0-detection-metadata")

# Check where the model is stored
print(f"Model downloaded to: {model_path}")

# Specify the path to the .tflite file (corrected to match the actual file name)
tflite_model_path = f"{model_path}/1.tflite"
print(f"Path to the .tflite model: {tflite_model_path}")
