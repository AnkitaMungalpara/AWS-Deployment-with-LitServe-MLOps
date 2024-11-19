import torch
import timm
from PIL import Image
import io
import litserve as ls
import base64
import boto3

from timm_classifier import TimmClassifier  # Import the TimmClassifier class

# Define precision globally - can be changed to torch.float16 or torch.bfloat16
precision = torch.bfloat16

class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components"""
        self.device = device
        # Load model from S3
        s3 = boto3.client('s3')
        bucket_name = 'mlops-aws'
        model_key = 'model/cat_dog_model.ckpt'
        
        # Download the model file from S3
        model_file = 'cat_dog_model.ckpt'
        # s3.download_file(bucket_name, model_key, model_file)
        
        # Load the model from checkpoint and move to appropriate device with specified precision
        self.model = TimmClassifier.load_from_checkpoint(model_file)
        self.model = self.model.to(device).to(precision)
        self.model.eval()

        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        # Load class labels
        self.labels = ["Cat", "Dog"]

    def decode_request(self, request):
        """Convert base64 encoded image to tensor"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(image_bytes)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(img_bytes))
        # Transform image to tensor and convert to specified precision
        tensor = self.transforms(image).to(self.device).to(precision)
        return tensor

    @torch.no_grad()
    def predict(self, x):
        """Run inference on the input batch"""
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities

    def encode_response(self, output):
        """Convert model output to API response"""
        # Get top 5 predictions
        probs, indices = torch.topk(output[0], k=2)
        
        return {
            "predictions": [
                {
                    "label": self.labels[idx.item()],
                    "probability": prob.item()
                }
                for prob, idx in zip(probs, indices)
            ]
        }

if __name__ == "__main__":
    api = ImageClassifierAPI()
    # Configure server with batching
    server = ls.LitServer(
        api,
        accelerator="gpu",
        max_batch_size=64,  # Adjust based on your GPU memory and requirements
        batch_timeout=0.01,  # Timeout in seconds to wait for forming batches
        workers_per_device=4
    )
    server.run(port=8000)