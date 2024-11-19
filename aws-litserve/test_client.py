import requests
from urllib.request import urlopen
import base64
import boto3  # Import boto3 for S3 access

def test_single_image():
    # Get test image from S3
    s3_bucket = 'mlops-aws'  # Replace with your S3 bucket name
    s3_key = 'input-images/4.jpg'  # Replace with the path to your image in S3
    s3 = boto3.client('s3')
    img_data = s3.get_object(Bucket=s3_bucket, Key=s3_key)['Body'].read()  # Fetch image from S3
    
    # Convert to base64 string
    img_bytes = base64.b64encode(img_data).decode('utf-8')
    
    # Send request
    response = requests.post(
        "<http://localhost:8000/predict>",
        json={"image": img_bytes}  # Send as JSON instead of files
    )
    
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        print("\\nTop 5 Predictions:")
        for pred in predictions:
            print(f"{pred['label']}: {pred['probability']:.2%}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_single_image()