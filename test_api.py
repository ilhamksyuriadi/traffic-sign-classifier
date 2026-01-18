"""
Test script for Traffic Sign Classification API
===============================================

This script tests the Flask API by sending sample images for prediction.

Usage:
    python test_api.py --image path/to/image.png
"""

import argparse
import requests
import json
from pathlib import Path


def test_health_check(base_url):
    """Test the health check endpoint"""
    print(f"\n{'='*50}")
    print(f"Testing Health Check Endpoint")
    print(f"{'='*50}")
    
    response = requests.get(f"{base_url}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_prediction(base_url, image_path):
    """Test the prediction endpoint with an image"""
    print(f"\n{'='*50}")
    print(f"Testing Prediction Endpoint")
    print(f"{'='*50}")
    print(f"Image: {image_path}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    # Open and send image
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{base_url}/predict", files=files)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Prediction Successful!")
        print(f"\n🎯 Best Prediction:")
        print(f"   Class ID: {result['prediction']['class_id']}")
        print(f"   Class Name: {result['prediction']['class_name']}")
        print(f"   Confidence: {result['prediction']['confidence']:.2%}")
        
        print(f"\n📊 Top 5 Predictions:")
        for i, pred in enumerate(result['top_5_predictions'], 1):
            print(f"   {i}. {pred['class_name']:<45s} - {pred['confidence']:.2%}")
    else:
        print(f"❌ Error: {response.json()}")


def test_get_classes(base_url):
    """Test the get classes endpoint"""
    print(f"\n{'='*50}")
    print(f"Testing Get Classes Endpoint")
    print(f"{'='*50}")
    
    response = requests.get(f"{base_url}/classes")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal Classes: {result['num_classes']}")
        print(f"\nFirst 10 Classes:")
        for i in range(min(10, result['num_classes'])):
            print(f"   {i}: {result['classes'][str(i)]}")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test Traffic Sign Classification API')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                        help='API base URL')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image')
    
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"TRAFFIC SIGN API TESTER")
    print(f"{'='*50}")
    print(f"API URL: {args.url}")
    
    # Test health check
    test_health_check(args.url)
    
    # Test get classes
    test_get_classes(args.url)
    
    # Test prediction if image provided
    if args.image:
        test_prediction(args.url, args.image)
    else:
        print(f"\n⚠️  No test image provided. Skipping prediction test.")
        print(f"   Use --image flag to test predictions:")
        print(f"   python test_api.py --image data/Train/0/00000_00000.png")
    
    print(f"\n{'='*50}")
    print(f"✅ Testing Complete!")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()