import io
import torch
import warnings
from torchvision import transforms, models
from PIL import Image, ImageStat, ImageEnhance
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

warnings.filterwarnings('ignore')

# Initialize Flask app and security configurations
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure CORS
CORS(app, resources={
    r"/predict": {
        "origins": ["http://localhost:5173", "https://*.madhan.xyz"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

# Global variables for model and device
model = None
device = None
transform = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Initialize model and required transformations"""
    global model, device, transform

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('./models/chest_xray_resnet_model.pth'))
    model = model.to(device)
    model.eval()

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    if torch.cuda.is_available():
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        print('Using CPU')

# Add these functions after the allowed_file function and before load_model function


def is_likely_xray(image):
    """
    Check if the image is likely to be an X-ray by analyzing its characteristics
    Returns: (bool, str) - (is_xray, error_message)
    """
    try:
        # Convert to grayscale for analysis
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image

        # Get image statistics
        stats = ImageStat.Stat(gray_image)

        # Check image properties
        mean_brightness = stats.mean[0]
        # Standard deviation as a measure of contrast
        contrast = stats.var[0] ** 0.5

        # X-ray specific checks
        is_xray = True
        reason = ""

        # Check 1: X-rays typically have medium brightness
        if mean_brightness < 30 or mean_brightness > 225:
            is_xray = False
            reason = "Image brightness is not typical for X-rays"

        # Check 2: X-rays should have decent contrast
        if contrast < 20:
            is_xray = False
            reason = "Image contrast is too low for an X-ray"

        # Check 3: Check if image is mostly grayscale
        if image.mode == 'RGB':
            r, g, b = image.split()
            if abs(ImageStat.Stat(r).mean[0] - ImageStat.Stat(g).mean[0]) > 20 or \
               abs(ImageStat.Stat(g).mean[0] - ImageStat.Stat(b).mean[0]) > 20 or \
               abs(ImageStat.Stat(b).mean[0] - ImageStat.Stat(r).mean[0]) > 20:
                is_xray = False
                reason = "Image contains too much color variation for an X-ray"

        return is_xray, reason if not is_xray else None

    except Exception as e:
        return False, f"Error analyzing image: {str(e)}"


def predict(image):
    """Make prediction on the input image"""
    try:
        # Apply transformations
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)

        # Get probabilities
        probs = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(probs, 1)

        # Get prediction details
        class_names = ['NORMAL', 'PNEUMONIA']
        predicted_label = class_names[predicted_class.item()]
        # Convert to float for JSON serialization
        predicted_prob = float(probs[0, predicted_class].item())

        return True, predicted_label, predicted_prob, None

    except Exception as e:
        return False, None, None, str(e)


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to Pneumonia Detection API'
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'alive'
    })


@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict_route():
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG'
            }), 400

        # Process image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        is_xray, error_message = is_likely_xray(img)
        if not is_xray:
            return jsonify({
                'success': False,
                'error': f'Invalid image. Please upload an X-ray image.',
                'reason':  error_message
            }), 400

        success, label, prob, error = predict(img)

        if not success:
            return jsonify({
                'success': False,
                'error': f'Prediction error,Please try again later.',
                'reason': error
            }), 500

        return jsonify({
            'success': True,
            'label': label,
            'probability': prob
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error,Please try again later!.',
            'reason': error
        }), 500
# Error handlers


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File is too large. Maximum size is 5MB'
    }), 413


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'success': False,
        'error': 'Rate limit exceeded. Please try again later.'
    }), 429


# Initialize model when starting the server
load_model()

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000, debug=False)
