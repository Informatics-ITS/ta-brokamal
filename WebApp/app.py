import os
import uuid
import json
import base64
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
from flask_cors import CORS
import PIL.Image

app = Flask(__name__)
CORS(app)  
app.secret_key = 'development-secret-key'  
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['VISUALIZATIONS_FOLDER'] = 'static/visualizations'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['VISUALIZATIONS_FOLDER'], exist_ok=True)

roboflow_client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="roboflow API key" # add your own API key
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def save_base64_image(base64_data, save_path):
    try:
        # Remove the data URL prefix if present
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Decode and save the image
        with open(save_path, 'wb') as f:
            f.write(base64.b64decode(base64_data))
        
        return True
    except Exception as e:
        print(f"Error saving base64 image: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text-inference')
def text_inference():
    return render_template('text_inference.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Accept multiple files: file0, file1, ...
    files = [f for key, f in request.files.items() if key.startswith('file') and allowed_file(f.filename)]
    if not files:
        flash('No valid files uploaded')
        return redirect(request.url)

    result_id = str(uuid.uuid4())
    all_results = []
    all_visualized_images = []
    all_filenames = []

    for idx, file in enumerate(files):
        filename = f"{result_id}_{idx+1}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        all_filenames.append(filename)

        try:
            result = roboflow_client.run_workflow(
                workspace_name="grind-space",
                workflow_id="detect-count-and-visualize",
                images={
                    "image": filepath
                },
                use_cache=True
            )
            all_results.append(result)

            # Visualized image
            local_visualized_image = None
            if isinstance(result, list) and len(result) > 0:
                result_item = result[0]
                if 'output_image' in result_item and result_item['output_image']:
                    visualization_filename = f"{result_id}_{idx+1}_visualization.jpg"
                    local_visualization_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], visualization_filename)
                    if save_base64_image(result_item['output_image'], local_visualization_path):
                        local_visualized_image = url_for('static', filename=f'visualizations/{visualization_filename}')
            else:
                if 'output_image' in result and result['output_image']:
                    visualization_filename = f"{result_id}_{idx+1}_visualization.jpg"
                    local_visualization_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], visualization_filename)
                    if save_base64_image(result['output_image'], local_visualization_path):
                        local_visualized_image = url_for('static', filename=f'visualizations/{visualization_filename}')
            all_visualized_images.append(local_visualized_image)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)

    results_data = {
        'original_images': all_filenames,
        'detection_results': all_results,
        'visualized_images': all_visualized_images
    }
    results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    with open(results_file, 'w') as f:
        json.dump(results_data, f)
    session['result_id'] = result_id
    return redirect(url_for('results'))

@app.route('/results')
def results():
    if 'result_id' not in session:
        flash('No results found. Please upload an image first.')
        return redirect(url_for('index'))
    
    result_id = session['result_id']
    results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    
    if not os.path.exists(results_file):
        flash('Results not found. Please try again.')
        return redirect(url_for('index'))
    
    # Load results from file
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return render_template('results.html', 
                          results=results,
                          original_images=results['original_images'])

@app.route('/send-to-inference', methods=['POST'])
def send_to_inference():
    try:
        # Get the classes text from the request
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        classes_text = data['text']
        
        # Send to inference endpoint
        response = requests.post(
            'http://127.0.0.1:9000/transliterate',
            headers={'Content-Type': 'application/json'},
            json={'text': classes_text}
        )
        # Ensure the response is valid JSON
        try:
            result = response.json()
            return jsonify(result), response.status_code
        except ValueError as e:
            return jsonify({
                'error': 'Invalid JSON response from API',
                'response_text': response.text
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/proxy/inference', methods=['POST'])
def proxy_inference():
    try:
        # Get the request data
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        
        # Forward the request to the target API
        response = requests.post(
            'http://127.0.0.1:9000/transliterate',
            headers={'Content-Type': 'application/json'},
            json={'text': text}
        )
        
        try:
            result = response.json()
            return jsonify(result), response.status_code
        except ValueError as e:
            return jsonify({
                'error': 'Invalid JSON response from API',
                'response_text': response.text
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
