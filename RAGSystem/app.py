from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime

# Import our RAG system
from rag_system import RAGSystem

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize RAG system
# rag = RAGSystem(api_key=os.environ.get("OPENAI_API_KEY"))
rag = RAGSystem()


# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Session storage (would use a database in production)
sessions = {}
uploaded_files = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate secure filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Store file info
        file_info = {
            'original_name': filename,
            'stored_name': unique_filename,
            'path': file_path,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'size': os.path.getsize(file_path)
        }
        
        uploaded_files.append(file_info)
        
        # Process the document with RAG system
        try:
            chunk_count = rag.add_document(file_path, {"original_name": filename})
            
            # Update file info with chunk count
            file_info['chunks'] = chunk_count
            
            return jsonify({
                'success': True,
                'filename': filename,
                'chunks_processed': chunk_count
            })
        except Exception as e:
            # Remove file if processing fails
            os.remove(file_path)
            uploaded_files.pop()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    
    # Check for required fields
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
    
    # Get session ID or create new one
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    # Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = {
            'history': [],
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Get query
    query = data['query']
    
    # Generate response
    try:
        response = rag.generate_response(query)
        
        # Update session history
        sessions[session_id]['history'].append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify({
        'success': True,
        'history': sessions[session_id]['history']
    })

@app.route('/files', methods=['GET'])
def get_files():
    return jsonify({
        'success': True,
        'files': uploaded_files
    })

@app.route('/sessions', methods=['GET'])
def get_sessions():
    # Create a simplified view of sessions
    session_list = []
    for session_id, session_data in sessions.items():
        session_list.append({
            'id': session_id,
            'created_at': session_data['created_at'],
            'message_count': len(session_data['history'])
        })
    
    return jsonify({
        'success': True,
        'sessions': session_list
    })

if __name__ == '__main__':
    app.run(debug=True)