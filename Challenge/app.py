"""
Flask Backend for Buddha AI
Main application server integrating RAG, state machine, classifier, and generator.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path

from config import Config, validate_config
from backend.rag_system import BuddhaRAG
from backend.state_machine import BuddhaStateMachine, ConversationState
from backend.classifier import UserInputClassifier
from backend.generator import BuddhaResponseGenerator

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
app.config.from_object(Config)
CORS(app)  # Enable CORS for React frontend

# Global components (initialized on first request)
rag_system = None
classifier = None
generator = None

# Session storage (in production, use Redis or database)
# For now, simple dict keyed by session_id
sessions = {}


def init_components():
    """Initialize AI components (lazy loading)."""
    global rag_system, classifier, generator

    if not validate_config():
        raise ValueError("Configuration invalid - check .env file")

    if rag_system is None:
        print("Initializing RAG system...")
        rag_system = BuddhaRAG(
            model_name=Config.EMBEDDING_MODEL,
            persist_dir=Config.CHROMA_PERSIST_DIR
        )
        rag_system.initialize_database(force_rebuild=False)

    if classifier is None:
        print("Initializing classifier...")
        classifier = UserInputClassifier(model=Config.OPENAI_MODEL_CLASSIFIER)

    if generator is None:
        print("Initializing generator...")
        generator = BuddhaResponseGenerator(model=Config.OPENAI_MODEL_GENERATOR)


def get_session(session_id: str) -> dict:
    """Get or create a session."""
    if session_id not in sessions:
        sessions[session_id] = {
            'state_machine': BuddhaStateMachine(),
            'history': []  # Conversation history
        }
    return sessions[session_id]


@app.route('/')
def index():
    """Serve the main page."""
    # For React app, serve index.html from build folder
    frontend_path = Path('frontend/build')
    if frontend_path.exists():
        return send_from_directory('frontend/build', 'index.html')
    else:
        return jsonify({
            'message': 'Buddha AI Backend Running',
            'status': 'Frontend not built yet. Run: cd frontend && npm run build'
        })


@app.route('/api/start', methods=['POST'])
def start_conversation():
    """Start a new conversation."""
    try:
        init_components()

        data = request.json
        session_id = data.get('session_id', 'default')

        # Create new session
        session = get_session(session_id)
        session['state_machine'] = BuddhaStateMachine()
        session['history'] = []

        # Get opening message
        state = session['state_machine'].get_state()
        opening_message = generator.generate_opening(state.topic)

        return jsonify({
            'success': True,
            'message': opening_message,
            'state': state.to_dict(),
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle a chat message."""
    try:
        init_components()

        data = request.json
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_input:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            }), 400

        # Get session
        session = get_session(session_id)
        state_machine = session['state_machine']
        history = session['history']

        # Get current state
        state = state_machine.get_state()

        # Step 1: Classify user input
        classification_result = classifier.classify_with_reasoning(
            user_input,
            context=state.topic
        )
        classification = classification_result['category']

        # Step 2: Update state machine
        state_changes = state_machine.update(classification)

        # Get updated state
        state = state_machine.get_state()

        # Step 3: Retrieve relevant passages via RAG
        rag_passages = rag_system.get_relevant_passages(
            topic=state.topic,
            user_input=user_input,
            n_results=Config.RAG_N_RESULTS
        )

        # Step 4: Generate response
        response = generator.generate_response(
            user_input=user_input,
            state=state.to_dict(),
            rag_passages=rag_passages,
            conversation_history=history,
            classification=classification
        )

        # Update conversation history
        history.append({'role': 'user', 'content': user_input})
        history.append({'role': 'assistant', 'content': response})

        # Keep history manageable (last 20 messages)
        if len(history) > 20:
            history = history[-20:]
        session['history'] = history

        # Prepare response
        return jsonify({
            'success': True,
            'message': response,
            'state': state.to_dict(),
            'classification': classification,
            'classification_reasoning': classification_result.get('reasoning', ''),
            'state_changes': state_changes
        })

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current conversation state."""
    try:
        session_id = request.args.get('session_id', 'default')
        session = get_session(session_id)
        state = session['state_machine'].get_state()

        return jsonify({
            'success': True,
            'state': state.to_dict()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation."""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')

        # Reset session
        if session_id in sessions:
            sessions[session_id]['state_machine'].reset()
            sessions[session_id]['history'] = []

        return jsonify({
            'success': True,
            'message': 'Conversation reset'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'components': {
            'rag': rag_system is not None,
            'classifier': classifier is not None,
            'generator': generator is not None
        }
    })


# Static file serving for images
@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images."""
    return send_from_directory(Config.IMAGES_DIR, filename)


if __name__ == '__main__':
    print("=" * 60)
    print("Buddha AI - Starting Server")
    print("=" * 60)

    # Validate config
    if not validate_config():
        print("\nERROR: Invalid configuration. Please check your .env file.")
        print("Required: OPENAI_API_KEY=sk-...")
        exit(1)

    # Run server
    port = int(os.getenv('PORT', 5000))
    print(f"\nServer running on http://localhost:{port}")
    print("=" * 60)

    app.run(
        host='0.0.0.0',
        port=port,
        debug=Config.DEBUG
    )
