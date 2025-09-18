# main.py
import threading
import webview
from app import app   # this imports your Flask app object

def run_flask():
    # Run Flask on a different port to avoid conflicts
    app.run(host="127.0.0.1", port=5002, debug=False, use_reloader=False)

if __name__ == "__main__":
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Create a desktop window showing the Flask app
    webview.create_window("Pill Annotation Tool", "http://127.0.0.1:5002")
    webview.start()
