from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('images')
        for file in files:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('gallery'))
    return render_template('index.html')

@app.route('/gallery')
def gallery():
    image_files = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    image_paths = [os.path.join('uploads', file) for file in image_files]
    return render_template('gallery.html', images=image_paths)

@app.route('/view/<int:index>')
def view_image(index):
    image_files = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    image_paths = [os.path.join('uploads', file) for file in image_files]
    total = len(image_paths)

    if total == 0:
        return "No images uploaded."

    index = max(0, min(index, total - 1))
    return render_template(
        'view.html',
        image=image_paths[index],
        index=index,
        total=total
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
