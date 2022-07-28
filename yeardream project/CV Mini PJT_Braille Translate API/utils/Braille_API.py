from genericpath import isfile
import os
from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import transfer


UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = {'png'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def flask():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.secret_key = 'secret braille'

    @app.route('/')
    @app.route('/braille', methods=['GET','POST'])
    def string_out():
        if request.method == 'POST':
            upload = request.files['file']
            if upload.filename == '':
                flash('No image selected for uploading')
                return redirect(request.url)
            if upload and allowed_file(upload.filename):
                upload_name = secure_filename(upload.filename)
                upload.save(os.path.join(app.config['UPLOAD_FOLDER'], upload_name))
                #print('upload_image filename: ' + filename)
                flash('Image successfully uploaded and displayed below')

                string = transfer.img_to_s(os.path.join(app.config['UPLOAD_FOLDER'], upload_name))
                return render_template('braille.html', output=string)
            else:
                flash('Allowed image types are -> png')
                return redirect(request.url)
        else:
            return render_template('braille.html', output = 'input image')

    @app.route('/')
    @app.route('/braille_out',methods=['GET','POST'])
    def braille_out(image=None):
        if request.method == 'POST':
            word = request.form['word']
            # print(type(word))
            transfer.s_to_img(word)
            image = f'{word}.png'
            return render_template('braille.html', output2= image)
            
        else:
            return render_template('braille.html', output = 'input image')

    # if __name__ == "__main__":
    #     app.run(host='0.0.0.0', port=5001)
    app.run(host='0.0.0.0', port=5001)

