from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inventory.db'
app.config['UPLOAD_FOLDER_CAMERAS'] = 'static/cameras'
app.config['UPLOAD_FOLDER_THUMBNAILS'] = 'static/thumbnails'
app.secret_key = 'your-secret-key'  # 🔐 Use a strong, random value in production
db = SQLAlchemy(app)

# Models
class Product(db.Model):
    product_id = db.Column(db.Integer, primary_key=True)
    product_name = db.Column(db.String(100), nullable=False)

class Camera(db.Model):
    camera_id = db.Column(db.String(100), primary_key=True)

class CameraProduct(db.Model):
    camera_id = db.Column(db.String(100), db.ForeignKey('camera.camera_id'), primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.product_id'), primary_key=True)
    count = db.Column(db.Integer)

# Routes
@app.route('/')
def home():
    return redirect(url_for('list_products'))

@app.route('/camera')
def list_cameras():
    cameras = db.session.query(Camera.camera_id).distinct().all()
    return render_template('camera_list.html', cameras=cameras)

@app.route('/camera/add', methods=['GET', 'POST'])
def add_camera():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_CAMERAS'], filename)
            image.save(filepath)
            camera_id = os.path.splitext(filename)[0]
            new_camera = Camera(camera_id=camera_id)
            db.session.add(new_camera)
            db.session.commit()
            return redirect(url_for('list_cameras'))
    return render_template('camera_add.html')

@app.route('/camera/<camera_id>')
def camera_details(camera_id):
    records = db.session.query(CameraProduct, Product).join(Product, Product.product_id == CameraProduct.product_id).filter(CameraProduct.camera_id == camera_id).all()
    return render_template('camera_detail.html', camera_id=camera_id, records=records)

import os
from flask import flash

@app.route('/camera/delete/<camera_id>', methods=['POST'])
def delete_camera(camera_id):
    # Query the camera to make sure it exists
    camera = db.session.query(Camera).filter_by(camera_id=camera_id).first()

    if camera:
        # Optionally, delete the image file associated with the camera
        filename = camera.camera_id + '.jpg'  # or whatever file format your images have
        filepath = os.path.join(app.config['UPLOAD_FOLDER_CAMERAS'], filename)

        if os.path.exists(filepath):
            os.remove(filepath)  # Delete the image file

        # Now delete the camera record from the database
        db.session.delete(camera)
        db.session.commit()

        flash('Camera deleted successfully!', 'success')
    else:
        flash('Camera not found!', 'error')

    return redirect(url_for('list_cameras'))


@app.route('/product')
def list_products():
    products = Product.query.all()
    return render_template('product_list.html', products=products)

@app.route('/product/add', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        product_name = request.form['product_name']
        image = request.files['product_image']
        if product_name and image:
            filename = secure_filename(image.filename)
            extension = filename.split('.')[-1]
            new_product = Product(product_name=product_name)
            db.session.add(new_product)
            db.session.commit()
            filename = str(new_product.product_id)+'.'+extension
            image.save(os.path.join(app.config['UPLOAD_FOLDER_THUMBNAILS'], filename))
            return redirect(url_for('list_products'))
    return render_template('product_add.html')

@app.route('/product/update/<int:product_id>', methods=['GET', 'POST'])
def update_product(product_id):
    product = Product.query.get_or_404(product_id)
    if request.method == 'POST':
        product_name = request.form['product_name']
        image = request.files.get('product_image')

        if product_name:
            product.product_name = product_name

        if image and image.filename:
            filename = secure_filename(image.filename)
            extension = filename.split('.')[-1]
            filename = str(product.product_id)+'.'+extension
            image.save(os.path.join(app.config['UPLOAD_FOLDER_THUMBNAILS'], filename))

        db.session.commit()
        flash('Product updated successfully!')
        return redirect(url_for('list_products'))

    return render_template('product_add.html', product=product)

@app.route('/product/delete/<int:product_id>', methods=['POST'])
def delete_product(product_id):
    product = Product.query.get_or_404(product_id)
    db.session.delete(product)
    db.session.commit()
    flash('Product deleted successfully!')
    return redirect(url_for('list_products'))


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER_CAMERAS'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_THUMBNAILS'], exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
