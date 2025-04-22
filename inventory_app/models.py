from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Product(db.Model):
    product_id = db.Column(db.Integer, primary_key=True)
    product_name = db.Column(db.String(100), nullable=False)
    inventory = db.relationship('Inventory', backref='product', lazy=True)

class Inventory(db.Model):
    __tablename__ = 'inventory'
    image_id = db.Column(db.String(100), primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.product_id'), primary_key=True)
    product_count = db.Column(db.Integer, default=1)

    product = db.relationship('Product', backref=db.backref('inventory_items', lazy=True))
