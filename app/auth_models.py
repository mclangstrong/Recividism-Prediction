from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Add to database.py after PDL model

class Officer(UserMixin, db.Model):
    """Model for BJMP officer/user accounts."""
    __tablename__ = 'officers'
    
    id = db.Column(db.Integer, primary_key=True)
    personnel_id = db.Column(db.String(50), unique=True, nullable=False)  # e.g., BJMP-2024-001
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Personal Information
    full_name = db.Column(db.String(200), nullable=False)
    rank = db.Column(db.String(100))  # e.g., Officer, Senior Officer, Chief
    email = db.Column(db.String(200))
    
    # Account Status
    role = db.Column(db.String(20), default='officer')  # officer, admin
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password matches hash."""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'personnel_id': self.personnel_id,
            'username': self.username,
            'full_name': self.full_name,
            'rank': self.rank,
            'role': self.role,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<Officer {self.personnel_id}: {self.full_name}>'


def create_default_admin():
    """Create default admin account if none exists."""
    admin = Officer.query.filter_by(role='admin').first()
    if not admin:
        default_admin = Officer(
            personnel_id='BJMP-ADMIN-001',
            username='admin',
            full_name='System Administrator',
            rank='Administrator',
            role='admin',
            email='admin@bjmp.gov.ph'
        )
        default_admin.set_password('admin123')  # Change this in production!
        db.session.add(default_admin)
        db.session.commit()
        print("✓ Default admin account created (username: admin, password: admin123)")
        return default_admin
    return admin
