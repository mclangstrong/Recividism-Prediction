"""
Database models and initialization for the Recidivism Prediction System.
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class PDL(db.Model):
    """Model for storing PDL (Person Deprived of Liberty) records."""
    __tablename__ = 'pdl'
    
    id = db.Column(db.Integer, primary_key=True)
    pdl_id = db.Column(db.String(50), unique=True, nullable=False)
    
    # Personal Information
    name = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    civil_status = db.Column(db.String(50))
    religion = db.Column(db.String(100))
    education = db.Column(db.String(100))
    employment = db.Column(db.String(100))
    
    # Criminal Information
    index_crime = db.Column(db.String(200))
    prior_convictions = db.Column(db.Integer, default=0)
    infractions_count = db.Column(db.Integer, default=0)
    sentence_length = db.Column(db.Float)
    time_served = db.Column(db.Float)
    admission_date = db.Column(db.DateTime)
    
    # Risk Factors
    substance_abuse = db.Column(db.String(50))
    mental_health = db.Column(db.String(50))
    family_support = db.Column(db.String(50))
    gang_affiliation = db.Column(db.String(50))
    program_participation = db.Column(db.String(50))
    
    # Current Status
    latest_risk_level = db.Column(db.String(20))
    latest_assessment_date = db.Column(db.DateTime)
    latest_probability = db.Column(db.Float)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='pdl_record', lazy=True,
                                 foreign_keys='Prediction.pdl_id')
    
    def to_dict(self):
        return {
            'id': self.id,
            'pdl_id': self.pdl_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'civil_status': self.civil_status,
            'religion': self.religion,
            'education': self.education,
            'employment': self.employment,
            'index_crime': self.index_crime,
            'prior_convictions': self.prior_convictions,
            'infractions_count': self.infractions_count,
            'sentence_length': self.sentence_length,
            'time_served': self.time_served,
            'substance_abuse': self.substance_abuse,
            'mental_health': self.mental_health,
            'family_support': self.family_support,
            'gang_affiliation': self.gang_affiliation,
            'program_participation': self.program_participation,
            'admission_date': self.admission_date.isoformat() if self.admission_date else None,
            'latest_risk_level': self.latest_risk_level,
            'latest_assessment_date': self.latest_assessment_date.isoformat() if self.latest_assessment_date else None,
            'latest_probability': self.latest_probability,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Officer(db.Model):
    """Model for BJMP officer/user accounts."""
    __tablename__ = 'officers'
    
    id = db.Column(db.Integer, primary_key=True)
    personnel_id = db.Column(db.String(50), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Personal Information
    full_name = db.Column(db.String(200), nullable=False)
    rank = db.Column(db.String(100))
    email = db.Column(db.String(200))
    
    # Account Status
    role = db.Column(db.String(20), default='officer')
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
            'is_active': self.is_active
        }
    
    # Flask-Login integration
    @property
    def is_authenticated(self):
        return True
    
    @property
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)

class Prediction(db.Model):
    """Model for storing prediction records."""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Link to PDL record
    pdl_id = db.Column(db.String(50), db.ForeignKey('pdl.pdl_id'), nullable=True)
    
    # Personal Information
    name = db.Column(db.String(200))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    civil_status = db.Column(db.String(50))
    religion = db.Column(db.String(100))
    education = db.Column(db.String(100))
    employment = db.Column(db.String(100))
    
    # Criminal History
    prior_convictions = db.Column(db.Integer)
    offense_type = db.Column(db.String(100))
    sentence_length =db.Column(db.Float)
    time_served = db.Column(db.Float)
    
    # Behavioral Factors
    substance_abuse = db.Column(db.String(10))
    mental_health = db.Column(db.String(10))
    family_support = db.Column(db.String(50))
    gang_affiliation = db.Column(db.String(10))
    
    # Program Data
    program_participation = db.Column(db.String(10))
    behavior_score = db.Column(db.Integer)
    
    # Prediction Results
    prediction = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    
    # Hybrid RNR Data
    ml_probability = db.Column(db.Float, nullable=True)
    rnr_probability = db.Column(db.Float, nullable=True)
    rnr_breakdown = db.Column(db.Text, nullable=True) # JSON string
    
    # Rehabilitation Plan (JSON string)
    rehabilitation_plan = db.Column(db.Text, nullable=True)
    
    # Officer who performed assessment (link to Officer)
    assessed_by = db.Column(db.Integer, db.ForeignKey('officers.id'), nullable=True)
    
    # Model Version Tracking
    model_version = db.Column(db.String(50), default='1.0', nullable=True)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.name} - {self.risk_level}>'
    
    def to_dict(self):
        """Convert prediction to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'name': self.name or 'Unknown',
            'age': self.age,
            'gender': self.gender,
            'civil_status': self.civil_status,
            'religion': self.religion,
            'education': self.education,
            'employment': self.employment,
            'prior_convictions': self.prior_convictions,
            'offense_type': self.offense_type,
            'sentence_length': self.sentence_length,
            'time_served': self.time_served,
            'substance_abuse': self.substance_abuse,
            'mental_health': self.mental_health,
            'family_support': self.family_support,
            'gang_affiliation': self.gang_affiliation,
            'program_participation': self.program_participation,
            'behavior_score': self.behavior_score,
            'prediction': self.prediction,
            'probability': self.probability,
            'risk_level': self.risk_level,
            'ml_probability': self.ml_probability,
            'rnr_probability': self.rnr_probability,
            'rnr_breakdown': json.loads(self.rnr_breakdown) if self.rnr_breakdown else []
        }

def init_db(app):
    """Initialize database with app context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")
        create_default_admin()

def create_default_admin():
    """Create default admin account if none exists."""
    try:
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
            default_admin.set_password('admin123')
            db.session.add(default_admin)
            db.session.commit()
            print("✓ Default admin account created (username: admin, password: admin123)")
    except Exception as e:
        print(f"Note: {e}")

def create_or_update_pdl(prediction_data, prediction_result):
    """
    Create or update PDL record from prediction data.
    Links the prediction to the PDL record.
    """
    name = prediction_data.get('Name', 'Unknown')
    
    # Generate PDL ID if name is provided
    if name and name != 'Unknown':
        # Simple PDL ID generation: PDL-YYYY-XXXX
        year = datetime.utcnow().year
        # Count existing PDLs this year
        count = PDL.query.filter(PDL.pdl_id.like(f'PDL-{year}-%')).count()
        pdl_id_num = f'PDL-{year}-{str(count + 1).zfill(4)}'
        
        # Check if PDL with this name already exists
        existing_pdl = PDL.query.filter_by(name=name).first()
        
        if existing_pdl:
            # Update existing PDL with ALL edited profile data
            if prediction_data.get('Age'):
                existing_pdl.age = prediction_data.get('Age')
            if prediction_data.get('Gender'):
                existing_pdl.gender = prediction_data.get('Gender')
            if prediction_data.get('Civil Status'):
                existing_pdl.civil_status = prediction_data.get('Civil Status')
            if prediction_data.get('Religion'):
                existing_pdl.religion = prediction_data.get('Religion')
            if prediction_data.get('Educational Attainment'):
                existing_pdl.education = prediction_data.get('Educational Attainment')
            if prediction_data.get('Employment Status'):
                existing_pdl.employment = prediction_data.get('Employment Status')
            if prediction_data.get('Offense Type'):
                existing_pdl.index_crime = prediction_data.get('Offense Type')
            if prediction_data.get('Prior_Convictions') is not None:
                existing_pdl.prior_convictions = int(prediction_data.get('Prior_Convictions', 0))
            if prediction_data.get('Infractions Count') is not None:
                existing_pdl.infractions_count = int(prediction_data.get('Infractions Count', 0))
            if prediction_data.get('Length of Current Sentence (yrs)') is not None:
                existing_pdl.sentence_length = float(prediction_data.get('Length of Current Sentence (yrs)', 0))
            if prediction_data.get('Time Served (years)') is not None:
                existing_pdl.time_served = float(prediction_data.get('Time Served (years)', 0))
            if prediction_data.get('Substance Abuse History'):
                existing_pdl.substance_abuse = prediction_data.get('Substance Abuse History')
            if prediction_data.get('Mental Health Issues'):
                existing_pdl.mental_health = prediction_data.get('Mental Health Issues')
            if prediction_data.get('Family Support'):
                existing_pdl.family_support = prediction_data.get('Family Support')
            if prediction_data.get('Gang Affiliation'):
                existing_pdl.gang_affiliation = prediction_data.get('Gang Affiliation')
            if prediction_data.get('Program_Participation'):
                existing_pdl.program_participation = prediction_data.get('Program_Participation')
            
            existing_pdl.latest_risk_level = prediction_result['risk_level']
            existing_pdl.latest_probability = prediction_result['probability']
            existing_pdl.latest_assessment_date = datetime.utcnow()
            existing_pdl.updated_at = datetime.utcnow()
            db.session.commit()
            return existing_pdl.pdl_id
        else:
            # Create new PDL record with ALL fields
            new_pdl = PDL(
                pdl_id=pdl_id_num,
                name=name,
                age=prediction_data.get('Age'),
                gender=prediction_data.get('Gender'),
                civil_status=prediction_data.get('Civil Status'),
                religion=prediction_data.get('Religion'),
                education=prediction_data.get('Educational Attainment'),
                employment=prediction_data.get('Employment Status'),
                index_crime=prediction_data.get('Offense Type'),
                prior_convictions=int(prediction_data.get('Prior_Convictions', 0)) if prediction_data.get('Prior_Convictions') else 0,
                infractions_count=int(prediction_data.get('Infractions Count', 0)) if prediction_data.get('Infractions Count') else 0,
                sentence_length=float(prediction_data.get('Length of Current Sentence (yrs)', 0)) if prediction_data.get('Length of Current Sentence (yrs)') else 0,
                time_served=float(prediction_data.get('Time Served (years)', 0)) if prediction_data.get('Time Served (years)') else 0,
                substance_abuse=prediction_data.get('Substance Abuse History'),
                mental_health=prediction_data.get('Mental Health Issues'),
                family_support=prediction_data.get('Family Support'),
                gang_affiliation=prediction_data.get('Gang Affiliation'),
                program_participation=prediction_data.get('Program_Participation'),
                admission_date=datetime.utcnow(),
                latest_risk_level=prediction_result['risk_level'],
                latest_probability=prediction_result['probability'],
                latest_assessment_date=datetime.utcnow()
            )
            db.session.add(new_pdl)
            db.session.commit()
            return pdl_id_num
    
    return None


class AuditLog(db.Model):
    """Model for tracking all critical system operations."""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Action Details
    action_type = db.Column(db.String(50), nullable=False)  # 'prediction', 'plan_update', 'login', etc.
    action_description = db.Column(db.Text, nullable=True)
    
    # User Information
    user_id = db.Column(db.Integer, db.ForeignKey('officers.id'), nullable=True)
    username = db.Column(db.String(100), nullable=True)
    
    # Related Records
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=True)
    pdl_id = db.Column(db.String(50), nullable=True)
    
    # Technical Details
    model_version = db.Column(db.String(50), nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    
    # Additional Data (JSON string)
    extra_data = db.Column(db.Text, nullable=True)  # Changed from 'metadata' (reserved keyword)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'action_type': self.action_type,
            'action_description': self.action_description,
            'username': self.username,
            'pdl_id': self.pdl_id,
            'model_version': self.model_version
        }


class Notification(db.Model):
    """Model for in-app notifications (NO email for security)."""
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Notification Details
    notification_type = db.Column(db.String(50), nullable=False)  # 'high_risk', 'system', etc.
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    
    # Target User
    user_id = db.Column(db.Integer, db.ForeignKey('officers.id'), nullable=True)
    
    # Related Records
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=True)
    pdl_id = db.Column(db.String(50), nullable=True)
    
    # Status
    is_read = db.Column(db.Boolean, default=False, nullable=False)
    read_at = db.Column(db.DateTime, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'type': self.notification_type,
            'title': self.title,
            'message': self.message,
            'is_read': self.is_read,
            'read_at': self.read_at.isoformat() if self.read_at else None,
            'prediction_id': self.prediction_id,
            'pdl_id': self.pdl_id
        }


class ModelPerformanceLog(db.Model):
    """Model for tracking prediction accuracy over time."""
    __tablename__ = 'model_performance_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Prediction Reference
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    
    # Prediction Details
    predicted_risk_level = db.Column(db.String(20), nullable=False)
    predicted_probability = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(50), nullable=True)
    
    # Actual Outcome (to be updated later)
    actual_recidivism = db.Column(db.Integer, nullable=True)  # 0 or 1, updated after release
    outcome_recorded_date = db.Column(db.DateTime, nullable=True)
    
    # Accuracy Metrics
    was_accurate = db.Column(db.Boolean, nullable=True)  # Calculated after actual outcome known
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'prediction_id': self.prediction_id,
            'predicted_risk_level': self.predicted_risk_level,
            'predicted_probability': self.predicted_probability,
            'model_version': self.model_version,
            'actual_recidivism': self.actual_recidivism,
            'was_accurate': self.was_accurate
        }
