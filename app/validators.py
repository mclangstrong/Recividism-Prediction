"""
Input validators for the Recidivism Prediction System.
Provides server-side validation for prediction data to ensure data integrity.
"""
from typing import Dict, List, Tuple, Any, Optional


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


# Valid options for categorical fields
VALID_GENDERS = ['Male', 'Female']
VALID_CIVIL_STATUS = ['Single', 'Married', 'Widowed', 'Separated']
VALID_EDUCATION = ['None', 'Elementary', 'High School', 'College', 'Vocational', 'Graduate']
VALID_OFFENSE_TYPES = ['Theft', 'Robbery', 'Homicide', 'Assault', 'Illegal Drugs', 'Estafa', 'Fraud']
VALID_RELIGIONS = ['Catholic', 'Protestant', 'Islam', 'Other', 'None']
VALID_YES_NO = ['Yes', 'No']
VALID_FAMILY_SUPPORT = ['Strong', 'Moderate', 'Poor', 'Absent']
VALID_JAIL_BEHAVIOR = ['Good', 'Fair', 'Poor']
VALID_AGGRESSION = ['Low', 'Moderate', 'High']
VALID_REHAB_ATTITUDE = ['Cooperative', 'Uncooperative', 'Indifferent']
VALID_REHAB_STATUS = ['Not Applicable', 'Not Started', 'In Progress', 'Completed']


def validate_required(data: Dict, field: str, field_label: str = None) -> Optional[str]:
    """Validate that a required field is present and not empty."""
    label = field_label or field
    value = data.get(field)
    if value is None or (isinstance(value, str) and value.strip() == ''):
        return f"{label} is required"
    return None


def validate_string_length(value: str, min_len: int = 0, max_len: int = 200, field_label: str = "Field") -> Optional[str]:
    """Validate string length."""
    if not isinstance(value, str):
        return f"{field_label} must be a string"
    if len(value) < min_len:
        return f"{field_label} must be at least {min_len} characters"
    if len(value) > max_len:
        return f"{field_label} must be no more than {max_len} characters"
    return None


def validate_integer_range(value: Any, min_val: int, max_val: int, field_label: str = "Field") -> Optional[str]:
    """Validate integer is within range."""
    try:
        int_val = int(value)
        if int_val < min_val or int_val > max_val:
            return f"{field_label} must be between {min_val} and {max_val}"
    except (ValueError, TypeError):
        return f"{field_label} must be a valid number"
    return None


def validate_float_range(value: Any, min_val: float, max_val: float, field_label: str = "Field") -> Optional[str]:
    """Validate float is within range."""
    try:
        float_val = float(value)
        if float_val < min_val or float_val > max_val:
            return f"{field_label} must be between {min_val} and {max_val}"
    except (ValueError, TypeError):
        return f"{field_label} must be a valid number"
    return None


def validate_option(value: str, valid_options: List[str], field_label: str = "Field") -> Optional[str]:
    """Validate that value is one of the valid options."""
    if value and value not in valid_options:
        return f"Invalid {field_label}. Must be one of: {', '.join(valid_options)}"
    return None


def validate_prediction_data(data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate all prediction input data.
    
    Args:
        data: Dictionary of prediction form data
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields validation
    required_error = validate_required(data, 'Name', 'Full Name')
    if required_error:
        errors.append(required_error)
    else:
        # Validate name length
        name_error = validate_string_length(data.get('Name', ''), min_len=2, max_len=200, field_label='Full Name')
        if name_error:
            errors.append(name_error)
    
    # Age validation (required, 18-100)
    age_error = validate_required(data, 'Age', 'Age')
    if age_error:
        errors.append(age_error)
    else:
        age_range_error = validate_integer_range(data.get('Age'), 18, 100, 'Age')
        if age_range_error:
            errors.append(age_range_error)
    
    # Gender validation (required)
    gender_error = validate_required(data, 'Gender', 'Gender')
    if gender_error:
        errors.append(gender_error)
    else:
        gender_option_error = validate_option(data.get('Gender'), VALID_GENDERS, 'Gender')
        if gender_option_error:
            errors.append(gender_option_error)
    
    # Civil Status (optional but must be valid if provided)
    if data.get('Civil Status'):
        civil_error = validate_option(data.get('Civil Status'), VALID_CIVIL_STATUS, 'Civil Status')
        if civil_error:
            errors.append(civil_error)
    
    # Educational Attainment (optional but must be valid)
    if data.get('Educational Attainment'):
        edu_error = validate_option(data.get('Educational Attainment'), VALID_EDUCATION, 'Educational Attainment')
        if edu_error:
            errors.append(edu_error)
    
    # Offense Type (optional but must be valid)
    if data.get('Offense Type'):
        offense_error = validate_option(data.get('Offense Type'), VALID_OFFENSE_TYPES, 'Offense Type')
        if offense_error:
            errors.append(offense_error)
    
    # Religion (optional but must be valid)
    if data.get('Religion'):
        religion_error = validate_option(data.get('Religion'), VALID_RELIGIONS, 'Religion')
        if religion_error:
            errors.append(religion_error)
    
    # Prior Convictions (0-50)
    if data.get('Prior_Convictions') is not None and data.get('Prior_Convictions') != '':
        prior_error = validate_integer_range(data.get('Prior_Convictions'), 0, 50, 'Prior Convictions')
        if prior_error:
            errors.append(prior_error)
    
    # Infractions Count (0-100)
    if data.get('Infractions Count') is not None and data.get('Infractions Count') != '':
        inf_error = validate_integer_range(data.get('Infractions Count'), 0, 100, 'Infractions Count')
        if inf_error:
            errors.append(inf_error)
    
    # Sentence Length (0-100 years)
    if data.get('Length of Current Sentence (yrs)') is not None and data.get('Length of Current Sentence (yrs)') != '':
        sentence_error = validate_float_range(data.get('Length of Current Sentence (yrs)'), 0, 100, 'Sentence Length')
        if sentence_error:
            errors.append(sentence_error)
    
    # Time Served (0-100 years)
    if data.get('Time Served (years)') is not None and data.get('Time Served (years)') != '':
        time_error = validate_float_range(data.get('Time Served (years)'), 0, 100, 'Time Served')
        if time_error:
            errors.append(time_error)
    
    # Yes/No fields
    yes_no_fields = [
        ('Substance Abuse History', 'Substance Abuse History'),
        ('Mental Health Issues', 'Mental Health Issues'),
        ('Gang Affiliation', 'Gang Affiliation'),
        ('Program_Participation', 'Program Participation'),
        ('Homelessness', 'Homelessness'),
        ('Peer Influence', 'Peer Influence'),
        ('Vocational Training', 'Vocational Training'),
        ('Therapy Attendance', 'Therapy Attendance'),
    ]
    
    for field, label in yes_no_fields:
        if data.get(field):
            yn_error = validate_option(data.get(field), VALID_YES_NO, label)
            if yn_error:
                errors.append(yn_error)
    
    # Family Support
    if data.get('Family Support'):
        family_error = validate_option(data.get('Family Support'), VALID_FAMILY_SUPPORT, 'Family Support')
        if family_error:
            errors.append(family_error)
    
    # Jail Behavior Rating
    if data.get('Jail Behavior Rating'):
        behavior_error = validate_option(data.get('Jail Behavior Rating'), VALID_JAIL_BEHAVIOR, 'Jail Behavior Rating')
        if behavior_error:
            errors.append(behavior_error)
    
    # Aggression
    if data.get('Aggression'):
        aggression_error = validate_option(data.get('Aggression'), VALID_AGGRESSION, 'Aggression')
        if aggression_error:
            errors.append(aggression_error)
    
    # Rehab Attitude
    if data.get('Rehab Attitude'):
        attitude_error = validate_option(data.get('Rehab Attitude'), VALID_REHAB_ATTITUDE, 'Rehab Attitude')
        if attitude_error:
            errors.append(attitude_error)
    
    # Rehab Completed Status
    if data.get('Rehab_Completed'):
        status_error = validate_option(data.get('Rehab_Completed'), VALID_REHAB_STATUS, 'Rehabilitation Status')
        if status_error:
            errors.append(status_error)
    
    return (len(errors) == 0, errors)


def sanitize_string(value: str) -> str:
    """Sanitize string input by stripping whitespace and limiting length."""
    if not isinstance(value, str):
        return str(value) if value is not None else ''
    return value.strip()[:500]  # Limit to 500 chars for safety


def sanitize_prediction_data(data: Dict) -> Dict:
    """
    Sanitize prediction data before processing.
    
    Args:
        data: Raw prediction form data
        
    Returns:
        Sanitized data dictionary
    """
    sanitized = {}
    
    # String fields
    string_fields = ['Name', 'Gender', 'Civil Status', 'Educational Attainment', 
                     'Offense Type', 'Religion', 'Substance Abuse History',
                     'Mental Health Issues', 'Family Support', 'Gang Affiliation',
                     'Employment Status', 'Program_Participation', 'Homelessness',
                     'Peer Influence', 'Jail Behavior Rating', 'Aggression',
                     'Remorse', 'Vocational Training', 'Therapy Attendance',
                     'Rehab Attitude', 'Rehab_Completed', 'Job History',
                     'Post-Release Housing', 'Juvenile Records']
    
    for field in string_fields:
        if field in data:
            sanitized[field] = sanitize_string(data.get(field, ''))
    
    # Integer fields
    int_fields = ['Age', 'Prior_Convictions', 'Infractions Count', 'Behavior Score']
    for field in int_fields:
        try:
            if field in data and data[field] not in (None, ''):
                sanitized[field] = int(data[field])
            elif field in data:
                sanitized[field] = 0
        except (ValueError, TypeError):
            sanitized[field] = 0
    
    # Float fields
    float_fields = ['Length of Current Sentence (yrs)', 'Time Served (years)']
    for field in float_fields:
        try:
            if field in data and data[field] not in (None, ''):
                sanitized[field] = float(data[field])
            elif field in data:
                sanitized[field] = 0.0
        except (ValueError, TypeError):
            sanitized[field] = 0.0
    
    return sanitized
