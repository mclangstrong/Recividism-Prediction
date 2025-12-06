// Assessment Wizard JavaScript
let currentStep = 1;
const totalSteps = 3;

function nextStep() {
    if (currentStep < totalSteps) {
        const currentStepEl = document.getElementById(`step${currentStep}`);
        const inputs = currentStepEl.querySelectorAll('input[required], select[required]');
        let valid = true;

        inputs.forEach(input => {
            if (!input.value) {
                input.classList.add('invalid');
                valid = false;
            } else {
                input.classList.remove('invalid');
            }
        });

        if (!valid) {
            alert('Please fill in all required fields');
            return;
        }

        if (currentStep === 2) {
            generateReview();
        }

        document.getElementById(`step${currentStep}`).style.display = 'none';
        document.getElementById(`step${currentStep}-indicator`).classList.remove('active');

        currentStep++;
        document.getElementById(`step${currentStep}`).style.display = 'block';
        document.getElementById(`step${currentStep}-indicator`).classList.add('active');

        if (currentStep > 1) {
            document.getElementById(`line${currentStep - 1}`).classList.add('active');
        }
    }
}

function prevStep() {
    if (currentStep > 1) {
        document.getElementById(`step${currentStep}`).style.display = 'none';
        document.getElementById(`step${currentStep}-indicator`).classList.remove('active');
        document.getElementById(`line${currentStep - 1}`).classList.remove('active');

        currentStep--;
        document.getElementById(`step${currentStep}`).style.display = 'block';
        document.getElementById(`step${currentStep}-indicator`).classList.add('active');
    }
}

function generateReview() {
    const form = document.getElementById('assessmentForm');
    const formData = new FormData(form);
    let html = '<div class="card"><div class="card-body">';
    html += '<h6 class="mb-3">Review Your Inputs:</h6>';
    html += '<table class="table table-sm">';

    for (let [key, value] of formData.entries()) {
        html += `<tr><td class="text-secondary">${key}:</td><td><strong>${value}</strong></td></tr>`;
    }

    html += '</table></div></div>';
    document.getElementById('reviewSummary').innerHTML = html;
}

// Form submission
document.getElementById('assessmentForm').addEventListener('submit', function (e) {
    e.preventDefault();

    // Get all form data, but also read from disabled selects directly
    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());

    // For disabled selects, read value directly from the element
    this.querySelectorAll('select:disabled').forEach(select => {
        data[select.name] = select.value;
    });

    // Convert numeric fields
    data['Age'] = parseInt(data['Age']) || 30;
    data['Prior Convictions'] = parseInt(data['Prior_Convictions']) || 0;
    data['Length of Current Sentence (yrs)'] = parseFloat(data['Length of Current Sentence (yrs)']) || 0;
    data['Time Served (yrs)'] = parseFloat(data['Time Served (years)']) || 0;
    // Jail Behavior Rating is now a string (Good/Fair/Poor) from HTML

    // Map form fields to model features with logic
    data['Prior Convictions'] = data['Prior_Convictions'];

    // Job History Mapping
    if (data['Employment Status'] === 'Unemployed') {
        data['Job History'] = 'Unemployed';
    } else {
        // Map Employed to Clerk to allow for Low Risk scores (Clerk is a protective feature in the model)
        data['Job History'] = 'Clerk';
    }

    // Vocational Training Mapping
    // If they have any education beyond Elementary, assume some training/skills
    if (['Vocational', 'College', 'Graduate', 'High School'].includes(data['Educational Attainment'])) {
        data['Vocational Training'] = 'Yes';
    } else {
        data['Vocational Training'] = 'No';
    }

    // Therapy Attendance Mapping
    // If they participate OR if they don't have issues (so attendance is N/A / Satisfied)
    if (data['Program_Participation'] === 'Yes' ||
        (data['Substance Abuse History'] === 'No' && data['Mental Health Issues'] === 'No')) {
        data['Therapy Attendance'] = 'Yes';
    } else {
        data['Therapy Attendance'] = 'No';
    }

    // Rehab Attitude Mapping
    // Map Remorse to Rehab Attitude as a proxy
    if (data['Remorse'] === 'Yes') {
        data['Rehab Attitude'] = 'Cooperative';
    } else {
        data['Rehab Attitude'] = 'Indifferent';
    }

    // Add missing model fields with defaults
    data['Type of Current Offense'] = data['Offense Type'] || 'Property';
    data['Infractions Count'] = parseInt(data['Infractions Count']) || 0;
    data['Solitary Time (days)'] = 0;
    data['Release Date'] = '2025-12-31';
    data['Impulsivity'] = 'Low';
    data['Manipulativeness'] = 'Low';
    data['Psych Assessment Result'] = 'Normal';
    data['Dependents'] = '0';
    data['Living Before Arrest'] = 'With Family';
    data['Family Relationship'] = data['Family Support'] || 'Moderate';
    data['Family Reunification Intent'] = 'Yes';
    data['Barangay'] = 'Unknown';
    data['Domestic Trauma'] = 'No';
    data['Peer Type'] = 'Positive';
    data['Friends w/ Record'] = 'No';
    data['Rehab Programs'] = data['Program_Participation'] || 'No';
    data['Drug Rehab'] = 'No';
    data['NGO Support'] = 'No';
    data['Education in Jail'] = 'No';
    data['Religious Activities'] = 'Sometimes';
    data['Jail Duties'] = 'Yes';
    data['Job After Release'] = 'Seeking';
    data['Post-Release Housing'] = data['Homelessness'] === 'Yes' ? 'Homeless' : 'With Family';
    data['Post-Release Programs'] = 'None';
    data['Law Belief'] = 'Respect';
    data['Conflict in Jail'] = 'No';
    data['Juvenile Street Living'] = 'No';

    const submitBtn = this.querySelector('button[type="submit"]');
    const originalHTML = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Processing...';

    console.log('Sending data:', data);

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(result => {
            console.log('Result:', result);

            if (result.error) {
                alert('Error: ' + result.error);
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalHTML;
                return;
            }

            sessionStorage.setItem('lastPrediction', JSON.stringify(result));
            sessionStorage.setItem('lastPredictionData', JSON.stringify(data));
            window.location.href = '/results';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalHTML;
        });
});

// Stepper CSS
const style = document.createElement('style');
style.textContent = `
.stepper-step {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.stepper-number {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--border-color);
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 1.125rem;
    margin-bottom: 0.5rem;
}

.stepper-step.active .stepper-number {
    background: var(--bjmp-primary);
    color: white;
}

.stepper-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.stepper-step.active .stepper-label {
    color: var(--bjmp-primary);
    font-weight: 600;
}

.stepper-line {
    flex: 1;
    height: 2px;
    background: var(--border-color);
    margin: 0 1rem;
    margin-bottom: 1.5rem;
}

.stepper-line.active {
    background: var(--bjmp-primary);
}
`;
document.head.appendChild(style);
