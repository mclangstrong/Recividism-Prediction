// Batch Predictions JavaScript

document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('uploadForm');
    const csvFile = document.getElementById('csvFile');
    const fileNameDisplay = document.getElementById('fileName');
    const uploadBtn = document.getElementById('uploadBtn');
    const progressBar = document.getElementById('progressBar');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadStatus = document.getElementById('uploadStatus');
    const resultsSection = document.getElementById('resultsSection');
    const resultsTableBody = document.querySelector('#resultsTable tbody');

    // Drag and Drop Handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadForm.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadForm.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadForm.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadForm.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadForm.classList.remove('dragover');
    }

    uploadForm.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // File Input Handling
    csvFile.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                fileNameDisplay.textContent = `Selected: ${file.name}`;
                uploadBtn.classList.remove('d-none');
                uploadStatus.innerHTML = '';
            } else {
                uploadStatus.innerHTML = '<div class="alert alert-danger">Please upload a valid CSV file.</div>';
                fileNameDisplay.textContent = '';
                uploadBtn.classList.add('d-none');
            }
        }
    }

    // Upload and Predict
    uploadBtn.addEventListener('click', function (e) {
        e.preventDefault();

        const file = csvFile.files[0];
        if (!file) return;

        // Reset UI
        uploadProgress.classList.remove('d-none');
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        uploadBtn.disabled = true;
        uploadStatus.innerHTML = '';
        resultsSection.classList.add('d-none');

        const formData = new FormData();
        formData.append('file', file);

        // Simulate progress (since fetch doesn't support upload progress easily without XHR)
        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            if (progress > 90) clearInterval(interval);
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${progress}%`;
        }, 100);

        fetch('/api/batch_predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                clearInterval(interval);
                progressBar.style.width = '100%';
                progressBar.textContent = '100%';

                if (data.error) {
                    throw new Error(data.error);
                }

                displayResults(data);

                setTimeout(() => {
                    uploadProgress.classList.add('d-none');
                    uploadBtn.disabled = false;
                }, 1000);
            })
            .catch(error => {
                clearInterval(interval);
                progressBar.style.width = '0%';
                uploadProgress.classList.add('d-none');
                uploadBtn.disabled = false;
                uploadStatus.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
    });

    function displayResults(results) {
        resultsSection.classList.remove('d-none');
        resultsTableBody.innerHTML = '';

        let lowCount = 0;
        let mediumCount = 0;
        let highCount = 0;

        results.forEach((result, index) => {
            // Count risks
            if (result.risk_level === 'Low') lowCount++;
            else if (result.risk_level === 'Medium') mediumCount++;
            else if (result.risk_level === 'High') highCount++;

            // Create row
            const row = document.createElement('tr');

            // Risk Badge
            // Risk Badge
            let riskBadge = '';
            if (result.risk_level === 'Low') {
                riskBadge = '<span class="badge risk-low"><i class="fas fa-check-circle me-1"></i>Low</span>';
            } else if (result.risk_level === 'Medium') {
                riskBadge = '<span class="badge risk-medium"><i class="fas fa-exclamation-triangle me-1"></i>Medium</span>';
            } else if (result.risk_level === 'High') {
                riskBadge = '<span class="badge risk-high"><i class="fas fa-exclamation-circle me-1"></i>High</span>';
            } else {
                riskBadge = '<span class="badge bg-secondary">Error</span>';
            }

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${result.id || '-'}</td>
                <td>${result.name}</td>
                <td>${riskBadge}</td>
                <td>${(result.probability * 100).toFixed(1)}%</td>
                <td>
                    <button class="btn btn-sm btn-outline-info" onclick="viewDetails('${result.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            `;
            resultsTableBody.appendChild(row);
        });

        // Update Stats
        document.getElementById('resultCount').textContent = results.length;
        document.getElementById('lowRiskCount').textContent = lowCount;
        document.getElementById('mediumRiskCount').textContent = mediumCount;
        document.getElementById('highRiskCount').textContent = highCount;

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Download CSV Results
    document.getElementById('downloadBtn').addEventListener('click', function () {
        const rows = Array.from(document.querySelectorAll('#resultsTable tr'));
        const csvContent = rows.map(row => {
            const cells = Array.from(row.querySelectorAll('th, td'));
            return cells.map(cell => cell.innerText).join(',');
        }).join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction_results.csv';
        a.click();
        window.URL.revokeObjectURL(url);
    });
});
