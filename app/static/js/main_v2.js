document.addEventListener('DOMContentLoaded', function () {

    // Fetch and display statistics from database
    fetchStatistics();

    // ==================== Dashboard Logic ====================
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function (e) {
            e.preventDefault();

            const formData = new FormData(predictionForm);
            const data = Object.fromEntries(formData.entries());

            // Convert numeric types
            const numericFields = ['Age', 'Prior_Convictions', 'Sentence Length (years)', 'Time Served (years)', 'Behavior Score'];
            numericFields.forEach(field => {
                if (data[field]) data[field] = parseFloat(data[field]);
            });

            const submitBtn = predictionForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            submitBtn.disabled = true;

            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
                .then(response => response.json())
                .then(result => {
                    displayPredictionResult(result);
                    // Refresh statistics after prediction
                    fetchStatistics();
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while processing the request.');
                })
                .finally(() => {
                    submitBtn.innerHTML = originalBtnText;
                    submitBtn.disabled = false;
                });
        });
    }

    // ==================== Charts Initialization ====================

    // Fetch and display model metrics
    fetchModelMetrics();

    // Risk Distribution Pie Chart
    let pieChart;
    if (document.getElementById('myPieChart')) {
        const ctxPie = document.getElementById('myPieChart').getContext('2d');
        pieChart = new Chart(ctxPie, {
            type: 'doughnut',
            data: {
                labels: ["Low Risk", "Medium Risk", "High Risk"],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#28A745', '#FFC107', '#DC3545'],
                    borderColor: ['#28A745', '#FFC107', '#DC3545'],
                    borderWidth: 2
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            },
        });
    }

    // Model Performance Bar Chart
    let performanceChart;
    if (document.getElementById('myBarChart')) {
        const ctxBar = document.getElementById('myBarChart').getContext('2d');
        performanceChart = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
                datasets: [{
                    label: "Score (%)",
                    data: [88, 84, 100, 91],
                    backgroundColor: '#0066CC',
                    borderColor: '#0052A3',
                    borderWidth: 1
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function (value) {
                                return value + '%';
                            }
                        }
                    }
                },
            }
        });
    }

    // ==================== Batch Prediction Logic ====================
    const uploadForm = document.getElementById('uploadForm');
    const csvFileInput = document.getElementById('csvFile');

    if (uploadForm && csvFileInput) {
        // ... (keeping existing batch upload logic)

        uploadForm.addEventListener('submit', function (e) {
            e.preventDefault();

            const file = csvFileInput.files[0];
            if (!file) {
                alert('Please select a file first.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            const uploadBtn = document.getElementById('uploadBtn');
            const originalBtnText = uploadBtn.innerHTML;

            uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            uploadBtn.disabled = true;

            fetch('/api/batch_predict', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                    } else {
                        displayBatchResults(data);
                        // Refresh statistics after batch prediction
                        fetchStatistics();
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred during batch prediction.');
                })
                .finally(() => {
                    uploadBtn.innerHTML = originalBtnText;
                    uploadBtn.disabled = false;
                });
        });
    }

    // ==================== Helper Functions ====================

    function fetchStatistics() {
        fetch('/api/statistics')
            .then(response => response.json())
            .then(data => {
                // Update dashboard cards
                if (document.getElementById('totalCount')) {
                    document.getElementById('totalCount').textContent = data.total;
                }
                if (document.getElementById('lowRiskPct')) {
                    document.getElementById('lowRiskPct').textContent = data.low.percentage;
                }
                if (document.getElementById('lowCount')) {
                    document.getElementById('lowCount').textContent = data.low.count;
                }
                if (document.getElementById('mediumRiskPct')) {
                    document.getElementById('mediumRiskPct').textContent = data.medium.percentage;
                }
                if (document.getElementById('mediumCount')) {
                    document.getElementById('mediumCount').textContent = data.medium.count;
                }
                if (document.getElementById('highRiskPct')) {
                    document.getElementById('highRiskPct').textContent = data.high.percentage;
                }
                if (document.getElementById('highCount')) {
                    document.getElementById('highCount').textContent = data.high.count;
                }

                // Update pie chart if it exists
                if (pieChart && data.total > 0) {
                    pieChart.data.datasets[0].data = [
                        data.low.count,
                        data.medium.count,
                        data.high.count
                    ];
                    pieChart.update();
                }
            })
            .catch(error => {
                console.error('Error fetching statistics:', error);
            });
    }

    function displayPredictionResult(result) {
        const resultArea = document.getElementById('resultArea');
        const resultAlert = document.getElementById('resultAlert');
        const resultTitle = document.getElementById('resultTitle');
        const resultText = document.getElementById('resultText');
        const resultProbability = document.getElementById('resultProbability');
        const resultClassification = document.getElementById('resultClassification');
        const resultBadge = document.getElementById('resultBadge');
        const resultRecommendation = document.getElementById('resultRecommendation');

        resultArea.classList.remove('d-none');

        if (result.error) {
            resultAlert.className = 'alert alert-danger';
            resultTitle.textContent = 'Error';
            resultText.textContent = result.error;
            resultProbability.textContent = '';
            resultClassification.textContent = '';
            resultBadge.style.display = 'none';
            resultRecommendation.textContent = '';
        } else {
            // Set alert style based on risk level
            if (result.risk_level === 'High') {
                resultAlert.className = 'alert alert-danger';
                resultBadge.className = 'badge bg-danger';
            } else if (result.risk_level === 'Medium') {
                resultAlert.className = 'alert alert-warning';
                resultBadge.className = 'badge bg-warning';
            } else {
                resultAlert.className = 'alert alert-success';
                resultBadge.className = 'badge bg-success';
            }

            resultTitle.textContent = `${result.risk_level} Risk Assessment`;
            resultText.textContent = `Based on the provided information, the system predicts a ${result.risk_level.toLowerCase()} risk of recidivism.`;
            resultProbability.textContent = `${(result.probability * 100).toFixed(2)}%`;
            resultClassification.textContent = result.prediction === 1 ? 'Likely to Reoffend' : 'Not Likely to Reoffend';
            resultBadge.textContent = result.risk_level;
            resultBadge.style.display = 'inline-block';

            // Add recommendations
            const recommendations = {
                'Low': 'Continue standard rehabilitation programs. Monitor progress regularly.',
                'Medium': 'Recommend enhanced supervision and targeted intervention programs.',
                'High': 'Require intensive rehabilitation, close monitoring, and comprehensive support services.'
            };
            resultRecommendation.textContent = `Recommendation: ${recommendations[result.risk_level]}`;

            // Scroll to result
            resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function displayBatchResults(results) {
        const resultsSection = document.getElementById('resultsSection');
        const tbody = document.querySelector('#resultsTable tbody');

        tbody.innerHTML = '';

        let low = 0, medium = 0, high = 0;

        results.forEach((item, index) => {
            const tr = document.createElement('tr');

            // Count risk levels
            if (item.risk_level === 'Low') low++;
            else if (item.risk_level === 'Medium') medium++;
            else if (item.risk_level === 'High') high++;

            let badgeClass = 'bg-success';
            if (item.risk_level === 'High') badgeClass = 'bg-danger';
            else if (item.risk_level === 'Medium') badgeClass = 'bg-warning';

            tr.innerHTML = `
                <td><strong>${index + 1}</strong></td>
                <td>${item.id}</td>
                <td>${item.name}</td>
                <td><span class="badge ${badgeClass}">${item.risk_level}</span></td>
                <td><strong>${(item.probability * 100).toFixed(2)}%</strong></td>
                <td>
                    <button class="btn btn-sm btn-outline-primary">
                        <i class="fas fa-eye"></i> View
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        // Store results for download
        window.batchResults = results;

        resultsSection.classList.remove('d-none');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Download results as CSV
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function () {
            if (!window.batchResults) return;

            const headers = ['ID', 'Name', 'Risk Level', 'Probability', 'Prediction'];
            const rows = window.batchResults.map(r => [
                r.id,
                r.name,
                r.risk_level,
                (r.probability * 100).toFixed(2) + '%',
                r.prediction === 1 ? 'Likely to Reoffend' : 'Not Likely'
            ]);

            let csvContent = headers.join(',') + '\n';
            csvContent += rows.map(row => row.join(',')).join('\n');

            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `recidivism_predictions_${new Date().toISOString().split('T')[0]}.csv`;
            a.click();
            window.URL.revokeObjectURL(url);
        });
    }

    // Fetch model metrics from API
    async function fetchModelMetrics() {
        try {
            const response = await fetch('/api/model_info');
            const data = await response.json();

            if (data.metrics && performanceChart) {
                performanceChart.data.datasets[0].data = [
                    data.metrics.accuracy,
                    data.metrics.precision,
                    data.metrics.recall,
                    data.metrics.f1_score
                ];
                performanceChart.update();
            }
        } catch (error) {
            console.error('Error fetching model metrics:', error);
        }
    }
});

// Reset form function
function resetForm() {
    const form = document.getElementById('predictionForm');
    if (form) form.reset();
    const resultArea = document.getElementById('resultArea');
    if (resultArea) resultArea.classList.add('d-none');
}
