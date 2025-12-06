import urllib.request
import json
import urllib.error

BASE_URL = "http://localhost:5000"

def test_prediction():
    print("Testing Hybrid RNR Prediction System...")
    
    # Test Case 1: Low Risk (Clean Record)
    print("\nTest Case 1: Low Risk Profile (Clean Record)")
    low_risk_data = {
        "Name": "Low Risk Subject",
        "Age": "25",
        "Gender": "Female",
        "Civil Status": "Single",
        "Religion": "Catholic",
        "Educational Attainment": "College",
        "Employment Status": "Employed",
        "Prior Convictions": "0",
        "Offense Type": "Theft",
        "Length of Current Sentence (yrs)": "1",
        "Time Served (years)": "0.5",
        "Substance Abuse History": "No",
        "Mental Health Issues": "No",
        "Family Support": "Good",
        "Gang Affiliation": "No",
        "Program_Participation": "Yes",
        "Behavior Score": "10",
        "Infractions Count": "0",
        "Rehab Attitude": "Cooperative",
        "Jail Behavior Rating": "Good",
        "Aggression": "Low",
        "Vocational Training": "Yes",
        "Therapy Attendance": "Regular"
    }
    
    send_request(low_risk_data)

    # Test Case 2: High Risk RNR Profile (Should be High Risk > 70%)
    print("\nTest Case 2: High Risk RNR Profile (3 Priors, Substance Abuse, Gang)")
    high_rnr_data = {
        "Name": "RNR Test Subject",
        "Age": "30",
        "Gender": "Male",
        "Civil Status": "Single",
        "Religion": "Catholic",
        "Educational Attainment": "High School",
        "Employment Status": "Unemployed",
        "Prior Convictions": "3",
        "Offense Type": "Illegal Drugs",
        "Length of Current Sentence (yrs)": "5",
        "Time Served (years)": "2",
        "Substance Abuse History": "Yes",
        "Mental Health Issues": "No",
        "Family Support": "Poor",
        "Gang Affiliation": "Yes",
        "Program_Participation": "No",
        "Behavior Score": "0",
        "Infractions Count": "6",
        "Rehab Attitude": "Uncooperative",
        "Jail Behavior Rating": "Poor",
        "Aggression": "High",
        "Vocational Training": "No",
        "Therapy Attendance": "None"
    }
    
    send_request(high_rnr_data)

def send_request(data):
    url = f"{BASE_URL}/api/predict"
    headers = {'Content-Type': 'application/json'}
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode('utf-8'), 
            headers=headers, 
            method='POST'
        )
        
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                response_body = response.read().decode('utf-8')
                try:
                    result = json.loads(response_body)
                    print(f"Risk Level: {result['risk_level']}")
                    print(f"Final Probability: {result['probability']:.2%}")
                    print(f"ML Probability: {result.get('ml_probability', 0):.2%}")
                    print(f"RNR Probability: {result.get('rnr_probability', 0):.2%}")
                    if 'rnr_breakdown' in result:
                        print(f"RNR Breakdown: {result['rnr_breakdown']}")
                except json.JSONDecodeError:
                    print(f"Error: Failed to decode JSON. Raw response: {response_body}")
            else:
                print(f"Error: {response.status} - {response.read().decode('utf-8')}")
                
    except urllib.error.URLError as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'read'):
             print(e.read().decode('utf-8'))

if __name__ == "__main__":
    test_prediction()
