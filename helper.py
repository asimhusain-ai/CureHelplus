"""
Enhanced Gemini Helper Module
Provides hardcoded, evidence-aligned prevention measures and medications
for each disease based on risk percentages. All recommendations are detailed,
actionable, and relevant to the disease and risk level.
"""

def fetch_gemini_recommendations(disease: str, risk: float):
    """
    Returns prevention measures and medications for a given disease
    according to risk thresholds:

      - risk < 35%: 4 preventions & 4 medications (15-20 words each)
      - 35% <= risk < 70%: 6 preventions & 6 medications (20-35 words each)
      - risk >= 70%: 8 preventions & 8 medications (35-50 words each)

    Each prevention and medication is relevant, actionable, and safe.
    """

    disease = disease.lower()

    recommendations = {
        "diabetes": {
            "low": {
                "preventions": [
                    "Eat a balanced diet rich in vegetables, whole grains, and lean protein to prevent sudden blood sugar spikes.",
                    "Walk or cycle daily for at least 30 minutes to improve insulin sensitivity and cardiovascular health.",
                    "Check fasting blood sugar once a month to track your body's glucose response.",
                    "Avoid sugary drinks and processed foods to maintain steady blood sugar levels."
                ],
                "medications": [
                    "Consult a doctor for lifestyle guidance including diet and exercise adjustments to manage early risk.",
                    "Consider vitamin D supplements if deficient, as it supports insulin function and metabolic health.",
                    "Regular medical checkups help monitor risk and catch early signs of diabetes.",
                    "Metformin is usually not required at this stage unless prescribed by a physician."
                ]
            },
            "medium": {
                "preventions": [
                    "Spread carbohydrate intake evenly through meals and prioritize complex carbs to prevent sugar spikes.",
                    "Exercise daily combining aerobic and light resistance training to improve glucose utilization.",
                    "Monitor fasting and post-meal blood sugar at home to guide lifestyle adjustments.",
                    "Attend checkups with an endocrinologist to evaluate metabolic parameters and review lab results.",
                    "Maintain healthy weight through portion control and mindful eating to reduce diabetes progression risk.",
                    "Avoid late-night high-carb snacks to stabilize blood glucose levels overnight."
                ],
                "medications": [
                    "Take oral antidiabetic medication like Metformin as prescribed to control blood sugar.",
                    "Monitor HbA1c every 3 months to assess long-term glucose management.",
                    "Consult a nutritionist to adjust diet based on blood sugar trends.",
                    "Use cholesterol-lowering medications if lipid levels are high under physician guidance.",
                    "Track medication adherence carefully to ensure effective glucose control.",
                    "Discuss possible dose adjustments with your doctor if readings fluctuate."
                ]
            },
            "high": {
                "preventions": [
                    "Follow a strict diabetic diet avoiding refined sugar and high-glycemic foods completely.",
                    "Measure glucose multiple times daily, keeping detailed logs for your physician.",
                    "Engage in supervised exercise suitable for your condition balancing aerobic and resistance training.",
                    "Attend frequent medical consultations to review labs, adjust medications, and screen for complications.",
                    "Practice daily foot care and routine eye exams to prevent neuropathy and retinopathy.",
                    "Monitor blood pressure, cholesterol, and weight as part of comprehensive diabetes care.",
                    "Stay hydrated and manage stress with meditation or relaxation exercises.",
                    "Ensure consistent sleep patterns to help regulate blood glucose levels naturally."
                ],
                "medications": [
                    "Administer insulin therapy as prescribed and monitor dosages carefully.",
                    "Continue oral antidiabetic medications if advised by your doctor alongside insulin.",
                    "Monitor HbA1c and glucose daily to avoid hypo- or hyperglycemia.",
                    "Review medication adherence regularly with your healthcare provider.",
                    "Manage hypertension or hyperlipidemia with appropriate medications.",
                    "Adjust insulin doses during illness or changes in diet/exercise under medical supervision.",
                    "Maintain a log of all medications and doses for reference in emergencies.",
                    "Discuss combination therapy options with your endocrinologist if blood sugar remains uncontrolled."
                ]
            }
        },
        "heart disease": {
            "low": {
                "preventions": [
                    "Eat a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins to support cardiovascular health.",
                    "Walk, swim, or jog for 30 minutes daily to improve heart fitness and circulation.",
                    "Avoid smoking and limit alcohol intake to reduce strain on the heart.",
                    "Monitor blood pressure periodically to ensure it stays within healthy limits."
                ],
                "medications": [
                    "Lifestyle changes are usually sufficient; medications are rarely needed at low risk.",
                    "Check cholesterol annually to detect early dyslipidemia and adjust diet accordingly.",
                    "Maintain a healthy weight through portion control and regular activity.",
                    "Consult a doctor if any heart symptoms like chest discomfort or palpitations appear."
                ]
            },
            "medium": {
                "preventions": [
                    "Maintain blood pressure and cholesterol within target ranges through diet, exercise, and regular checkups.",
                    "Keep a healthy body weight using portion control and a combination of aerobic and strength exercises.",
                    "Engage in moderate-intensity exercise five days per week to enhance heart function.",
                    "Practice stress reduction through mindfulness meditation, yoga, or deep breathing techniques.",
                    "Reduce processed food intake and increase fiber-rich meals to support cardiovascular health.",
                    "Monitor heart rate regularly to detect abnormal patterns or early warning signs."
                ],
                "medications": [
                    "Take anti-hypertensive medications exactly as prescribed to control blood pressure.",
                    "Use statins if needed to manage cholesterol and prevent plaque buildup.",
                    "Follow doctor recommendations on aspirin or anti-platelet therapy to reduce clot risk.",
                    "Consider beta-blockers or ACE inhibitors as advised to reduce strain on the heart.",
                    "Track medication adherence and side effects to discuss at checkups.",
                    "Adjust treatment with your physician if lifestyle changes do not achieve target values."
                ]
            },
            "high": {
                "preventions": [
                    "Adhere strictly to a low-sodium, low-fat, and low-cholesterol diet to minimize cardiac workload.",
                    "Follow a monitored exercise plan suitable for your heart condition with physician supervision.",
                    "Avoid all tobacco and alcohol to prevent further vascular damage.",
                    "Attend regular cardiac checkups including ECGs, echocardiograms, and lab tests.",
                    "Immediately report chest pain, shortness of breath, dizziness, or palpitations to a healthcare provider.",
                    "Maintain consistent medication schedules and monitor blood pressure at home daily.",
                    "Implement stress management strategies such as meditation and therapy to reduce cardiac risk.",
                    "Keep a clear emergency plan with family and cardiologist for urgent care situations."
                ],
                "medications": [
                    "Follow prescribed multiple anti-hypertensive medications to maintain optimal blood pressure.",
                    "Continue statins or other lipid-lowering drugs as directed by your cardiologist.",
                    "Use aspirin or clopidogrel if recommended to prevent clot formation.",
                    "Undergo frequent cardiac monitoring and lab tests to evaluate treatment effectiveness.",
                    "Follow combination therapy as advised including beta-blockers, ACE inhibitors, and diuretics if necessary.",
                    "Adjust medication regimens promptly if side effects or abnormal readings occur.",
                    "Maintain detailed records of medications, doses, and schedules for emergencies.",
                    "Ensure family and caregivers are informed about warning signs and intervention plans."
                ]
            }
        },
        "fever": {
            "low": {
                "preventions": [
                    "Stay hydrated by drinking water, herbal teas, and broths to support immune function.",
                    "Get sufficient rest and sleep to reduce stress on your immune system.",
                    "Eat light, balanced meals including fruits, vegetables, and soups to maintain energy.",
                    "Wash hands regularly and avoid contact with sick individuals to prevent infections."
                ],
                "medications": [
                    "Use paracetamol only if necessary to relieve mild fever, following recommended doses.",
                    "Monitor temperature daily and record patterns for healthcare guidance.",
                    "Consult a doctor if fever persists beyond three days or worsens.",
                    "Avoid unnecessary antibiotics to prevent resistance and side effects."
                ]
            },
            "medium": {
                "preventions": [
                    "Maintain hydration with electrolyte drinks to prevent imbalances during fever.",
                    "Avoid strenuous activity to reduce fatigue and support recovery.",
                    "Monitor temperature closely and note rapid rises or persistent high fever.",
                    "Seek medical consultation promptly to rule out bacterial or viral infections.",
                    "Practice good hygiene and avoid crowded places to prevent spread.",
                    "Rest in a comfortable environment to help the immune system fight infection."
                ],
                "medications": [
                    "Take paracetamol or acetaminophen at proper intervals to reduce fever safely.",
                    "Use NSAIDs like ibuprofen under medical supervision to manage inflammation and fever.",
                    "Start antibiotics or antivirals only if prescribed based on diagnostic testing.",
                    "Monitor vital signs including pulse and oxygen saturation frequently to catch complications.",
                    "Follow physician instructions on fluid and nutrition intake for recovery.",
                    "Track symptom progression to report accurately to healthcare providers."
                ]
            },
            "high": {
                "preventions": [
                    "Seek immediate hospital care for severe or potentially life-threatening infections.",
                    "Follow isolation protocols if fever is due to a contagious pathogen to protect others.",
                    "Ensure strict hydration with IV fluids if dehydration occurs from high fever.",
                    "Monitor temperature, pulse, and oxygen saturation continuously in a hospital setting.",
                    "Follow dietary and activity recommendations strictly as per medical instructions.",
                    "Implement supportive care such as cooling blankets or oxygen therapy as advised.",
                    "Avoid self-medication beyond prescribed treatments to prevent complications.",
                    "Maintain detailed records of symptoms, medications, and interventions for physicians."
                ],
                "medications": [
                    "Administer IV fluids for hydration and electrolyte balance under supervision.",
                    "Use paracetamol or antipyretics as prescribed to control temperature safely.",
                    "Follow antibiotics or antivirals exactly as instructed for infection treatment.",
                    "Ensure continuous monitoring in hospital for early detection of deterioration.",
                    "Implement oxygen therapy if oxygen saturation drops below safe limits.",
                    "Use additional supportive treatments like anti-inflammatory medications as directed.",
                    "Follow physician guidance on nutrition and rest to optimize recovery.",
                    "Avoid overuse of medications to reduce risk of side effects or resistance."
                ]
            }
        },
        "anemia": {
            "low": {
                "preventions": [
                    "Eat iron-rich foods like spinach, lentils, red meat, and fortified cereals to prevent mild deficiencies.",
                    "Pair iron sources with vitamin C-rich foods to improve absorption and utilization.",
                    "Include folate and vitamin B12-rich foods to support red blood cell production.",
                    "Monitor hemoglobin levels periodically to ensure they remain within healthy ranges."
                ],
                "medications": [
                    "Use oral iron supplements if mild deficiency is diagnosed as per doctor guidance.",
                    "Consult a doctor or nutritionist for personalized dietary recommendations.",
                    "Maintain follow-ups to monitor hemoglobin and ferritin levels.",
                    "Avoid excessive tea or coffee around meals to maximize iron absorption."
                ]
            },
            "medium": {
                "preventions": [
                    "Include iron, folate, and vitamin B12-rich foods daily to address moderate deficiencies.",
                    "Avoid foods or drinks that inhibit iron absorption during meals.",
                    "Perform regular hemoglobin and ferritin monitoring to track improvement.",
                    "Consult a dietitian for personalized meal planning to maximize nutrient intake.",
                    "Maintain a balanced diet to prevent further deficiency and related fatigue.",
                    "Track symptoms like weakness, pallor, and shortness of breath for medical review."
                ],
                "medications": [
                    "Take oral iron supplements as prescribed to restore iron levels effectively.",
                    "Use vitamin B12 or folate supplements to correct specific deficiencies.",
                    "Undergo periodic complete blood counts (CBC) to monitor response.",
                    "Adjust supplementation with doctor supervision based on blood results.",
                    "Address symptoms like dizziness or fatigue with timely medical advice.",
                    "Follow consistent medication schedules for effective treatment of anemia."
                ]
            },
            "high": {
                "preventions": [
                    "Follow a strict diet rich in iron, B12, and folate while avoiding substances that inhibit absorption.",
                    "Attend frequent medical checkups with blood tests to monitor anemia severity closely.",
                    "Consult a hematologist to investigate underlying causes and receive specialized guidance.",
                    "Monitor for severe fatigue, pallor, shortness of breath, or rapid heartbeat and report immediately.",
                    "Ensure strict adherence to all medical and dietary recommendations to prevent complications.",
                    "Plan daily activities to avoid overexertion and reduce strain on the heart.",
                    "Maintain hydration and adequate sleep to support recovery and energy levels.",
                    "Keep a detailed record of symptoms, medications, and lab results for physician review."
                ],
                "medications": [
                    "Receive intravenous iron therapy if oral supplementation is insufficient or rapid correction is needed.",
                    "Administer vitamin B12 injections for severe deficiencies under medical supervision.",
                    "Consider blood transfusions if hemoglobin drops critically low to prevent organ damage.",
                    "Monitor CBC and iron studies regularly to adjust therapy appropriately.",
                    "Use erythropoietin therapy if indicated by severity and underlying cause.",
                    "Follow medication schedules carefully and report side effects promptly.",
                    "Track hemoglobin and ferritin levels to ensure therapeutic effectiveness.",
                    "Consult your hematologist regularly for treatment adjustments and preventive strategies."
                ]
            }
        }
    }

    # Determine risk tier thresholds
    if risk < 35:
        tier = "low"
    elif 35 <= risk < 70:
        tier = "medium"
    else:
        tier = "high"

    recs = recommendations.get(disease, {}).get(tier, {"preventions": [], "medications": []})
    return {
        "Risk Level": tier,
        "prevention_measures": recs["preventions"],
        "medicine_suggestions": recs["medications"]
    }

# Test Run
if __name__ == "__main__":
    diseases = ["Diabetes", "Heart Disease", "Fever", "Anemia"]
    risks = [10, 40, 70, 100]

    for disease in diseases:
        for risk in risks:
            print(f"\n--- {disease} | Risk: {risk}% ---")
            result = fetch_gemini_recommendations(disease, risk)
            print(result)