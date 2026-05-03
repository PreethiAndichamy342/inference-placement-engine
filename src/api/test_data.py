"""
Sample prompts for each HIPAA data-sensitivity tier.

Used by the ``GET /test-prompts`` endpoint and the dashboard's preset test
buttons.  All PHI in the ``phi`` and ``phi_strict`` tiers is entirely
fabricated — no real patient data is included.
"""

from __future__ import annotations


def get_sample_prompts() -> dict[str, list[str]]:
    """
    Return a dict mapping each sensitivity tier to a list of sample prompts.

    Tiers
    -----
    public       General health questions — no patient identity, safe for any cloud.
    internal     Clinical workflows and protocols — no patient names or IDs.
    sensitive    Contains some PII (names, dates) but not full PHI sets.
    phi          Full PHI: names, DOBs, phone numbers, medications, diagnoses.
    phi_strict   Heavy PHI + 42 CFR Part 2 substance abuse data, SSNs, MRNs.
    """
    return {
        "public": [
            "What are the early warning signs of type 2 diabetes?",
            "How does hypertension affect long-term cardiovascular risk?",
            "What lifestyle changes are recommended after a myocardial infarction?",
            "Explain the difference between Type 1 and Type 2 diabetes management.",
            "What are the CDC guidelines for colorectal cancer screening?",
        ],
        "internal": [
            "Summarize the discharge note structure used in cardiology wards.",
            "What is the standard pre-op checklist for elective laparoscopic procedures?",
            "Describe the workflow for triage in a level-2 trauma center.",
            "What documentation is required before initiating anticoagulation therapy?",
            "List the key elements of a SOAP note for an initial psychiatric evaluation.",
        ],
        "sensitive": [
            "Patient Emily Carter, DOB 09/14/1967, presented with chest pain. Summarize next steps.",
            "Dr. Nguyen ordered an MRI for Robert Fields on March 3rd. Draft the radiology referral.",
            "Follow-up note for Sarah Kim, seen on 2024-11-20, complaints of persistent migraines.",
            "Patient Thomas Reyes, age 52, allergic to penicillin — recommend alternative antibiotics.",
            "Nurse handoff note: Linda Patel in room 14B, post-op day 2, vitals stable, mild pain 4/10.",
        ],
        "phi": [
            (
                "Patient Mary Smith, DOB 05/12/1980, phone 617-555-0142, called to request a refill "
                "for Metformin 500mg twice daily. Last seen 2024-10-05. Insured under BlueCross "
                "member ID BCX-447821."
            ),
            (
                "Generate a visit summary for James O'Brien, DOB 11/03/1955, MRN A2039481. "
                "Chief complaint: shortness of breath. Diagnosed with COPD exacerbation. "
                "Prescribed Prednisone 40mg x 5 days and Albuterol inhaler."
            ),
            (
                "Patient Angela Russo, email angela.russo@email.com, DOB 02/28/1990. "
                "Referred to cardiology by Dr. Kim for evaluation of palpitations. "
                "ECG scheduled for 2025-03-10 at 9:00 AM."
            ),
            (
                "Discharge instructions for David Chen, DOB 07/19/1972, admitted 2025-01-15 "
                "for appendectomy. Discharged 2025-01-17. Follow up with Dr. Patel in 2 weeks. "
                "Prescriptions: Oxycodone 5mg PRN pain, Keflex 500mg QID x 7 days."
            ),
            (
                "Pre-authorization request for Maria Gonzalez, DOB 03/31/1985, insurance policy "
                "HMO-5512990. Requesting approval for laparoscopic cholecystectomy. "
                "Diagnosis: acute cholecystitis ICD-10 K81.0. Surgeon: Dr. Hassan, NPI 1234567890."
            ),
        ],
        "phi_strict": [
            (
                "Patient John Doe, SSN 123-45-6789, MRN 987654, DOB 08/22/1978. "
                "Admitted for inpatient substance abuse treatment per 42 CFR Part 2. "
                "Primary substance: opioids. Consent on file. Facility: Cedar Ridge Treatment Center."
            ),
            (
                "Confidential alcohol treatment record — 42 CFR Part 2 applies. "
                "Patient: Sandra Williams, SSN 987-65-4320, DOB 04/15/1983, MRN 00234561. "
                "Admitted 2025-02-01 for medically supervised alcohol detox. "
                "Prescribed Librium taper; daily CIWA monitoring in progress."
            ),
            (
                "Re-disclosure authorization for Kevin Martinez, SSN 456-78-9012, MRN B9912345. "
                "Patient consents to share substance use disorder records with primary care "
                "provider Dr. Lisa Tran, NPI 9876543210, per 42 CFR Part 2 requirements."
            ),
            (
                "Court-ordered assessment report for patient Brenda Okonkwo, DOB 12/01/1969, "
                "SSN 321-54-8765, MRN 00445678. Referred by county court case #2025-CR-00412. "
                "Assessment for methamphetamine use disorder; 42 CFR Part 2 protections apply. "
                "Treating clinician: Dr. Reyes, LCSW."
            ),
            (
                "Minors substance abuse treatment note — 42 CFR Part 2 & state minor consent laws apply. "
                "Patient: Tyler Nguyen, DOB 06/14/2007, MRN JV-20078834. "
                "Parent/guardian not notified per minor consent statute. "
                "Intake assessment: cannabis and benzodiazepine misuse. "
                "Treatment plan: outpatient CBT, weekly sessions."
            ),
        ],
    }
