import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Define the folder structure
folders = {
    "data/finance": "Finance_Policy_2024.pdf",
    "data/hr": "HR_Employee_Handbook.pdf",
    "data/general": "General_IT_Guidelines.pdf"
}

# Define content for each file (Testing RBAC)
content_map = {
    "data/finance": [
        "CONFIDENTIAL FINANCE DOCUMENT",
        "Subject: Fiscal Year 2025 Budget Caps",
        "1. Travel Budget: $5,000 per department quarterly.",
        "2. Meal Allowance: $50 per day per employee.",
        "3. Approval Limit: Any expense over $200 requires CFO approval.",
        "4. Payroll: Executive payroll runs on the 25th; General on the 30th."
    ],
    "data/hr": [
        "HR DEPARTMENT - INTERNAL POLICY",
        "Subject: Employee Leave & Conduct",
        "1. Work Hours: Standard hours are 9:00 AM to 6:00 PM EST.",
        "2. Paid Leave: Employees get 20 days of PTO per year.",
        "3. Remote Work: Allowed on Tuesdays and Thursdays only.",
        "4. Dress Code: Business casual Monday-Thursday; Casual Friday."
    ],
    "data/general": [
        "COMPANY-WIDE SOP",
        "Subject: IT & General Guidelines",
        "1. Passwords: Must be changed every 90 days.",
        "2. Wi-Fi: Use 'Corp_Secure' network. Password is 'SecurePass2025'.",
        "3. Holidays: Office is closed on New Year's, July 4th, and Christmas.",
        "4. Emergency: In case of fire, use the East Wing stairs."
    ]
}

def create_pdf(path, filename, text_lines):
    full_path = os.path.join(path, filename)
    
    # Ensure folder exists
    if not os.path.exists(path):
        os.makedirs(path)
        
    c = canvas.Canvas(full_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    y_position = 750  # Start writing from top
    for line in text_lines:
        c.drawString(72, y_position, line)
        y_position -= 20  # Move down for next line
        
    c.save()
    print(f"✅ Created: {full_path}")

if __name__ == "__main__":
    print("Generating sample PDFs...")
    for folder, filename in folders.items():
        text = content_map[folder]
        create_pdf(folder, filename, text)
    print("\nDone! You can now run 'python src/ingest.py'")