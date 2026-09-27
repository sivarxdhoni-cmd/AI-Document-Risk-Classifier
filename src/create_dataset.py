import csv
import os


DATASET_PATH = os.path.join("data", "risk_dataset.csv")


samples = [
    # =========================
    # LOW RISK
    # =========================

    {
        "text": "The office will conduct a monthly team meeting on Friday at 10 AM. Employees should review the agenda before attending.",
        "label": "low",
        "policy_type": "General Business Policy",
    },
    {
        "text": "Employees may use the company meeting room for internal discussions by booking the room through the office calendar.",
        "label": "low",
        "policy_type": "General Business Policy",
    },
    {
        "text": "The company will provide basic office stationery including notebooks, pens, folders, and printing supplies.",
        "label": "low",
        "policy_type": "General Business Policy",
    },
    {
        "text": "The weekly project status meeting will be held every Monday afternoon. Team members should submit their updates before the meeting.",
        "label": "low",
        "policy_type": "General Business Policy",
    },
    {
        "text": "Employees can request standard office equipment through the internal administration portal.",
        "label": "low",
        "policy_type": "General Business Policy",
    },
    {
        "text": "The company cafeteria will operate from 9 AM to 5 PM on working days.",
        "label": "low",
        "policy_type": "General Business Policy",
    },
    {
        "text": "The project team will maintain a shared calendar containing important meetings and internal deadlines.",
        "label": "low",
        "policy_type": "General Business Policy",
    },
    {
        "text": "Employees should inform their manager when they need access to a shared project folder.",
        "label": "low",
        "policy_type": "General Business Policy",
    },

    # =========================
    # MEDIUM RISK
    # =========================

    {
        "text": "Employees must submit their monthly expense reports within ten business days. Missing the deadline may delay reimbursement.",
        "label": "medium",
        "policy_type": "Financial Compliance",
    },
    {
        "text": "Business travel expenses must be supported by receipts and approved by the employee's department manager.",
        "label": "medium",
        "policy_type": "Financial Compliance",
    },
    {
        "text": "Employees must complete the annual workplace safety training before the end of the calendar year.",
        "label": "medium",
        "policy_type": "Employment/HR",
    },
    {
        "text": "All contractors must complete the company's required onboarding documentation before receiving system access.",
        "label": "medium",
        "policy_type": "Employment/HR",
    },
    {
        "text": "Purchase requests above the department approval limit require authorization from the finance manager.",
        "label": "medium",
        "policy_type": "Financial Compliance",
    },
    {
        "text": "Employees must notify Human Resources within five working days when their personal contact information changes.",
        "label": "medium",
        "policy_type": "Employment/HR",
    },
    {
        "text": "Vendor invoices should be reviewed against purchase orders before payment is released.",
        "label": "medium",
        "policy_type": "Financial Compliance",
    },
    {
        "text": "Managers must document employee performance discussions and retain the records according to company policy.",
        "label": "medium",
        "policy_type": "Employment/HR",
    },

    # =========================
    # HIGH RISK
    # =========================

    {
        "text": "Customer personal information including identity documents and financial details must be protected from unauthorized access and disclosure.",
        "label": "high",
        "policy_type": "Data Privacy",
    },
    {
        "text": "Employees must not share confidential customer information with unauthorized third parties. Any suspected disclosure must be reported immediately.",
        "label": "high",
        "policy_type": "Confidentiality",
    },
    {
        "text": "The agreement permits the company to process sensitive personal data for specified business purposes and requires appropriate security controls.",
        "label": "high",
        "policy_type": "Data Privacy",
    },
    {
        "text": "Financial records containing confidential account information must only be accessed by authorized personnel.",
        "label": "high",
        "policy_type": "Financial Compliance",
    },
    {
        "text": "The contract includes confidentiality obligations that continue after termination and restrict disclosure of proprietary business information.",
        "label": "high",
        "policy_type": "Contractual Risk",
    },
    {
        "text": "A security incident involving customer personal data must be escalated to the designated compliance team for investigation.",
        "label": "high",
        "policy_type": "Data Privacy",
    },
    {
        "text": "The supplier agreement requires compliance with applicable privacy and data protection obligations when processing customer information.",
        "label": "high",
        "policy_type": "Regulatory",
    },
    {
        "text": "Unauthorized disclosure of trade secrets, proprietary source code, or confidential business information may result in contractual remedies.",
        "label": "high",
        "policy_type": "Confidentiality",
    },
]


def create_dataset():
    os.makedirs("data", exist_ok=True)

    fieldnames = [
        "document_id",
        "text",
        "label",
        "policy_type",
    ]

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for index, sample in enumerate(samples, start=1):
            writer.writerow(
                {
                    "document_id": f"DOC_{index:03d}",
                    "text": sample["text"],
                    "label": sample["label"],
                    "policy_type": sample["policy_type"],
                }
            )

    print(f"Dataset created successfully: {DATASET_PATH}")
    print(f"Total documents: {len(samples)}")


if __name__ == "__main__":
    create_dataset()