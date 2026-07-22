import re
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Rwanda Academic Guidance Portal",
    page_icon="🎓",
    layout="wide",
)

# =========================================================
# MODEL LOADING
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "academic_pathway_model.joblib"

INPUT_FEATURES = [
    "EducationType",
    "Pathway",
    "Stream_or_Trade",
    "BestSubject",
    "WeakestSubject",
    "InterestArea",
    "AverageScoreRange",
    "DigitalSkillLevel",
    "CareerCluster",
]

# This dictionary follows the bridge-course mapping used in the final model-training notebook.
DEFAULT_PROGRAM_CATEGORY_TO_BRIDGE_COURSE = {
    "Medicine and Surgery": "Human Biology, Chemistry Foundations, and Scientific Study Skills",
    "Nursing and Midwifery": "Human Biology, Patient Care Basics, and Health Communication",
    "Pharmacy and Pharmaceutical Sciences": "Chemistry Foundations, Biology, and Pharmaceutical Science Basics",
    "Biomedical Laboratory Sciences": "Biology, Chemistry, Laboratory Safety, and Scientific Methods",
    "Biotechnology and Applied Biosciences": "Biology, Chemistry, Genetics Basics, and Laboratory Skills",

    "Civil Engineering and Construction Technology": "Mathematics, Physics, Technical Drawing, and Construction Materials",
    "Electrical Engineering and Power Systems": "Mathematics, Physics, Electrical Circuits, and Power Systems Basics",
    "Electrical Technology and Power Systems": "Electrical Installation, Circuit Theory, Safety, and Power Systems Basics",
    "Mechanical and Manufacturing Engineering": "Mathematics, Physics, Technical Drawing, and Mechanical Systems Basics",
    "Mechanical Fabrication and Welding Technology": "Welding Safety, Fabrication, Technical Drawing, and Materials Basics",
    "Water, Sanitation and Building Services Technology": "Plumbing Systems, Water Supply, Sanitation, and Technical Drawing",

    "Data Science and Analytics": "Mathematics for Data, Statistics, Python, Excel, and SQL Basics",
    "Statistics and Applied Mathematics": "Statistics, Probability, Excel, and Applied Mathematics Foundations",
    "Computer Science and Information Systems": "Programming Fundamentals, Computer Systems, and Database Basics",
    "Software Engineering and Application Development": "Programming Fundamentals, Web Development Basics, and Problem Solving",
    "Information Technology, Networking and Information Security": "Computer Networking, Systems Administration, and Information Security Basics",
    "Information Technology and Systems Administration": "Computer Hardware, Operating Systems, Networking, and IT Support Basics",
    "Computer Engineering and Embedded Systems": "Electronics Basics, C/C++ Programming, and Embedded Systems Foundations",

    "Finance and Banking": "Business Mathematics, Financial Literacy, Banking Basics, and Excel",
    "Accounting and Finance": "Business Mathematics, Accounting Principles, and Financial Literacy",
    "Economics and Development Finance": "Economics Foundations, Business Mathematics, and Data Interpretation",
    "Business Administration and Management": "Management Principles, Entrepreneurship, Business Communication, and Excel",

    "Geography, GIS and Environmental Planning": "Geography, GIS Basics, Environmental Planning, and Research Skills",
    "Urban and Regional Planning": "Geography, GIS, Urban Studies, and Environmental Planning",

    "Crop Science and Agribusiness": "Crop Production, Soil Science, Agribusiness, and Farm Management Basics",
    "Crop Production and Agribusiness": "Crop Production, Soil Science, Agribusiness, and Farm Management Basics",
    "Food Science and Processing Technology": "Food Safety, Chemistry Foundations, Nutrition, and Processing Basics",
    "Environmental Science and Sustainability": "Environmental Management, Geography, Sustainability, and Research Basics",

    "Public Administration and Governance": "Governance, Public Policy, Academic Writing, and Leadership Basics",
    "Law and Legal Studies": "Academic Writing, Critical Thinking, Governance, and Legal Foundations",
    "International Relations and Diplomacy": "Global Studies, Diplomacy, Communication, and Research Basics",
    "Psychology and Counselling Studies": "Psychology Foundations, Counselling Basics, Academic Writing, and Communication Skills",
    "Sociology and Social Sciences": "Social Sciences Foundations, Research Methods, Academic Writing, and Community Studies",
    "Social Work and Community Development": "Community Development, Social Work Basics, Communication, and Case Management",
    "Development Studies and Community Development": "Development Studies, Research Basics, Community Engagement, and Academic Writing",
    "Education in Arts and Humanities": "Academic Communication, Learning Psychology, and Teaching Methods",

    "Translation and Interpretation": "Advanced Language Skills, Public Speaking, Translation Practice, and Communication",
    "English, Literature and Language Studies": "Academic Writing, Literature Analysis, Communication, and Language Skills",
    "Journalism and Media Studies": "Media Writing, Public Speaking, Digital Media Basics, and Communication",
    "Communication and Public Relations": "Public Speaking, Media Writing, PR Basics, and Digital Communication",
    "Education in Languages": "Language Teaching Methods, Academic Communication, and Education Foundations",

    "Hospitality Management and Culinary Arts": "Food Safety, Nutrition, Kitchen Operations, and Hospitality Communication",
    "Hospitality Management and Food and Beverage Services": "Food and Beverage Service, Customer Care, Hygiene, and Hospitality Communication",
    "Hospitality Management and Room Division": "Front Office Operations, Housekeeping, Customer Care, and Hospitality Systems",
    "Tourism and Travel Management": "Tourism Operations, Customer Care, Geography, and Communication",

    "Multimedia, Graphic Design and Digital Media Production": "Graphic Design Basics, Video Editing, Digital Storytelling, and Creative Software",
    "Fashion Design and Garment Production": "Design Basics, Textile Studies, Pattern Making, and Entrepreneurship",
}

DEFAULT_ALTERNATIVE_PATHWAY = {
    "Medicine and Surgery": "Biomedical Laboratory Sciences, Nursing and Midwifery, Public Health, or foundation science preparation.",
    "Nursing and Midwifery": "Community Health, Public Health, Biomedical Laboratory Sciences, or health-science foundation preparation.",
    "Pharmacy and Pharmaceutical Sciences": "Biomedical Laboratory Sciences, Biotechnology, or chemistry-focused foundation preparation.",
    "Biomedical Laboratory Sciences": "Biotechnology, Public Health, Pharmacy, or laboratory-skills preparation.",
    "Biotechnology and Applied Biosciences": "Biomedical Laboratory Sciences, Environmental Science, Food Science, or applied bioscience preparation.",
    "Civil Engineering and Construction Technology": "Construction Technology, Built Environment, Quantity Surveying, or technical drawing preparation.",
    "Electrical Engineering and Power Systems": "Electrical Technology, Renewable Energy, Electronics, or physics and mathematics preparation.",
    "Electrical Technology and Power Systems": "Electrical Engineering, Renewable Energy, Industrial Electricity, or circuit theory preparation.",
    "Mechanical and Manufacturing Engineering": "Manufacturing Technology, Mechanical Fabrication, Automobile Technology, or technical drawing preparation.",
    "Mechanical Fabrication and Welding Technology": "Mechanical Engineering, Manufacturing Technology, or welding/fabrication certification pathways.",
    "Water, Sanitation and Building Services Technology": "Civil Engineering, Plumbing Technology, Environmental Health, or sanitation systems preparation.",
    "Data Science and Analytics": "Software Engineering, Statistics, Information Systems, or Python/data-skills preparation.",
    "Statistics and Applied Mathematics": "Data Science, Economics, Actuarial Studies, or statistics foundation preparation.",
    "Computer Science and Information Systems": "Software Engineering, Data Science, IT Systems, or programming foundation preparation.",
    "Software Engineering and Application Development": "Computer Science, Information Systems, Data Science, or software project certification.",
    "Information Technology, Networking and Information Security": "Cybersecurity, Systems Administration, Computer Science, or networking certification.",
    "Information Technology and Systems Administration": "Networking, Cybersecurity, IT Support, or systems administration certification.",
    "Computer Engineering and Embedded Systems": "Electronics, Telecommunications, Software Engineering, or embedded-systems certification.",
    "Finance and Banking": "Accounting and Finance, Economics, Business Administration, or financial literacy preparation.",
    "Accounting and Finance": "Finance and Banking, Business Administration, Economics, or accounting certification.",
    "Economics and Development Finance": "Finance, Statistics, Development Studies, or economics foundation preparation.",
    "Business Administration and Management": "Entrepreneurship, Accounting, Finance, Marketing, or business-skills certification.",
    "Geography, GIS and Environmental Planning": "Urban Planning, Environmental Science, Development Studies, or GIS certification.",
    "Urban and Regional Planning": "Geography and GIS, Civil Engineering, Environmental Planning, or urban studies preparation.",
    "Crop Science and Agribusiness": "Agribusiness, Food Science, Environmental Science, or agriculture extension pathways.",
    "Crop Production and Agribusiness": "Crop Science, Agribusiness, Food Processing, or farm management certification.",
    "Food Science and Processing Technology": "Nutrition, Agribusiness, Food Processing, or food-safety certification.",
    "Environmental Science and Sustainability": "GIS, Agriculture, Development Studies, or environmental management certification.",
    "Public Administration and Governance": "Law, Political Science, International Relations, or governance foundation preparation.",
    "Law and Legal Studies": "Public Administration, Governance, International Relations, or legal writing preparation.",
    "International Relations and Diplomacy": "Public Administration, Law, Development Studies, or communication preparation.",
    "Psychology and Counselling Studies": "Social Work, Education, Sociology, or counselling foundation preparation.",
    "Sociology and Social Sciences": "Social Work, Development Studies, Psychology, or research methods preparation.",
    "Social Work and Community Development": "Development Studies, Sociology, Psychology, or community engagement preparation.",
    "Development Studies and Community Development": "Public Administration, Social Work, Economics, or community development preparation.",
    "Education in Arts and Humanities": "Social Sciences, Languages, Public Administration, or teaching-methods preparation.",
    "Translation and Interpretation": "Languages, Communication, International Relations, or public speaking practice.",
    "English, Literature and Language Studies": "Translation, Communication, Education in Languages, or academic writing preparation.",
    "Journalism and Media Studies": "Communication and PR, Multimedia Production, Languages, or digital media preparation.",
    "Communication and Public Relations": "Journalism, Media Studies, Business Communication, or digital communication preparation.",
    "Education in Languages": "Languages, Translation, English Studies, or teaching-methods preparation.",
    "Hospitality Management and Culinary Arts": "Food and Beverage Services, Tourism, Hotel Operations, or hospitality certification.",
    "Hospitality Management and Food and Beverage Services": "Culinary Arts, Hotel Operations, Tourism, or customer-service certification.",
    "Hospitality Management and Room Division": "Hospitality Management, Tourism, Front Office, or housekeeping operations certification.",
    "Tourism and Travel Management": "Hospitality Management, Event Management, Geography, or tour-guiding certification.",
    "Multimedia, Graphic Design and Digital Media Production": "Journalism, Communication, UI/UX Design, or digital media certification.",
    "Fashion Design and Garment Production": "Creative Design, Entrepreneurship, Textile Studies, or fashion production certification.",
}

# =========================================================
# EDUCATION STRUCTURE AND INPUT MAPPINGS
# =========================================================
CORE_GENERAL_EDUCATION_SUBJECTS = ["English", "Kinyarwanda", "Mathematics", "Entrepreneurship"]

GENERAL_EDUCATION_STRUCTURE = {
    "Mathematics and Sciences": {
        "Stream 1": ["Mathematics", "Physics", "Chemistry", "Biology"],
        "Stream 2": ["Mathematics", "Economics", "Geography", "Physics"],
    },
    "Arts and Humanities": {
        "Arts and Humanities": ["History", "Geography", "Literature", "Psychology"],
    },
    "Languages": {
        "Languages": ["English", "French", "Kinyarwanda", "Kiswahili"],
    },
}

RWANDA_INTEREST_AREAS = [
    "Agriculture, Food Processing, and Environment",
    "Arts, Media, and Creative Industries",
    "Business Management and Entrepreneurship",
    "Communication, Marketing, and Public Relations",
    "Construction and Technical Services",
    "Cybersecurity",
    "Data Science, AI, and Machine Learning",
    "Education",
    "Finance, Accounting, and Banking",
    "Hospitality, Tourism, and Service Sector",
    "International Relations and Diplomacy",
    "Languages, Translation, and Interpretation",
    "Law, Governance, and Public Administration",
    "Manufacturing and Industrial Production",
    "Medicine and Health Sciences",
    "Multimedia and Digital Content Production",
    "Networking and Cloud Infrastructure",
    "Psychology, Counseling, and Social Sciences",
    "Science, Engineering, and Mathematics",
    "Software Engineering and Development",
    "Transport and Logistics",
    "UI/UX Design and Digital Product Design",
]

TVET_STRUCTURE = {
    "ICT and Multimedia Sector": {
        "Software Development": ["Software Project Requirements Analysis", "UI/UX Design", "JavaScript Fundamentals", "Web Development", "Database Development", "Machine Learning with Python"],
        "Networking and Internet Technologies": ["Network Basics", "IP Addressing and Subnetting", "Routing and Switching", "Network Security Management", "Linux System Administration", "Network Troubleshooting"],
        "Computer Systems and Architecture": ["Computer Hardware Assembly", "Computer Maintenance and Repair", "Operating Systems Installation", "Basic Electronics", "IT Helpdesk Troubleshooting"],
        "Multimedia Production": ["Graphic Design and Layout", "Photography and Digital Imaging", "Video Editing", "2D Animation", "3D Modeling and Animation", "Motion Graphics"],
        "Software Programming and Embedded Systems": ["C/C++ Programming", "Object-Oriented Programming", "Digital Circuit Electronics", "Microcontroller Programming", "Embedded Linux", "IoT Prototyping"],
    },
    "Construction and Building Services Sector": {
        "Building Construction": ["Technical Architectural Drawing", "Building Materials Technology", "Bricklaying", "Reinforced Concrete Structures", "Estimating and Quantity Surveying", "Computer-Aided Design"],
        "Public Works / Road Construction": ["Road and Highway Engineering", "Soil Mechanics", "Foundation Engineering", "Asphalt and Concrete Pavement Technology", "Drainage System Engineering", "Project Site Management"],
        "Plumbing Technology": ["Technical Drawing for Piping", "Domestic Water Supply Installation", "Drainage and Waste Management Systems", "Plumbing and Pipe-fitting", "Sanitary Fixtures Installation"],
        "Land Surveying": ["Topographical Surveying", "Levelling and Distance Measurements", "Total Station Operations", "Geographic Information Systems", "GPS Mapping"],
        "Interior Design / Painting and Decoration": ["Painting, Decoration and Finishes", "Floor Cladding", "Tiling and Interlocking", "Space Planning and Color Theory", "Architectural Rendering"],
    },
    "Hospitality and Tourism Sector": {
        "Culinary Arts": ["Kitchen Organization", "Food Hygiene and Safety", "Local and International Cuisine", "Baking and Pastry Arts", "Nutrition and Menu Planning"],
        "Food and Beverage Operations": ["Food and Beverage Service Operations", "Table Setting and Etiquette", "Mixology and Bar Operations", "Banquet and Event Catering Service", "Customer Care"],
        "Front Office and Housekeeping Operations": ["Front Office Management", "Property Management Systems", "Guest Relations", "Housekeeping Operations", "Hotel Safety and Security"],
        "Tourism": ["Tour Guiding and Heritage Interpretation", "Travel Agency Operations", "Cultural and Eco-Tourism", "Destination Marketing", "Itinerary Planning"],
    },
    "Energy and Technical Services Sector": {
        "Electrical Technology / Electrical Installation": ["Domestic Electrical Wiring", "Electrical Safety Regulations", "Electrical Testing Instruments", "Electrical Schematics", "Basic Electronics"],
        "Industrial Electricity": ["Three-Phase Motor Control", "Industrial Machinery Installation", "Programmable Logic Controllers", "Substation Operations", "Variable Speed Drives"],
        "Renewable Energy Technology": ["Solar PV Installation", "Solar Thermal Systems", "Battery Storage Configuration", "Hybrid Power Systems", "Energy Efficiency Auditing"],
        "Electronics and Telecommunication": ["Analogue and Digital Electronics", "Telecommunication Infrastructure", "Fiber Optic Installation", "Mobile Communications Technology", "PCB Designing"],
    },
    "Manufacturing, Mining, and Transport Sector": {
        "Automobile Technology / Automobile Mechanic": ["Automotive Engine Repair", "Transmission Systems", "Braking and Steering Systems", "Suspension Systems", "Workshop Safety"],
        "Auto Electricity and Electronic Systems": ["Vehicle Electrical Wiring", "Electronic Fuel Injection", "Automotive Sensor Testing", "Vehicle Diagnostics", "ECU Basics"],
        "Manufacturing and Production Technology": ["Machining Operations", "CNC Operations", "Foundry and Metal Casting", "Mechanical Blueprint Reading", "Precision Measurements"],
        "Welding and Fabrication": ["Shielded Metal Arc Welding", "Gas Metal Arc Welding", "Structural Metal Fabrication", "Mechanical Blueprints", "Welding Safety"],
    },
    "Agriculture and Food Processing Sector": {
        "Crop Production": ["Crop Physiology and Agronomy", "Soil Science", "Plant Pathology", "Irrigation and Water Management", "Post-Harvest Handling"],
        "Animal Health / Livestock Farming": ["Animal Anatomy and Physiology", "Animal Nutrition", "Veterinary First Aid", "Livestock Farming", "Animal Pathology"],
        "Food Processing": ["Milk and Dairy Processing", "Meat and Poultry Processing", "Fruit and Vegetable Preservation", "Food Safety", "Food Packaging"],
        "Forestry and Wood Technology / Carpentry": ["Woodwork", "Carpentry and Joinery", "Furniture Making", "Wood Machining", "Furniture Design"],
    },
    "Business and Arts/Crafts Sector": {
        "Accounting": ["Financial Accounting Principles", "Cost and Management Accounting", "Computerized Accounting", "Tax Law and Declaration", "Auditing Basics"],
        "Fashion Design and Tailoring / Garment Making": ["Fashion Design and Garment Making", "Pattern Making", "Textile Science", "Material Cutting", "Fashion Merchandising"],
        "Fine and Plastic Arts": ["Observational Drawing", "Painting Techniques", "Sculpture", "Graphic Illustration", "Product Design"],
        "Hairdressing and Beauty Therapy / Cosmetology": ["Hair Cutting and Styling", "Makeup Artistry", "Manicure and Pedicure", "Cosmetology Science", "Salon Hygiene and Management"],
    },
}

TVET_TRADE_TO_INTEREST_AREA = {
    "Software Development": "Software Engineering and Development",
    "Networking and Internet Technologies": "Networking and Cloud Infrastructure",
    "Computer Systems and Architecture": "Networking and Cloud Infrastructure",
    "Multimedia Production": "Multimedia and Digital Content Production",
    "Software Programming and Embedded Systems": "Software Engineering and Development",
    "Building Construction": "Construction and Technical Services",
    "Public Works / Road Construction": "Construction and Technical Services",
    "Plumbing Technology": "Construction and Technical Services",
    "Land Surveying": "Construction and Technical Services",
    "Interior Design / Painting and Decoration": "Arts, Media, and Creative Industries",
    "Culinary Arts": "Hospitality, Tourism, and Service Sector",
    "Food and Beverage Operations": "Hospitality, Tourism, and Service Sector",
    "Front Office and Housekeeping Operations": "Hospitality, Tourism, and Service Sector",
    "Tourism": "Hospitality, Tourism, and Service Sector",
    "Electrical Technology / Electrical Installation": "Construction and Technical Services",
    "Industrial Electricity": "Manufacturing and Industrial Production",
    "Renewable Energy Technology": "Science, Engineering, and Mathematics",
    "Electronics and Telecommunication": "Networking and Cloud Infrastructure",
    "Automobile Technology / Automobile Mechanic": "Transport and Logistics",
    "Auto Electricity and Electronic Systems": "Transport and Logistics",
    "Manufacturing and Production Technology": "Manufacturing and Industrial Production",
    "Welding and Fabrication": "Manufacturing and Industrial Production",
    "Crop Production": "Agriculture, Food Processing, and Environment",
    "Animal Health / Livestock Farming": "Agriculture, Food Processing, and Environment",
    "Food Processing": "Agriculture, Food Processing, and Environment",
    "Forestry and Wood Technology / Carpentry": "Agriculture, Food Processing, and Environment",
    "Accounting": "Finance, Accounting, and Banking",
    "Fashion Design and Tailoring / Garment Making": "Arts, Media, and Creative Industries",
    "Fine and Plastic Arts": "Arts, Media, and Creative Industries",
    "Hairdressing and Beauty Therapy / Cosmetology": "Business Management and Entrepreneurship",
}

# These values are intentionally aligned to the final program categories used in the notebook.
TVET_TRADE_TO_CAREER_CLUSTER = {
    "Software Development": "Software Engineering and Application Development",
    "Networking and Internet Technologies": "Information Technology, Networking and Information Security",
    "Computer Systems and Architecture": "Information Technology and Systems Administration",
    "Multimedia Production": "Multimedia, Graphic Design and Digital Media Production",
    "Software Programming and Embedded Systems": "Computer Engineering and Embedded Systems",
    "Building Construction": "Civil Engineering and Construction Technology",
    "Public Works / Road Construction": "Civil Engineering and Construction Technology",
    "Plumbing Technology": "Water, Sanitation and Building Services Technology",
    "Land Surveying": "Geography, GIS and Environmental Planning",
    "Interior Design / Painting and Decoration": "Civil Engineering and Construction Technology",
    "Culinary Arts": "Hospitality Management and Culinary Arts",
    "Food and Beverage Operations": "Hospitality Management and Food and Beverage Services",
    "Front Office and Housekeeping Operations": "Hospitality Management and Room Division",
    "Tourism": "Tourism and Travel Management",
    "Electrical Technology / Electrical Installation": "Electrical Technology and Power Systems",
    "Industrial Electricity": "Electrical Engineering and Power Systems",
    "Renewable Energy Technology": "Electrical Engineering and Power Systems",
    "Electronics and Telecommunication": "Computer Engineering and Embedded Systems",
    "Automobile Technology / Automobile Mechanic": "Mechanical and Manufacturing Engineering",
    "Auto Electricity and Electronic Systems": "Mechanical and Manufacturing Engineering",
    "Manufacturing and Production Technology": "Mechanical and Manufacturing Engineering",
    "Welding and Fabrication": "Mechanical Fabrication and Welding Technology",
    "Crop Production": "Crop Science and Agribusiness",
    "Animal Health / Livestock Farming": "Crop Science and Agribusiness",
    "Food Processing": "Food Science and Processing Technology",
    "Forestry and Wood Technology / Carpentry": "Environmental Science and Sustainability",
    "Accounting": "Accounting and Finance",
    "Fashion Design and Tailoring / Garment Making": "Fashion Design and Garment Production",
    "Fine and Plastic Arts": "Multimedia, Graphic Design and Digital Media Production",
    "Hairdressing and Beauty Therapy / Cosmetology": "Business Administration and Management",
}

PROGRAM_CATEGORY_OPTIONS = list(DEFAULT_PROGRAM_CATEGORY_TO_BRIDGE_COURSE.keys())

INTEREST_AREA_TO_CAREER_CLUSTERS = {
    "Agriculture, Food Processing, and Environment": ["Crop Science and Agribusiness", "Food Science and Processing Technology", "Environmental Science and Sustainability"],
    "Arts, Media, and Creative Industries": ["Multimedia, Graphic Design and Digital Media Production", "Fashion Design and Garment Production", "Journalism and Media Studies"],
    "Business Management and Entrepreneurship": ["Business Administration and Management", "Accounting and Finance", "Finance and Banking"],
    "Communication, Marketing, and Public Relations": ["Communication and Public Relations", "Journalism and Media Studies", "Business Administration and Management"],
    "Construction and Technical Services": ["Civil Engineering and Construction Technology", "Water, Sanitation and Building Services Technology", "Geography, GIS and Environmental Planning"],
    "Cybersecurity": ["Information Technology, Networking and Information Security", "Computer Science and Information Systems"],
    "Data Science, AI, and Machine Learning": ["Data Science and Analytics", "Statistics and Applied Mathematics", "Software Engineering and Application Development"],
    "Education": ["Education in Arts and Humanities", "Education in Languages"],
    "Finance, Accounting, and Banking": ["Accounting and Finance", "Finance and Banking", "Economics and Development Finance"],
    "Hospitality, Tourism, and Service Sector": ["Hospitality Management and Culinary Arts", "Hospitality Management and Food and Beverage Services", "Tourism and Travel Management"],
    "International Relations and Diplomacy": ["International Relations and Diplomacy", "Public Administration and Governance", "Development Studies and Community Development"],
    "Languages, Translation, and Interpretation": ["Translation and Interpretation", "English, Literature and Language Studies", "Education in Languages"],
    "Law, Governance, and Public Administration": ["Law and Legal Studies", "Public Administration and Governance", "International Relations and Diplomacy"],
    "Manufacturing and Industrial Production": ["Mechanical and Manufacturing Engineering", "Mechanical Fabrication and Welding Technology", "Electrical Technology and Power Systems"],
    "Medicine and Health Sciences": ["Medicine and Surgery", "Nursing and Midwifery", "Biomedical Laboratory Sciences", "Pharmacy and Pharmaceutical Sciences"],
    "Multimedia and Digital Content Production": ["Multimedia, Graphic Design and Digital Media Production", "Journalism and Media Studies", "Communication and Public Relations"],
    "Networking and Cloud Infrastructure": ["Information Technology, Networking and Information Security", "Information Technology and Systems Administration", "Computer Science and Information Systems"],
    "Psychology, Counseling, and Social Sciences": ["Psychology and Counselling Studies", "Sociology and Social Sciences", "Social Work and Community Development"],
    "Science, Engineering, and Mathematics": ["Data Science and Analytics", "Statistics and Applied Mathematics", "Civil Engineering and Construction Technology", "Electrical Engineering and Power Systems", "Medicine and Surgery"],
    "Software Engineering and Development": ["Software Engineering and Application Development", "Computer Science and Information Systems", "Data Science and Analytics"],
    "Transport and Logistics": ["Mechanical and Manufacturing Engineering", "Business Administration and Management"],
    "UI/UX Design and Digital Product Design": ["Multimedia, Graphic Design and Digital Media Production", "Software Engineering and Application Development"],
}

# =========================================================
# LOAD MODEL ARTIFACT
# =========================================================
@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at: {MODEL_PATH}. Please place academic_pathway_model.joblib inside the models folder."
        )
        st.stop()

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        model = artifact.get("model")
        label_encoder = artifact.get("label_encoder")
        input_features = artifact.get("input_features", INPUT_FEATURES)
        bridge_course_map = artifact.get("bridge_course_map", {}) or {}
        alternative_pathway_map = artifact.get("alternative_pathway_map", {}) or {}
        model_name = artifact.get("model_name", "Saved Scikit-learn model")
        timestamp = artifact.get("timestamp", "Not specified")
    else:
        model = artifact
        label_encoder = None
        input_features = INPUT_FEATURES
        bridge_course_map = {}
        alternative_pathway_map = {}
        model_name = "Saved Scikit-learn model"
        timestamp = "Not specified"

    if model is None:
        st.error("The model file was loaded, but no model object was found inside it.")
        st.stop()

    if not bridge_course_map:
        bridge_course_map = DEFAULT_PROGRAM_CATEGORY_TO_BRIDGE_COURSE

    if not alternative_pathway_map:
        alternative_pathway_map = DEFAULT_ALTERNATIVE_PATHWAY

    return {
        "model": model,
        "label_encoder": label_encoder,
        "input_features": input_features,
        "bridge_course_map": bridge_course_map,
        "alternative_pathway_map": alternative_pathway_map,
        "model_name": model_name,
        "timestamp": timestamp,
    }

ARTIFACT = load_model_artifact()
MODEL = ARTIFACT["model"]
LABEL_ENCODER = ARTIFACT["label_encoder"]
MODEL_INPUT_FEATURES = ARTIFACT["input_features"]
PROGRAM_CATEGORY_TO_BRIDGE_COURSE = ARTIFACT["bridge_course_map"]
PROGRAM_CATEGORY_TO_ALTERNATIVE_PATHWAY = ARTIFACT["alternative_pathway_map"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def has_term(text, terms):
    text = str(text).lower()
    return any(re.search(rf"\b{re.escape(str(term).lower())}\b", text) for term in terms)


def get_general_education_subjects(pathway, stream):
    subjects = CORE_GENERAL_EDUCATION_SUBJECTS + GENERAL_EDUCATION_STRUCTURE[pathway][stream]
    return list(dict.fromkeys(subjects))


def get_general_education_interest_areas(pathway, stream_or_trade, best_subject, weakest_subject):
    suggested = set()

    if pathway == "Mathematics and Sciences":
        suggested.update([
            "Science, Engineering, and Mathematics",
            "Medicine and Health Sciences",
            "Data Science, AI, and Machine Learning",
            "Software Engineering and Development",
            "Construction and Technical Services",
            "Agriculture, Food Processing, and Environment",
        ])
        if stream_or_trade == "Stream 2":
            suggested.update([
                "Finance, Accounting, and Banking",
                "Business Management and Entrepreneurship",
                "International Relations and Diplomacy",
            ])
    elif pathway == "Arts and Humanities":
        suggested.update([
            "Psychology, Counseling, and Social Sciences",
            "Law, Governance, and Public Administration",
            "International Relations and Diplomacy",
            "Communication, Marketing, and Public Relations",
            "Arts, Media, and Creative Industries",
            "Education",
            "Business Management and Entrepreneurship",
        ])
    elif pathway == "Languages":
        suggested.update([
            "Languages, Translation, and Interpretation",
            "Communication, Marketing, and Public Relations",
            "International Relations and Diplomacy",
            "Arts, Media, and Creative Industries",
            "Education",
        ])

    subject = str(best_subject).lower()
    if subject in ["mathematics", "physics"]:
        suggested.update(["Science, Engineering, and Mathematics", "Data Science, AI, and Machine Learning", "Software Engineering and Development", "Construction and Technical Services"])
    if subject in ["biology", "chemistry"]:
        suggested.update(["Medicine and Health Sciences", "Agriculture, Food Processing, and Environment", "Science, Engineering, and Mathematics"])
    if subject in ["economics", "entrepreneurship"]:
        suggested.update(["Finance, Accounting, and Banking", "Business Management and Entrepreneurship"])
    if subject in ["history", "geography"]:
        suggested.update(["Law, Governance, and Public Administration", "International Relations and Diplomacy", "Geography, GIS and Environmental Planning"])
    if subject in ["literature", "english", "french", "kinyarwanda", "kiswahili"]:
        suggested.update(["Languages, Translation, and Interpretation", "Communication, Marketing, and Public Relations", "Education"])
    if subject == "psychology":
        suggested.update(["Psychology, Counseling, and Social Sciences", "Education", "Medicine and Health Sciences"])

    final = [area for area in RWANDA_INTEREST_AREAS if area in suggested]
    return final if final else RWANDA_INTEREST_AREAS


def get_program_area(program_name):
    program = str(program_name).lower()
    if has_term(program, ["business", "finance", "accounting", "management", "economics", "entrepreneurship", "banking"]):
        return "Business and Finance"
    if has_term(program, ["computer", "software", "information technology", "data", "cyber", "network", "ai", "statistics"]):
        return "ICT and Data"
    if has_term(program, ["medicine", "nursing", "pharmacy", "biomedical", "health", "laboratory", "bioscience"]):
        return "Health Sciences"
    if has_term(program, ["engineering", "civil", "mechanical", "electrical", "construction", "water", "sanitation"]):
        return "Engineering and Applied Sciences"
    if has_term(program, ["agriculture", "food", "crop", "agribusiness", "environment", "sustainability"]):
        return "Agriculture and Environment"
    if has_term(program, ["law", "governance", "public administration", "international relations", "diplomacy"]):
        return "Law and Governance"
    if has_term(program, ["psychology", "counselling", "sociology", "social work", "development studies"]):
        return "Social Sciences"
    if has_term(program, ["translation", "language", "literature", "journalism", "communication", "media"]):
        return "Languages, Media and Communication"
    if has_term(program, ["hospitality", "tourism", "culinary", "food and beverage"]):
        return "Hospitality and Tourism"
    if has_term(program, ["education", "teaching"]):
        return "Education"
    return "General Academic Pathway"


def normalize_program_category(predicted_program):
    """Converts older exact degree outputs into the final program categories used in the notebook."""
    program = str(predicted_program).strip()
    text = program.lower()

    if program in PROGRAM_CATEGORY_TO_BRIDGE_COURSE:
        return program

    mapping_checks = [
        (["tourism", "travel"], "Tourism and Travel Management"),
        (["food and beverage"], "Hospitality Management and Food and Beverage Services"),
        (["hospitality", "hotel", "culinary"], "Hospitality Management and Culinary Arts"),
        (["software", "application development", "programming"], "Software Engineering and Application Development"),
        (["network", "cyber", "security"], "Information Technology, Networking and Information Security"),
        (["computer science", "information systems"], "Computer Science and Information Systems"),
        (["data", "analytics", "machine learning", "ai"], "Data Science and Analytics"),
        (["statistics", "applied mathematics"], "Statistics and Applied Mathematics"),
        (["accounting"], "Accounting and Finance"),
        (["finance", "banking"], "Finance and Banking"),
        (["economics"], "Economics and Development Finance"),
        (["business", "management", "entrepreneurship"], "Business Administration and Management"),
        (["civil", "construction", "building", "road", "public works"], "Civil Engineering and Construction Technology"),
        (["plumbing", "water", "sanitation"], "Water, Sanitation and Building Services Technology"),
        (["electrical", "power", "energy"], "Electrical Engineering and Power Systems"),
        (["mechanical", "manufacturing", "automotive", "automobile"], "Mechanical and Manufacturing Engineering"),
        (["welding", "fabrication"], "Mechanical Fabrication and Welding Technology"),
        (["medicine", "surgery", "doctor", "medical"], "Medicine and Surgery"),
        (["nursing", "midwifery"], "Nursing and Midwifery"),
        (["pharmacy", "pharmaceutical"], "Pharmacy and Pharmaceutical Sciences"),
        (["biomedical", "laboratory"], "Biomedical Laboratory Sciences"),
        (["biotechnology", "bioscience"], "Biotechnology and Applied Biosciences"),
        (["law", "legal"], "Law and Legal Studies"),
        (["public administration", "governance"], "Public Administration and Governance"),
        (["international relations", "diplomacy"], "International Relations and Diplomacy"),
        (["psychology", "counselling", "counseling"], "Psychology and Counselling Studies"),
        (["sociology", "social sciences"], "Sociology and Social Sciences"),
        (["social work"], "Social Work and Community Development"),
        (["development studies", "community development"], "Development Studies and Community Development"),
        (["translation", "interpretation"], "Translation and Interpretation"),
        (["english", "literature", "language"], "English, Literature and Language Studies"),
        (["journalism", "media"], "Journalism and Media Studies"),
        (["communication", "public relations"], "Communication and Public Relations"),
        (["fashion", "garment", "tailoring"], "Fashion Design and Garment Production"),
        (["multimedia", "graphic", "digital media"], "Multimedia, Graphic Design and Digital Media Production"),
        (["crop", "agribusiness", "agriculture"], "Crop Science and Agribusiness"),
        (["food science", "food processing", "nutrition"], "Food Science and Processing Technology"),
        (["environment", "sustainability"], "Environmental Science and Sustainability"),
        (["geography", "gis"], "Geography, GIS and Environmental Planning"),
    ]

    for keywords, category in mapping_checks:
        if any(word in text for word in keywords):
            return category

    return program


def rule_based_program_recommendation(student_profile):
    education_type = str(student_profile.get("EducationType", "")).lower()
    pathway = str(student_profile.get("Pathway", "")).lower()
    stream_or_trade = str(student_profile.get("Stream_or_Trade", "")).lower()
    best_subject = str(student_profile.get("BestSubject", "")).lower()
    weakest_subject = str(student_profile.get("WeakestSubject", "")).lower()
    interest_area = str(student_profile.get("InterestArea", "")).lower()
    career_cluster = str(student_profile.get("CareerCluster", "")).lower()
    digital_skill = str(student_profile.get("DigitalSkillLevel", "")).lower()

    combined = " ".join([education_type, pathway, stream_or_trade, best_subject, weakest_subject, interest_area, career_cluster, digital_skill])
    combined = combined.replace("machine leraning", "machine learning").replace("machin learning", "machine learning").replace("data scince", "data science").replace("maths", "mathematics")

    if has_term(combined, ["food and beverage"]):
        return "Hospitality Management and Food and Beverage Services"
    if has_term(combined, ["culinary"]):
        return "Hospitality Management and Culinary Arts"
    if has_term(combined, ["front office", "housekeeping", "room division"]):
        return "Hospitality Management and Room Division"
    if has_term(combined, ["tourism", "travel"]):
        return "Tourism and Travel Management"
    if has_term(combined, ["fashion", "garment", "tailoring", "textile"]):
        return "Fashion Design and Garment Production"
    if has_term(combined, ["multimedia", "graphic design", "digital media"]):
        return "Multimedia, Graphic Design and Digital Media Production"
    if has_term(combined, ["welding", "fabrication"]):
        return "Mechanical Fabrication and Welding Technology"
    if has_term(combined, ["plumbing", "water", "sanitation"]):
        return "Water, Sanitation and Building Services Technology"
    if has_term(combined, ["construction", "road", "public works", "building", "interior", "painting", "decoration", "civil"]):
        return "Civil Engineering and Construction Technology"

    if has_term(combined, ["nursing"]):
        return "Nursing and Midwifery"
    if has_term(combined, ["medicine", "doctor", "medical"]):
        if has_term(best_subject, ["chemistry", "biology"]) or "mathematics and sciences" in pathway:
            return "Medicine and Surgery"
    if has_term(combined, ["pharmacy", "pharmaceutical"]):
        return "Pharmacy and Pharmaceutical Sciences"
    if has_term(combined, ["biomedical", "laboratory"]):
        return "Biomedical Laboratory Sciences"

    if has_term(combined, ["machine learning", "artificial intelligence", "ai", "data science", "data analytics", "analytics"]):
        return "Data Science and Analytics"
    if has_term(combined, ["statistics", "applied mathematics"]):
        return "Statistics and Applied Mathematics"
    if has_term(combined, ["software", "programming", "application development", "web development"]):
        return "Software Engineering and Application Development"
    if has_term(combined, ["computer science", "information systems"]):
        return "Computer Science and Information Systems"
    if has_term(combined, ["network", "cyber", "security"]):
        return "Information Technology, Networking and Information Security"
    if has_term(combined, ["system administration", "it support"]):
        return "Information Technology and Systems Administration"
    if has_term(combined, ["computer engineering", "embedded", "electronics"]):
        return "Computer Engineering and Embedded Systems"

    if has_term(combined, ["finance", "banking"]):
        return "Finance and Banking"
    if has_term(combined, ["accounting"]):
        return "Accounting and Finance"
    if has_term(combined, ["economics", "development finance"]):
        return "Economics and Development Finance"
    if has_term(combined, ["business", "management", "entrepreneurship"]):
        return "Business Administration and Management"

    if has_term(combined, ["electrical installation"]):
        return "Electrical Technology and Power Systems"
    if has_term(combined, ["electrical", "power", "energy"]):
        return "Electrical Engineering and Power Systems"
    if has_term(combined, ["mechanical", "manufacturing", "automotive", "automobile"]):
        return "Mechanical and Manufacturing Engineering"

    if has_term(combined, ["agriculture", "agribusiness", "crop", "livestock", "animal"]):
        return "Crop Science and Agribusiness"
    if has_term(combined, ["food science", "food processing", "nutrition"]):
        return "Food Science and Processing Technology"
    if has_term(combined, ["environment", "sustainability", "climate", "forestry"]):
        return "Environmental Science and Sustainability"
    if has_term(combined, ["gis", "geography", "surveying"]):
        return "Geography, GIS and Environmental Planning"

    if has_term(combined, ["law", "legal"]):
        return "Law and Legal Studies"
    if has_term(combined, ["governance", "public administration"]):
        return "Public Administration and Governance"
    if has_term(combined, ["international relations", "diplomacy"]):
        return "International Relations and Diplomacy"
    if has_term(combined, ["psychology", "counselling", "counseling"]):
        return "Psychology and Counselling Studies"
    if has_term(combined, ["sociology", "social sciences"]):
        return "Sociology and Social Sciences"
    if has_term(combined, ["social work", "community development"]):
        return "Social Work and Community Development"
    if has_term(combined, ["development studies"]):
        return "Development Studies and Community Development"
    if has_term(combined, ["translation", "interpretation"]):
        return "Translation and Interpretation"
    if has_term(combined, ["english", "literature", "language"]):
        return "English, Literature and Language Studies"
    if has_term(combined, ["journalism", "media studies"]):
        return "Journalism and Media Studies"
    if has_term(combined, ["communication", "public relations"]):
        return "Communication and Public Relations"
    if has_term(combined, ["education", "teaching", "teacher"]):
        if has_term(combined, ["language"]):
            return "Education in Languages"
        return "Education in Arts and Humanities"

    return None


def predict_with_model(student_profile):
    student_df = pd.DataFrame([student_profile])
    for col in MODEL_INPUT_FEATURES:
        if col not in student_df.columns:
            student_df[col] = "Unknown"
    student_df = student_df[MODEL_INPUT_FEATURES]

    raw_prediction = MODEL.predict(student_df)[0]

    # Handles either encoded numeric class, string class, or old multi-output prediction rows.
    if isinstance(raw_prediction, str):
        predicted = raw_prediction
    elif hasattr(raw_prediction, "__len__") and not isinstance(raw_prediction, str):
        predicted = raw_prediction[0]
    else:
        predicted = raw_prediction

    if LABEL_ENCODER is not None:
        try:
            if not isinstance(predicted, str):
                predicted = LABEL_ENCODER.inverse_transform([predicted])[0]
        except Exception:
            predicted = str(predicted)

    return normalize_program_category(predicted)


def recommend_student(student_profile):
    rule_category = rule_based_program_recommendation(student_profile)
    if rule_category:
        category = rule_category
        source = "Rule-based eligibility filter + model-aligned mapping"
    else:
        category = predict_with_model(student_profile)
        source = "Machine learning model"

    category = normalize_program_category(category)
    bridge = PROGRAM_CATEGORY_TO_BRIDGE_COURSE.get(category, "Academic Writing, Study Skills, Digital Literacy, and Career Readiness")
    alternative = PROGRAM_CATEGORY_TO_ALTERNATIVE_PATHWAY.get(category, DEFAULT_ALTERNATIVE_PATHWAY.get(category, "Related diploma, certificate, foundation, or advisor-recommended academic pathway."))
    return category, bridge, alternative, source

def build_explanation(profile, program, bridge, alternative, source):
    education_type = profile.get("EducationType", "the selected education route")
    pathway = profile.get("Pathway", "the selected pathway")
    stream_or_trade = profile.get("Stream_or_Trade", "the selected stream or trade")
    best_subject = profile.get("BestSubject", "the learner’s strongest area")
    weakest_subject = profile.get("WeakestSubject", "the area needing support")
    interest_area = profile.get("InterestArea", "the selected interest area")
    career_cluster = profile.get("CareerCluster", "the selected career direction")

    return (
        f"Based on the learner’s profile, **{program}** is a suitable academic direction. "
        f"The learner is under **{education_type}**, following **{pathway}**, with a focus on "
        f"**{stream_or_trade}**. Their strength in **{best_subject}** supports this pathway, "
        f"while **{weakest_subject}** shows where extra preparation may be helpful.\n\n"
        f"The learner’s interest in **{interest_area}** and career direction toward "
        f"**{career_cluster}** also make this recommendation relevant. To support readiness, "
        f"the system suggests **{bridge}** as a bridge course before progressing further.\n\n"
        f"**Alternative pathway for advisor discussion:** {alternative}\n\n"
        f"**Recommendation source:** {source}."
    )

def make_guidance_report(profile, program, bridge, alternative, source, explanation):
    return f"""# Rwanda Academic Guidance Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Learner Profile
- Education Type: {profile.get('EducationType')}
- Pathway: {profile.get('Pathway')}
- Stream or TVET Trade: {profile.get('Stream_or_Trade')}
- Strongest Subject / Competency: {profile.get('BestSubject')}
- Subject / Competency Needing Support: {profile.get('WeakestSubject')}
- Interest Area: {profile.get('InterestArea')}
- Average Score Range: {profile.get('AverageScoreRange')}
- Digital Skill Level: {profile.get('DigitalSkillLevel')}
- Career Cluster: {profile.get('CareerCluster')}

## Recommendation Output
- Recommended Academic Program Category: {program}
- Recommended Bridge Course: {bridge}
- Alternative Academic Pathway: {alternative}
- Recommendation Source: {source}

## Explanation
{explanation}

## Responsible Use
This report is generated by an academic decision-support prototype. It is not an official MINEDUC, REB, RTB, or university admission decision. Final decisions should be confirmed with official admission requirements, academic advisors, and relevant institutions.
"""


def card(title, body, icon="", class_name="card"):
    st.markdown(f"""
    <div class="{class_name}">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, note, icon):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def result_card(label, value, text, color="green"):
    st.markdown(f"""
    <div class="result-card result-{color}">
        <div class="result-label">{label}</div>
        <div class="result-value">{value}</div>
        <div class="result-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F8FAFC; }
.block-container { max-width: 1400px; padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #031633 0%, #062B5D 56%, #031633 100%); border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] * { color: white; }
.sidebar-brand { padding: 10px 0 22px 0; border-bottom: 1px solid rgba(255,255,255,0.14); margin-bottom: 18px; }
.sidebar-emblem { width: 68px; height: 68px; border-radius: 22px; display:flex; align-items:center; justify-content:center; font-size:34px; background: rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.16); margin-bottom:14px; }
.sidebar-title { font-size: 23px; font-weight: 900; line-height: 1.25; letter-spacing: -0.02em; }
.sidebar-subtitle { color:#CFE3FF; font-size:13px; line-height:1.55; margin-top:10px; }
.sidebar-section-title { color:#A7C8FF; font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.11em; margin:20px 0 10px 0; }
.sidebar-box { background: rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14); border-radius:18px; padding:14px; font-size:13px; line-height:1.65; }
div[role="radiogroup"] > label { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 10px 12px; margin-bottom: 8px; }
div[role="radiogroup"] > label:hover { background: rgba(255,255,255,0.12); }
.top-header { background: rgba(255,255,255,0.88); border:1px solid #E2E8F0; border-radius:22px; padding:14px 18px; display:flex; justify-content:space-between; align-items:center; box-shadow: 0 10px 28px rgba(15,23,42,0.06); margin-bottom:18px; }
.header-left { display:flex; gap:12px; align-items:center; }
.header-icon { width:42px; height:42px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#DBEAFE,#ECFDF5); font-size:22px; }
.header-kicker { color:#0B5ED7; font-size:12px; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }
.header-title { color:#0F172A; font-size:16px; font-weight:850; margin-top:2px; }
.header-user { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:18px; padding:8px 12px; color:#334155; font-size:13px; }
.hero-card { background: linear-gradient(135deg, #FFFFFF 0%, #F3F8FF 58%, #EEF6FF 100%); border:1px solid #D7E8FF; border-radius:28px; padding:36px 42px; box-shadow:0 18px 45px rgba(15,23,42,0.08); margin-bottom:22px; }
.hero-grid { display:grid; grid-template-columns: 1.8fr .8fr; gap:28px; align-items:center; }
.hero-kicker { color:#0B5ED7; font-size:12px; font-weight:900; letter-spacing:.1em; text-transform:uppercase; margin-bottom:10px; }
.hero-title { color:#061B3A; font-size:39px; font-weight:950; line-height:1.08; letter-spacing:-.04em; margin-bottom:14px; }
.hero-subtitle { color:#1D4ED8; font-size:18px; line-height:1.55; font-weight:800; margin-bottom:12px; }
.hero-body { color:#475569; font-size:16px; line-height:1.75; }
.hero-illustration { min-height:185px; border-radius:26px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,rgba(255,255,255,.7),rgba(219,234,254,.82)); border:1px solid #DBEAFE; font-size:78px; }
.badge { border-radius:999px; padding:8px 13px; font-size:12px; font-weight:850; display:inline-flex; gap:7px; margin-right:8px; margin-top:14px; }
.badge-blue { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; } .badge-green { background:#ECFDF5; color:#047857; border:1px solid #A7F3D0; } .badge-gold { background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; } .badge-purple { background:#F5F3FF; color:#6D28D9; border:1px solid #DDD6FE; }
.section-card, .panel-card, .metric-card, .card, .result-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:22px; box-shadow:0 10px 30px rgba(15,23,42,0.055); }
.section-card { padding:28px; margin-bottom:22px; }
.panel-card { padding:24px; margin-bottom:18px; }
.section-title { color:#061B3A; font-size:24px; font-weight:950; letter-spacing:-.03em; margin-bottom:12px; }
.section-body { color:#475569; font-size:15px; line-height:1.75; }
.metric-card { padding:20px; display:flex; gap:15px; align-items:center; min-height:125px; }
.metric-icon { width:52px; height:52px; border-radius:18px; background:#EFF6FF; display:flex; align-items:center; justify-content:center; font-size:24px; }
.metric-label { color:#475569; font-size:13px; font-weight:800; line-height:1.35; }
.metric-value { color:#061B3A; font-size:30px; font-weight:950; letter-spacing:-.04em; margin:4px 0; }
.metric-note { color:#64748B; font-size:12px; line-height:1.4; }
.card { padding:22px; min-height:155px; }
.card-icon { font-size:28px; margin-bottom:8px; }
.card-title { color:#0F172A; font-size:16px; font-weight:900; margin-bottom:8px; }
.card-body { color:#64748B; font-size:13.5px; line-height:1.6; }
.result-card { padding:22px; margin-bottom:16px; }
.result-green { background:linear-gradient(135deg,#ECFDF5,#FFFFFF); border-color:#A7F3D0; }
.result-blue { background:linear-gradient(135deg,#EFF6FF,#FFFFFF); border-color:#BFDBFE; }
.result-purple { background:linear-gradient(135deg,#F5F3FF,#FFFFFF); border-color:#DDD6FE; }
.result-gold { background:linear-gradient(135deg,#FFFBEB,#FFFFFF); border-color:#FDE68A; }
.result-label { font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.07em; color:#475569; margin-bottom:7px; }
.result-value { font-size:25px; font-weight:950; line-height:1.22; letter-spacing:-.035em; color:#061B3A; margin-bottom:8px; }
.result-text { color:#475569; font-size:14px; line-height:1.65; }
.info-box { background:#EFF6FF; border:1px solid #BFDBFE; color:#1E3A8A; padding:18px 20px; border-radius:18px; line-height:1.7; font-size:14px; margin:16px 0; }
.warning-box { background:#FFF7ED; border:1px solid #FDBA74; color:#7C2D12; padding:18px 20px; border-radius:18px; line-height:1.7; font-size:14px; margin:16px 0; }
.footer { margin-top:28px; padding:20px 8px 8px 8px; border-top:1px solid #E2E8F0; color:#64748B; font-size:13px; line-height:1.7; text-align:center; }
div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button { border-radius:14px !important; border:none !important; background:linear-gradient(135deg,#2563EB 0%,#0B5ED7 55%,#063A8C 100%) !important; color:white !important; font-weight:850 !important; box-shadow:0 10px 20px rgba(37,99,235,.22) !important; padding:.78rem 1rem !important; }
.stSelectbox label, .stSlider label, .stTextArea label { font-weight:800 !important; color:#334155 !important; }
@media (max-width:1100px) { .hero-grid { grid-template-columns:1fr; } }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR AND HEADER
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-emblem">🎓</div>
        <div class="sidebar-title">Rwanda Academic<br>Guidance Portal</div>
        <div class="sidebar-subtitle">Academic pathway, program category, and bridge-course decision-support prototype.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Main Navigation</div>', unsafe_allow_html=True)
    pages = ["🏠 Home", "🧭 Get Recommendation", "📊 Advisor Dashboard", "🧪 Methodology", "🛡️ Data & Governance", "✅ Responsible Use", "💬 Feedback"]
    selected_page = st.radio("", pages, label_visibility="collapsed")

    st.markdown('<div class="sidebar-section-title">System Information</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-box">
        <b>Model:</b> {ARTIFACT['model_name']}<br>
        <b>Model file:</b> academic_pathway_model.joblib<br>
        <b>Dataset version:</b> 2026.06<br>
        <b>Use:</b> Advisory decision support
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Rwanda Education Context</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-box">
        Aligned with Rwanda academic guidance, TVET progression, bridge-course readiness, and responsible AI principles.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="top-header">
    <div class="header-left">
        <div class="header-icon">🇷🇼</div>
        <div>
            <div class="header-kicker">Rwanda Education Context</div>
            <div class="header-title">Academic Guidance Decision-Support Prototype</div>
        </div>
    </div>
    <div class="header-user">👤 Learner / Advisor · Advisory Use Only</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PAGES
# =========================================================
if selected_page == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-grid">
            <div>
                <div class="hero-kicker">Decision-Support Prototype • Explainable Guidance • Rwanda Context</div>
                <div class="hero-title">Welcome to the Rwanda Academic Guidance Portal</div>
                <div class="hero-subtitle">Academic Pathway, Program Category, and Bridge Course Recommendation System</div>
                <div class="hero-body">
                    A professional decision-support prototype that helps Rwandan learners explore suitable academic program categories,
                    preparatory bridge courses, and alternative pathways based on academic background, interests, digital skills, and career goals.
                </div>
                <span class="badge badge-blue">🔎 Prototype</span>
                <span class="badge badge-gold">⚠️ Advisory Use Only</span>
                <span class="badge badge-green">🛡️ Explainable AI</span>
                <span class="badge badge-purple">🇷🇼 Rwanda Context</span>
            </div>
            <div class="hero-illustration">🎓📚</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Survey Responses", "105", "Collected from students and advisors", "👥")
    with c2: metric_card("Program Recommendation Requested", "53.3%", "56 out of 105 respondents", "🎓")
    with c3: metric_card("Career Guidance Requested", "55.2%", "58 out of 105 respondents", "📖")
    with c4: metric_card("Willing to Use Bridge Courses", "88.6%", "93 out of 105 respondents", "🌉")
    with c5: metric_card("Willing to Test Prototype", "76.2%", "80 out of 105 respondents", "✅")

    st.markdown('<div class="section-card"><div class="section-title">How It Works</div><div class="section-body">The system follows a simple, explainable guidance workflow.</div></div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1: card("1. Learner Profile", "Learner enters education type, pathway, strengths, support needs, interests, and career direction.", "👤")
    with s2: card("2. Data Processing", "The profile is structured into the same input features used by the trained model.", "🧾")
    with s3: card("3. Eligibility Filter", "Strong TVET, subject, interest, and career signals are checked first.", "🧭")
    with s4: card("4. ML Prediction", "The saved Scikit-learn model predicts a program category where needed.", "🤖")
    with s5: card("5. Bridge Mapping", "The bridge course is mapped from the predicted program category.", "📚")
    with s6: card("6. Advisor Review", "The recommendation is explained for discussion with an advisor.", "💬")

elif selected_page == "🧭 Get Recommendation":
    left, right = st.columns([1, 1.35])

    with left:
        with st.container(border=True):
            st.markdown("### 1. Learner Academic Profile")
            st.caption("Enter the learner’s academic background and interests. The system will generate an explainable recommendation.")

            education_type = st.selectbox("Education Type", ["General Education", "TVET"], key="education_type")

            if education_type == "General Education":
                pathway = st.selectbox("REB General Education Section", list(GENERAL_EDUCATION_STRUCTURE.keys()), key="pathway")
                stream_or_trade = st.selectbox("Stream / Section", list(GENERAL_EDUCATION_STRUCTURE[pathway].keys()), key="stream")
                subjects = get_general_education_subjects(pathway, stream_or_trade)
                best_subject = st.selectbox("Strongest Subject", subjects, key="best_subject")
                weakest_subject = st.selectbox("Subject Needing Support", subjects, key="weakest_subject")
                interest_options = get_general_education_interest_areas(pathway, stream_or_trade, best_subject, weakest_subject)
                interest_area = st.selectbox("Interest Area", interest_options, key="interest_area")
                career_options = INTEREST_AREA_TO_CAREER_CLUSTERS.get(interest_area, PROGRAM_CATEGORY_OPTIONS)
                career_cluster = st.selectbox("Career Cluster", career_options, key="career_cluster")
            else:
                pathway = "TVET Route"
                tvet_sector = st.selectbox("TVET Sector", list(TVET_STRUCTURE.keys()), key="tvet_sector")
                stream_or_trade = st.selectbox("TVET Trade", list(TVET_STRUCTURE[tvet_sector].keys()), key="tvet_trade")
                courses = TVET_STRUCTURE[tvet_sector][stream_or_trade]
                best_subject = st.selectbox("Strongest Course / Competency", courses, key="best_competency")
                weakest_subject = st.selectbox("Course / Competency Needing Support", courses, key="weak_competency")

                default_interest = TVET_TRADE_TO_INTEREST_AREA.get(stream_or_trade, "Science, Engineering, and Mathematics")
                interest_options = [default_interest] + [x for x in RWANDA_INTEREST_AREAS if x != default_interest]
                interest_area = st.selectbox(
                    "Interest Area",
                    interest_options,
                    key="tvet_interest_area",
                    help="A suggested interest area is selected from the TVET trade, but it can be changed if needed.",
                )

                default_career = TVET_TRADE_TO_CAREER_CLUSTER.get(stream_or_trade, "Business Administration and Management")
                career_options = [default_career] + [x for x in PROGRAM_CATEGORY_OPTIONS if x != default_career]
                career_cluster = st.selectbox(
                    "Career Cluster",
                    career_options,
                    key="tvet_career_cluster",
                    help="A suggested career cluster is selected from the TVET trade, but it can be changed if needed.",
                )

            average_score_range = st.selectbox("Average Score Range", ["50–59%", "60–69%", "70–79%", "80–89%", "90–100%"], key="score_range")
            digital_skill_level = st.selectbox("Digital Skill Level", ["Beginner", "Intermediate", "Advanced"], key="digital_skill")
            submitted = st.button("Generate Recommendation", key="submit_recommendation")

    with right:
        with st.container(border=True):
            st.markdown("### 2. Recommendation Output")
            st.caption("The output combines the trained model with a transparent guidance layer for bridge-course alignment.")

            if submitted:
                profile = {
                    "EducationType": education_type,
                    "Pathway": pathway,
                    "Stream_or_Trade": stream_or_trade,
                    "BestSubject": best_subject,
                    "WeakestSubject": weakest_subject,
                    "InterestArea": interest_area,
                    "AverageScoreRange": average_score_range,
                    "DigitalSkillLevel": digital_skill_level,
                    "CareerCluster": career_cluster,
                }

                try:
                    recommended_program, recommended_bridge_course, alternative_pathway, source = recommend_student(profile)
                except Exception as e:
                    st.error("The system could not process this learner profile. Please confirm that the saved model matches the dashboard input features.")
                    st.exception(e)
                    st.stop()

                explanation = build_explanation(profile, recommended_program, recommended_bridge_course, alternative_pathway, source)

                result_card("Recommended Academic Program Category", recommended_program, "This category is produced from the learner profile and the notebook-aligned recommendation logic.", "green")
                result_card("Recommended Bridge Course", recommended_bridge_course, "This bridge course is mapped directly from the recommended program category.", "blue")
                result_card("Alternative Academic Pathway", alternative_pathway, "This provides a second route for discussion with an academic advisor.", "purple")
                result_card("Recommendation Source", source, "The system shows whether the result came from the eligibility filter or the ML model.", "gold")

                st.markdown("### Explanation of the Recommendation")
                st.markdown(explanation)

                report = make_guidance_report(profile, recommended_program, recommended_bridge_course, alternative_pathway, source, explanation)
                st.download_button(
                    "Download Guidance Report",
                    data=report,
                    file_name="rwanda_academic_guidance_report.md",
                    mime="text/markdown",
                )
            else:
                result_card("Recommended Academic Program Category", "Awaiting learner profile", "Complete the form and click Generate Recommendation to view the suggested category.", "green")
                result_card("Recommended Bridge Course", "Awaiting recommendation", "The bridge course will appear after the system processes the profile.", "blue")
                result_card("Explainability Summary", "Awaiting profile", "The explanation will show the factors considered by the system.", "purple")

elif selected_page == "📊 Advisor Dashboard":
    st.markdown('<div class="section-card"><div class="section-title">Advisor Dashboard</div><div class="section-body">This section summarizes prototype evaluation indicators for advisor review and stakeholder discussion.</div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Usefulness Rating", "4/5", "Prototype evaluation indicator", "⭐")
    with c2: metric_card("Explanation Clarity", "4/5", "Advisor-facing transparency", "💡")
    with c3: metric_card("Usability Rating", "4/5", "Interface ease of use", "🖥️")
    with c4: metric_card("Feedback Collection", "Active", "Stored in feedback_responses.csv", "💬")
    st.markdown('<div class="info-box"><b>Advisor note:</b> This dashboard is for prototype monitoring and presentation. It does not represent official national analytics.</div>', unsafe_allow_html=True)

elif selected_page == "🧪 Methodology":
    st.markdown('<div class="section-card"><div class="section-title">System Methodology</div><div class="section-body">The system follows the hybrid architecture described in the research proposal: learner profile input, rule-based eligibility filtering, machine learning prediction, bridge-course mapping, alternative pathway suggestion, explainability, and feedback collection.</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: card("Input Features", "Education Type, Pathway, Stream or TVET Trade, Strongest Subject, Weakest Subject, Interest Area, Average Score Range, Digital Skill Level, and Career Cluster.", "🧾")
    with c2: card("Prediction Layer", "The saved model predicts a recommended academic program category when a direct eligibility rule is not enough.", "🤖")
    with c3: card("Bridge Mapping", "The bridge course is mapped from the recommended program category to keep the output consistent with the training notebook.", "📚")
    c4, c5 = st.columns(2)
    with c4: card("Explainability", "The explanation summarizes how the learner profile connects to the recommendation.", "💬")
    with c5: card("Responsible Use", "The output is advisory and should be reviewed with official entry requirements and academic advisors.", "🛡️")

elif selected_page == "🛡️ Data & Governance":
    st.markdown('<div class="section-card"><div class="section-title">Data & Governance</div><div class="section-body">The prototype is designed around data minimization, transparent outputs, and human review. It avoids unnecessary personal identifiers and keeps the recommendation explainable.</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: card("Data Minimization", "The app does not require national ID, exact address, phone number, or sensitive personal identifiers.", "🔐")
    with c2: card("Human Review", "Recommendations should be discussed with advisors, institutions, parents, or guardians before decisions are made.", "👥")
    with c3: card("Model Consistency", "The app loads the saved model artifact and uses the same program-to-bridge-course mapping used in training.", "🧠")
    st.markdown('<div class="warning-box"><b>Important:</b> This is a capstone prototype and not an official government system.</div>', unsafe_allow_html=True)

elif selected_page == "✅ Responsible Use":
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Responsible Use and Professional Disclaimer</div>
        <div class="section-body">
            This system is an academic research prototype and decision-support tool. It supports exploration, counseling, and discussion. It should not be used as the final authority for university admission, scholarship selection, program placement, or official education policy decisions.
            <br><br>
            Students and guardians should verify recommendations with official admission requirements, school leadership, academic advisors, MINEDUC-related guidance, REB resources, RTB guidance, and relevant higher education institutions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="warning-box"><b>Important Disclaimer:</b> The recommendation is not an official MINEDUC, REB, RTB, or university admission decision. It does not guarantee admission into any program.</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>Ethical design:</b> The system avoids unnecessary personal identifiers and includes an explanation layer to support transparency, fairness, and human review.</div>', unsafe_allow_html=True)

elif selected_page == "💬 Feedback":
    st.markdown('<div class="section-card"><div class="section-title">System Feedback</div><div class="section-body">This section collects feedback on usefulness, explanation clarity, and usability for prototype evaluation.</div></div>', unsafe_allow_html=True)

    with st.form("feedback_form"):
        user_role = st.selectbox("Your Role", ["Student", "Academic Advisor", "Teacher", "Parent/Guardian", "Education Stakeholder", "Other"])
        usefulness = st.slider("How useful is the recommendation output?", 1, 5, 4)
        clarity = st.slider("How clear is the explanation?", 1, 5, 4)
        usability = st.slider("How easy is the system to use?", 1, 5, 4)
        comments = st.text_area("Additional comments", placeholder="Share suggestions to improve the recommendation, explanation, or dashboard design.")
        feedback_submitted = st.form_submit_button("Submit Feedback")

    if feedback_submitted:
        feedback_file = BASE_DIR / "feedback_responses.csv"
        feedback_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": user_role,
            "usefulness": usefulness,
            "clarity": clarity,
            "usability": usability,
            "comments": comments,
        }
        feedback_df = pd.DataFrame([feedback_record])
        if feedback_file.exists():
            feedback_df.to_csv(feedback_file, mode="a", header=False, index=False)
        else:
            feedback_df.to_csv(feedback_file, index=False)
        result_card("Feedback Submitted Successfully", "Thank you", "Your feedback has been recorded and will support prototype evaluation.", "green")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    <b>Rwanda Academic Guidance Portal — Decision-Support Prototype</b><br>
    Academic pathway, program category, bridge-course readiness, and alternative pathway exploration.<br>
    For advisory use only. Final academic decisions should be confirmed using official admission requirements, institutional guidance, and professional academic counseling.
</div>
""", unsafe_allow_html=True)
