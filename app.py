import base64
import re
from functools import lru_cache
from urllib.parse import quote
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# LOCAL PROJECT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "academic_pathway_model.joblib"

IMAGE_DIR = BASE_DIR / "static" / "images"
ICON_DIR = BASE_DIR / "static" / "icons"
BRANDING_DIR = BASE_DIR / "static" / "branding"
ICON_DIR = BASE_DIR / "static" / "icons"
BRANDING_DIR = BASE_DIR / "static" / "branding"

# Branding assets
PORTAL_LOGO = BRANDING_DIR / "portal_logo.png"
PORTAL_MARK_SVG = BRANDING_DIR / "portal_mark.svg"
PORTAL_MARK_PNG = BRANDING_DIR / "portal_mark.png"
PORTAL_LOGO_LIGHT = BRANDING_DIR / "portal_logo_light.png"
RWANDA_WATERMARK = BRANDING_DIR / "rwanda_map_watermark.svg"

# Real photographs (rule 10: banners and report area only)
HERO_PHOTO = IMAGE_DIR / "hero_graduate_rear_view.jpg"
LEARNER_PHOTO = IMAGE_DIR / "learner_profile_faceless.jpg"
BRIDGE_NOTES_PHOTO = IMAGE_DIR / "bridge_course_notes_faceless.jpg"
COLLABORATION_PHOTO = IMAGE_DIR / "student_collaboration_faceless.jpg"
ADVISOR_PHOTO = IMAGE_DIR / "advisor_review_faceless.jpg"
METHODOLOGY_PHOTO = IMAGE_DIR / "methodology_teacher_rear_view.jpg"
FEEDBACK_PHOTO = IMAGE_DIR / "feedback_writing_faceless.jpg"
ANALYTICS_PHOTO = IMAGE_DIR / "analytics_laptop_graphs.png"

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Rwanda Academic Guidance Portal",
    page_icon=str(PORTAL_MARK_PNG) if PORTAL_MARK_PNG.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# MODEL LOADING
# =========================================================

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

# =========================================================
# VISUAL ASSET HELPERS (icons, photos, watermark, banners)
# =========================================================
@lru_cache(maxsize=64)
def _load_icon_svg(rel_path):
    """Read an SVG from static/icons/, stripping the license comment for inline use."""
    path = ICON_DIR / rel_path
    if not path.exists():
        return ""
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.DOTALL)
    svg = re.sub(r"\s+", " ", svg)  # single line: multi-line HTML inside st.markdown can break parsing
    return svg.strip()


def icon(rel_path, color="#1D4ED8", size=22):
    """Return inline SVG markup tinted via currentColor. rel_path e.g. 'metrics/survey-responses.svg'."""
    svg = _load_icon_svg(rel_path)
    if not svg:
        return ""
    svg = svg.replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')
    return (
        f'<span class="svg-icon" style="color:{color};width:{size}px;height:{size}px;">{svg}</span>'
    )


def image_data_uri(image_path):
    """Convert a local image (png, jpg, or svg) into an HTML-safe data URI."""
    path = Path(image_path)
    if not path.exists():
        return ""

    suffix = path.suffix.lower()
    if suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".svg":
        mime_type = "image/svg+xml"
    else:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def render_html(html):
    """Render HTML safely: strip blank/whitespace-only lines, which would make
    Streamlit's markdown parser terminate the HTML block and print raw code."""
    cleaned = "\n".join(line for line in html.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def watermark_data_uri():
    """Rwanda map silhouette as a navy-tinted data URI for the fixed page watermark."""
    if not RWANDA_WATERMARK.exists():
        return ""
    svg = RWANDA_WATERMARK.read_text(encoding="utf-8").replace("currentColor", "#12356B")
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def banner(photo_path, kicker, title, body):
    """Full-width photo banner with the portal's navy overlay (rule 10: one real photograph)."""
    photo_uri = image_data_uri(photo_path)
    photo_tag = f'<img class="banner-photo" src="{photo_uri}" alt="{title}">' if photo_uri else ""
    render_html(
        f"""
        <div class="banner">
            {photo_tag}
            <div class="banner-overlay"></div>
            <div class="banner-content">
                <div class="banner-kicker">{kicker}</div>
                <div class="banner-title">{title}</div>
                <div class="banner-body">{body}</div>
            </div>
        </div>
        """
    )


def icon_card(title, body, icon_rel, icon_color="#1D4ED8"):
    """Information card with an SVG icon chip (rule 10: cards use SVG icons, not photos)."""
    render_html(
        f"""
        <div class="icon-card">
            <div class="icon-chip">{icon(icon_rel, icon_color, 26)}</div>
            <div class="card-title">{title}</div>
            <div class="card-body">{body}</div>
        </div>
        """
    )


METRIC_ACCENTS = {
    "blue":   ("#EFF6FF", "#BFDBFE", "#1D4ED8"),
    "green":  ("#ECFDF5", "#A7F3D0", "#047857"),
    "gold":   ("#FFFBEB", "#FDE68A", "#B45309"),
    "purple": ("#F5F3FF", "#DDD6FE", "#6D28D9"),
    "teal":   ("#F0FDFA", "#99F6E4", "#0F766E"),
}

def metric_card(label, value, note, icon_rel, accent="blue"):
    """KPI card with an SVG metric icon on a colored accent chip."""
    bg, border, fg = METRIC_ACCENTS.get(accent, METRIC_ACCENTS["blue"])
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-icon" style="background:{bg};border-color:{border};">{icon(icon_rel, fg, 26)}</div>
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
        </div>
        """
    )


def quick_action(title, subtitle, icon_rel, page):
    """Fully clickable action card (spec: icon + text + arrow, no separate button)."""
    href = "?nav=" + quote(page)
    render_html(
        f"""
        <a class="qa-link" href="{href}" target="_self">
            <span class="qa-icon">{icon("actions/" + icon_rel, "#1D4ED8", 24)}</span>
            <span class="qa-text">
                <span class="qa-title">{title}</span>
                <span class="qa-sub">{subtitle}</span>
            </span>
            <span class="qa-arrow">{icon("actions/arrow-right.svg", "#94A3B8", 18)}</span>
        </a>
        """
    )


def result_card(label, value, text, color="green"):
    render_html(f"""
    <div class="result-card result-{color}">
        <div class="result-label">{label}</div>
        <div class="result-value">{value}</div>
        <div class="result-text">{text}</div>
    </div>
    """)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F8FAFC; }
.block-container { max-width: 1400px; padding-top: 1.2rem; padding-bottom: 2rem; position: relative; z-index: 1; }
            
  /* Sidebar outer shell */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #031633 0%,
        #062B5D 56%,
        #031633 100%
    ) !important;

    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;

    border-right: 1px solid rgba(255,255,255,0.08) !important;
}

/* Remove the white inner panel appearing after deployment */
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    background: transparent !important;
}

/* Sidebar content spacing */
section[data-testid="stSidebar"]
[data-testid="stSidebarUserContent"] {
    padding: 1.4rem 1rem 1.5rem !important;
}

/* Keep sidebar text readable */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .sidebar-title {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] .sidebar-subtitle {
    color: #CFE3FF !important;
}

section[data-testid="stSidebar"] .sidebar-section-title {
    color: #A7C8FF !important;
}

/* Hide Streamlit's automatic multipage navigation */
section[data-testid="stSidebar"]
[data-testid="stSidebarNav"] {
    display: none !important;
}          
.svg-icon { display:inline-flex; align-items:center; justify-content:center; flex:none; }
.svg-icon svg { width:100%; height:100%; display:block; }
.sidebar-brand { padding: 10px 0 22px 0; border-bottom: 1px solid rgba(255,255,255,0.14); margin-bottom: 18px; }
.sidebar-emblem { width:72px; height:72px; border-radius:18px; overflow:hidden; background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.16); margin-bottom:14px; }
.sidebar-emblem img { width:100%; height:100%; object-fit:cover; object-position:center; display:block; }
.sidebar-logo { width:100%; max-width:236px; display:block; margin:6px 0 12px; }
.sidebar-title { font-size: 23px; font-weight: 900; line-height: 1.25; letter-spacing: -0.02em; }
.sidebar-subtitle { color:#CFE3FF; font-size:13px; line-height:1.55; margin-top:10px; }
.sidebar-section-title { color:#A7C8FF; font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.11em; margin:20px 0 10px 0; }
.sidebar-box { background: rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14); border-radius:18px; padding:14px; font-size:13px; line-height:1.65; }
.top-header { background: rgba(255,255,255,0.88); border:1px solid #E2E8F0; border-radius:22px; padding:14px 18px; display:flex; justify-content:space-between; align-items:center; box-shadow: 0 10px 28px rgba(15,23,42,0.06); margin-bottom:18px; }
.header-left { display:flex; gap:12px; align-items:center; }
.header-mark { width:42px; height:42px; border-radius:14px; overflow:hidden; border:1px solid #D7E8FF; box-shadow:0 4px 12px rgba(15,23,42,.08); }
.header-mark img { width:100%; height:100%; object-fit:cover; display:block; }
.header-kicker { color:#0B5ED7; font-size:12px; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }
.header-title { color:#0F172A; font-size:16px; font-weight:850; margin-top:2px; }
.header-right { display:flex; gap:10px; align-items:center; }
.header-control { width:38px; height:38px; border-radius:14px; background:#F8FAFC; border:1px solid #E2E8F0; display:flex; align-items:center; justify-content:center; }
.header-user { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:18px; padding:8px 12px; color:#334155; font-size:13px; display:flex; gap:8px; align-items:center; }
.hero-card { background: linear-gradient(135deg, #FFFFFF 0%, #F3F8FF 58%, #EEF6FF 100%); border:1px solid #D7E8FF; border-radius:28px; padding:36px 42px; box-shadow:0 18px 45px rgba(15,23,42,0.08); margin-bottom:22px; }
.hero-grid { display:grid; grid-template-columns: 1.8fr .8fr; gap:28px; align-items:center; }
.hero-kicker { color:#0B5ED7; font-size:12px; font-weight:900; letter-spacing:.1em; text-transform:uppercase; margin-bottom:10px; }
.hero-title { color:#061B3A; font-size:39px; font-weight:950; line-height:1.08; letter-spacing:-.04em; margin-bottom:14px; }
.hero-subtitle { color:#1D4ED8; font-size:18px; line-height:1.55; font-weight:800; margin-bottom:12px; }
.hero-body { color:#475569; font-size:16px; line-height:1.75; }
.hero-illustration { min-height:250px; border-radius:26px; overflow:hidden; background:#EAF2FA; border:1px solid #DBEAFE; box-shadow:0 12px 28px rgba(15,23,42,.10); }
.hero-illustration img { width:100%; height:250px; display:block; object-fit:cover; object-position:center 30%; }
.badge { border-radius:999px; padding:8px 13px; font-size:12px; font-weight:850; display:inline-flex; gap:7px; margin-right:8px; margin-top:14px; }
.badge-blue { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; } .badge-green { background:#ECFDF5; color:#047857; border:1px solid #A7F3D0; } .badge-gold { background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; } .badge-purple { background:#F5F3FF; color:#6D28D9; border:1px solid #DDD6FE; }
.section-card, .panel-card, .metric-card, .icon-card, .result-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:22px; box-shadow:0 10px 30px rgba(15,23,42,0.055); }
.section-card { padding:28px; margin-bottom:22px; }
.panel-card { padding:24px; margin-bottom:18px; }
.section-title { color:#061B3A; font-size:24px; font-weight:950; letter-spacing:-.03em; margin-bottom:12px; }
.section-body { color:#475569; font-size:15px; line-height:1.75; }
.metric-card { padding:20px; display:flex; gap:15px; align-items:center; min-height:125px; }
.metric-icon { min-width:52px; width:52px; height:52px; border-radius:18px; background:#EFF6FF; display:flex; align-items:center; justify-content:center; border:1px solid #BFDBFE; }
.metric-label { color:#475569; font-size:13px; font-weight:800; line-height:1.35; }
.metric-value { color:#061B3A; font-size:30px; font-weight:950; letter-spacing:-.04em; margin:4px 0; }
.metric-note { color:#64748B; font-size:12px; line-height:1.4; }
.icon-card { padding:22px 20px; min-height:190px; }
.icon-chip { width:52px; height:52px; border-radius:16px; background:#EFF6FF; border:1px solid #BFDBFE; display:flex; align-items:center; justify-content:center; margin-bottom:14px; }
.card-title { color:#0F172A; font-size:16px; font-weight:900; margin-bottom:8px; }
.card-body { color:#64748B; font-size:13.5px; line-height:1.6; }
.banner { position:relative; border-radius:26px; overflow:hidden; min-height:200px; margin-bottom:22px; box-shadow:0 14px 34px rgba(15,23,42,0.10); display:flex; align-items:flex-end; }
.banner-photo { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; object-position:center; }
.banner-overlay { position:absolute; inset:0; background:linear-gradient(90deg, rgba(6,27,58,0.90) 0%, rgba(6,27,58,0.58) 55%, rgba(0,161,222,0.28) 100%); }
.banner-content { position:relative; z-index:2; padding:28px 32px; max-width:760px; }
.banner-kicker { color:#9CD3F5; font-size:12px; font-weight:900; letter-spacing:.1em; text-transform:uppercase; margin-bottom:8px; }
.banner-title { color:#FFFFFF; font-size:28px; font-weight:950; letter-spacing:-.03em; line-height:1.15; margin-bottom:10px; }
.banner-body { color:#DBEAFE; font-size:14.5px; line-height:1.7; }
.qa-link { display:flex; align-items:center; gap:14px; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:18px; padding:15px 16px; margin-bottom:12px; text-decoration:none !important; box-shadow:0 8px 22px rgba(15,23,42,0.05); transition:border-color .15s, box-shadow .15s; }
.qa-link:hover { border-color:#BFDBFE; box-shadow:0 12px 28px rgba(37,99,235,0.12); }
.qa-icon { width:46px; height:46px; border-radius:14px; background:#EFF6FF; border:1px solid #BFDBFE; display:flex; align-items:center; justify-content:center; flex:none; }
.qa-text { display:flex; flex-direction:column; gap:2px; flex:1; min-width:0; }
.qa-title { color:#0F172A; font-size:14.5px; font-weight:900; line-height:1.3; }
.qa-sub { color:#64748B; font-size:12.5px; line-height:1.45; }
.qa-arrow { flex:none; }
.wf-row { display:flex; align-items:stretch; gap:6px; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:22px; box-shadow:0 10px 30px rgba(15,23,42,0.055); padding:22px 18px; flex-wrap:wrap; }
.wf-step { flex:1 1 112px; min-width:112px; text-align:center; padding:4px 4px; }
.wf-circle { width:54px; height:54px; border-radius:50%; background:#EFF6FF; border:1px solid #BFDBFE; display:flex; align-items:center; justify-content:center; margin:0 auto 10px; }
.wf-num { color:#0B5ED7; font-size:10.5px; font-weight:900; letter-spacing:.09em; }
.wf-title { color:#0F172A; font-size:13.5px; font-weight:900; margin:3px 0 6px; line-height:1.25; }
.wf-body { color:#64748B; font-size:11.8px; line-height:1.5; }
.wf-arrow { align-self:center; flex:none; }
@media (max-width:1250px) { .wf-arrow { display:none; } }
.result-card { padding:22px; margin-bottom:16px; }
.result-green { background:linear-gradient(135deg,#ECFDF5,#FFFFFF); border-color:#A7F3D0; }
.result-blue { background:linear-gradient(135deg,#EFF6FF,#FFFFFF); border-color:#BFDBFE; }
.result-purple { background:linear-gradient(135deg,#F5F3FF,#FFFFFF); border-color:#DDD6FE; }
.result-gold { background:linear-gradient(135deg,#FFFBEB,#FFFFFF); border-color:#FDE68A; }
.result-label { font-size:12px; font-weight:950; text-transform:uppercase; letter-spacing:.07em; color:#475569; margin-bottom:7px; }
.result-value { font-size:25px; font-weight:950; line-height:1.22; letter-spacing:-.035em; color:#061B3A; margin-bottom:8px; }
.result-text { color:#475569; font-size:14px; line-height:1.65; }
.report-photo-strip { position:relative; border-radius:18px; overflow:hidden; min-height:110px; margin:16px 0; display:flex; align-items:center; }
.report-photo-strip img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; object-position:center; }
.report-photo-strip .strip-overlay { position:absolute; inset:0; background:linear-gradient(90deg, rgba(6,27,58,0.88), rgba(6,27,58,0.35)); }
.report-photo-strip .strip-text { position:relative; z-index:2; color:white; padding:16px 20px; font-size:14px; font-weight:800; line-height:1.5; }
.info-box { background:#EFF6FF; border:1px solid #BFDBFE; color:#1E3A8A; padding:18px 20px; border-radius:18px; line-height:1.7; font-size:14px; margin:16px 0; }
.warning-box { background:#FFF7ED; border:1px solid #FDBA74; color:#7C2D12; padding:18px 20px; border-radius:18px; line-height:1.7; font-size:14px; margin:16px 0; }
.footer { margin-top:28px; padding:20px 8px 8px 8px; border-top:1px solid #E2E8F0; color:#64748B; font-size:13px; line-height:1.7; text-align:center; }
div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button { border-radius:14px !important; border:none !important; background:linear-gradient(135deg,#2563EB 0%,#0B5ED7 55%,#063A8C 100%) !important; color:white !important; font-weight:850 !important; box-shadow:0 10px 20px rgba(37,99,235,.22) !important; padding:.78rem 1rem !important; }
.stSelectbox label, .stSlider label, .stTextArea label { font-weight:800 !important; color:#334155 !important; }
@media (max-width:1100px) {
  .hero-grid { grid-template-columns:1fr; }
  .hero-illustration img { height:220px; }
}
@media (max-width:700px) {
  .hero-card { padding:26px 22px; }
  .hero-title { font-size:31px; }
  .top-header { align-items:flex-start; gap:12px; }
  .header-user { display:none; }
  .banner-title { font-size:22px; }
}

/* =====================================================
   GET RECOMMENDATION FORM VISIBILITY
   ===================================================== */

/* Titles and descriptions inside bordered form panels */
[data-testid="stMain"]
[data-testid="stVerticalBlockBorderWrapper"] h1,

[data-testid="stMain"]
[data-testid="stVerticalBlockBorderWrapper"] h2,

[data-testid="stMain"]
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    color: #0F172A !important;
    opacity: 1 !important;
}

[data-testid="stMain"] .stCaption p {
    color: #64748B !important;
    opacity: 1 !important;
}

/* Input labels */
[data-testid="stMain"] .stSelectbox label,
[data-testid="stMain"] .stSlider label,
[data-testid="stMain"] .stTextArea label,
[data-testid="stMain"] .stTextInput label {
    color: #334155 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* Select box container */
[data-testid="stMain"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    color: #0F172A !important;
    min-height: 48px !important;
}

/* Selected value and placeholder */
[data-testid="stMain"] div[data-baseweb="select"] span,
[data-testid="stMain"] div[data-baseweb="select"] input {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    opacity: 1 !important;
}

/* Dropdown arrow */
[data-testid="stMain"] div[data-baseweb="select"] svg {
    fill: #475569 !important;
    color: #475569 !important;
}

/* Focus state */
[data-testid="stMain"] div[data-baseweb="select"] > div:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12) !important;
}

/* Open dropdown menu */
div[data-baseweb="popover"] {
    background: #FFFFFF !important;
}

ul[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}

li[role="option"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}

li[role="option"] span {
    color: #0F172A !important;
}

li[role="option"]:hover {
    background-color: #EFF6FF !important;
    color: #1D4ED8 !important;
}

/* Slider and text-area values */
[data-testid="stMain"] textarea,
[data-testid="stMain"] input {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
                        
</style>
""", unsafe_allow_html=True)

# Sidebar navigation: real SVG icons, no radio circles, whole row clickable (spec: icon + label only)
NAV_ICON_FILES = [
    "home.svg",
    "recommendation.svg",
    "advisor-dashboard.svg",
    "methodology.svg",
    "data-governance.svg",
    "responsible-use.svg",
    "feedback.svg",
]

def _nav_icon_data_uri(filename, color):
    path = ICON_DIR / "navigation" / filename
    if not path.exists():
        return ""
    svg = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    svg = svg.replace("currentColor", color)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

_nav_rules = ["""
div[role="radiogroup"] > label { display:flex; align-items:center; gap:12px; padding:11px 14px; border-radius:9px; color:#D9E8FA; background:transparent; border:none; margin-bottom:6px; cursor:pointer; }
div[role="radiogroup"] > label:hover { background: rgba(255,255,255,0.08); }
div[role="radiogroup"] > label:has(input:checked) { background:#2367CF; color:#FFFFFF; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12); }
div[role="radiogroup"] > label > div:first-of-type { display:none; }
div[role="radiogroup"] > label p { color:inherit !important; font-weight:700; margin:0; font-size:14px; }
div[role="radiogroup"] > label::before { content:""; width:19px; height:19px; flex:none; background:center / contain no-repeat; }
"""]
for _idx, _fname in enumerate(NAV_ICON_FILES, start=1):
    _pale = _nav_icon_data_uri(_fname, "#D9E8FA")
    _white = _nav_icon_data_uri(_fname, "#FFFFFF")
    if _pale:
        _nav_rules.append(f'div[role="radiogroup"] > label:nth-child({_idx})::before {{ background-image:url("{_pale}"); }}')
    if _white:
        _nav_rules.append(f'div[role="radiogroup"] > label:nth-child({_idx}):has(input:checked)::before {{ background-image:url("{_white}"); }}')
render_html("<style>" + "\n".join(_nav_rules) + "</style>")

# Rwanda map watermark, fixed behind all page content
_wm_uri = watermark_data_uri()
if _wm_uri:
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            right: -10%;
            bottom: -16%;
            width: 58vw;
            height: 58vw;
            background: url("{_wm_uri}") no-repeat center / contain;
            opacity: 0.045;
            pointer-events: none;
            z-index: 0;
        }}
        .hero-card {{ position: relative; overflow: hidden; }}
        .hero-card::before {{
            content: "";
            position: absolute;
            top: -12%;
            right: 24%;
            width: 430px;
            height: 430px;
            background: url("{_wm_uri}") no-repeat center / contain;
            opacity: 0.07;
            pointer-events: none;
        }}
        .hero-grid {{ position: relative; z-index: 1; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# NAVIGATION STATE
# =========================================================
PAGES = [
    "Home",
    "Get Recommendation",
    "Advisor Dashboard",
    "Methodology",
    "Data & Governance",
    "Responsible Use",
    "Feedback",
]

# Quick-action cards navigate via ?nav=<page>; apply it before the radio is instantiated.
_qp_nav = st.query_params.get("nav")
if _qp_nav in PAGES:
    st.session_state["nav"] = _qp_nav
    st.query_params.clear()

# =========================================================
# SIDEBAR AND HEADER
# =========================================================
with st.sidebar:
    logo_light_uri = image_data_uri(PORTAL_LOGO_LIGHT)
    if logo_light_uri:
        brand_media = f'<img class="sidebar-logo" src="{logo_light_uri}" alt="Rwanda Academic Guidance Portal">'
        brand_title = ""
    else:
        mark_uri = image_data_uri(PORTAL_MARK_PNG) or image_data_uri(PORTAL_MARK_SVG)
        brand_media = f'<div class="sidebar-emblem"><img src="{mark_uri}" alt="Portal mark"></div>' if mark_uri else ""
        brand_title = '<div class="sidebar-title">Rwanda Academic<br>Guidance Portal</div>'

    render_html(
        f"""
        <div class="sidebar-brand">
            {brand_media}
            {brand_title}
            <div class="sidebar-subtitle">Academic pathway, program category, and bridge-course decision-support prototype.</div>
        </div>
        """
    )

    st.markdown('<div class="sidebar-section-title">Main Navigation</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "Main navigation",
        PAGES,
        key="nav",
        label_visibility="collapsed",
    )

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

header_mark_uri = image_data_uri(PORTAL_MARK_PNG) or image_data_uri(PORTAL_MARK_SVG)
header_mark_html = (
    f'<div class="header-mark"><img src="{header_mark_uri}" alt="Portal mark"></div>'
    if header_mark_uri
    else ""
)
render_html(f"""
<div class="top-header">
    <div class="header-left">
        {header_mark_html}
        <div>
            <div class="header-kicker">Rwanda Education Context</div>
            <div class="header-title">Academic Guidance Decision-Support Prototype</div>
        </div>
    </div>
    <div class="header-right">
        <div class="header-control">{icon("content/bell.svg", "#475569", 18)}</div>
        <div class="header-user">{icon("content/circle-user-round.svg", "#475569", 18)} Learner / Advisor · Advisory Use Only</div>
    </div>
</div>
""")

# =========================================================
# PAGES
# =========================================================
if selected_page == "Home":
    hero_photo_uri = image_data_uri(HERO_PHOTO)
    hero_media = (
        f'<img src="{hero_photo_uri}" alt="Graduate viewed from behind">'
        if hero_photo_uri
        else ""
    )

    render_html(
        f"""
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
                    <span class="badge badge-blue">Research Prototype</span>
                    <span class="badge badge-gold">Advisory Use Only</span>
                    <span class="badge badge-green">Explainable Recommendation</span>
                    <span class="badge badge-purple">Rwanda Education Context</span>
                </div>
                <div class="hero-illustration">{hero_media}</div>
            </div>
        </div>
        """
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Survey Responses", "105", "Collected from students and advisors", "metrics/survey-responses.svg", "blue")
    with c2:
        metric_card("Program Recommendation Requested", "53.3%", "56 out of 105 respondents", "metrics/program-guidance.svg", "green")
    with c3:
        metric_card("Career Guidance Requested", "55.2%", "58 out of 105 respondents", "metrics/career-guidance.svg", "gold")
    with c4:
        metric_card("Willing to Use Bridge Courses", "88.6%", "93 out of 105 respondents", "metrics/bridge-course.svg", "purple")
    with c5:
        metric_card("Willing to Test Prototype", "76.2%", "80 out of 105 respondents", "metrics/prototype-testing.svg", "teal")

    left_col, right_col = st.columns([2.2, 1], gap="medium")

    with left_col:
        render_html(
            '<div class="section-card" style="padding:20px 24px;margin-bottom:14px;">'
            '<div class="section-title" style="font-size:20px;margin-bottom:6px;">How It Works</div>'
            '<div class="section-body">The system follows a simple, explainable guidance workflow.</div></div>'
        )
        WORKFLOW_STEPS = [
            ("01", "Learner Profile", "Education type, pathway, strengths, interests, and career direction.", "workflow/learner-profile.svg"),
            ("02", "Data Processing", "The profile is structured into the model's input features.", "workflow/data-processing.svg"),
            ("03", "Eligibility Filter", "Strong TVET, subject, interest, and career signals are checked first.", "workflow/eligibility-filter.svg"),
            ("04", "ML Prediction", "The saved model predicts a program category where needed.", "workflow/ml-prediction.svg"),
            ("05", "Bridge Mapping", "The bridge course is mapped from the program category.", "workflow/bridge-mapping.svg"),
            ("06", "Advisor Review", "The recommendation is explained for advisor discussion.", "workflow/advisor-review.svg"),
        ]
        _wf_parts = []
        for _i, (_num, _title, _body, _ic) in enumerate(WORKFLOW_STEPS):
            if _i:
                _wf_parts.append(f'<div class="wf-arrow">{icon("actions/arrow-right.svg", "#94A3B8", 16)}</div>')
            _wf_parts.append(
                f'<div class="wf-step"><div class="wf-circle">{icon(_ic, "#1D4ED8", 24)}</div>'
                f'<div class="wf-num">STEP {_num}</div><div class="wf-title">{_title}</div>'
                f'<div class="wf-body">{_body}</div></div>'
            )
        render_html('<div class="wf-row">' + "".join(_wf_parts) + '</div>')

    with right_col:
        render_html(
            '<div class="section-card" style="padding:20px 24px;margin-bottom:14px;">'
            '<div class="section-title" style="font-size:20px;margin-bottom:6px;">Quick Actions</div>'
            '<div class="section-body">Jump to the most-used sections.</div></div>'
        )
        quick_action("Get Your Recommendation", "Start a new guidance session", "get-recommendation.svg", "Get Recommendation")
        quick_action("Advisor Dashboard", "Review prototype evaluation indicators", "open-dashboard.svg", "Advisor Dashboard")
        quick_action("How the System Works", "Read the hybrid methodology", "system-methodology.svg", "Methodology")
        quick_action("Responsible Use", "Advisory scope and limits of the prototype", "open-responsible-use.svg", "Responsible Use")

    st.markdown('<div class="info-box"><b>Advisory notice:</b> This portal is a decision-support prototype. Recommendations are for exploration and advisor discussion; final academic decisions must follow official MINEDUC, REB, RTB, and institutional admission requirements.</div>', unsafe_allow_html=True)

elif selected_page == "Get Recommendation":
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

                report_photo_uri = image_data_uri(HERO_PHOTO)
                if report_photo_uri:
                    render_html(
                        f"""
                        <div class="report-photo-strip">
                            <img src="{report_photo_uri}" alt="Graduate viewed from behind">
                            <div class="strip-overlay"></div>
                            <div class="strip-text">Your personalized guidance report is ready to download and discuss with an academic advisor.</div>
                        </div>
                        """
                    )

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

elif selected_page == "Advisor Dashboard":
    st.markdown('<div class="section-card"><div class="section-title">Advisor Dashboard</div><div class="section-body">This section summarizes prototype evaluation indicators for advisor review and stakeholder discussion.</div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Usefulness Rating", "4/5", "Prototype evaluation indicator", "content/star.svg", "gold")
    with c2: metric_card("Explanation Clarity", "4/5", "Advisor-facing transparency", "content/message-square-text.svg", "blue")
    with c3: metric_card("Usability Rating", "4/5", "Interface ease of use", "content/user-check.svg", "green")
    with c4: metric_card("Feedback Collection", "Active", "Stored in feedback_responses.csv", "content/clipboard-check.svg", "purple")
    st.markdown('<div class="info-box"><b>Advisor note:</b> This dashboard is for prototype monitoring and presentation. It does not represent official national analytics.</div>', unsafe_allow_html=True)

elif selected_page == "Methodology":
    banner(
        METHODOLOGY_PHOTO,
        "System Methodology",
        "How the System Works",
        "The system follows the hybrid architecture described in the research proposal: learner profile input, rule-based eligibility filtering, machine learning prediction, bridge-course mapping, alternative pathway suggestion, explainability, and feedback collection.",
    )
    c1, c2, c3 = st.columns(3)
    with c1: icon_card("Input Features", "Education Type, Pathway, Stream or TVET Trade, Strongest Subject, Weakest Subject, Interest Area, Average Score Range, Digital Skill Level, and Career Cluster.", "content/file-text.svg")
    with c2: icon_card("Prediction Layer", "The saved model predicts a recommended academic program category when a direct eligibility rule is not enough.", "workflow/ml-prediction.svg")
    with c3: icon_card("Bridge Mapping", "The bridge course is mapped from the recommended program category to keep the output consistent with the training notebook.", "workflow/bridge-mapping.svg")
    c4, c5 = st.columns(2)
    with c4: icon_card("Explainability", "The explanation summarizes how the learner profile connects to the recommendation.", "content/message-square-text.svg")
    with c5: icon_card("Responsible Use", "The output is advisory and should be reviewed with official entry requirements and academic advisors.", "content/shield-check.svg")

elif selected_page == "Data & Governance":
    st.markdown('<div class="section-card"><div class="section-title">Data & Governance</div><div class="section-body">The prototype is designed around data minimization, transparent outputs, and human review. It avoids unnecessary personal identifiers and keeps the recommendation explainable.</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: icon_card("Data Minimization", "The app does not require national ID, exact address, phone number, or sensitive personal identifiers.", "content/database.svg")
    with c2: icon_card("Human Review", "Recommendations should be discussed with advisors, institutions, parents, or guardians before decisions are made.", "content/user-check.svg")
    with c3: icon_card("Model Consistency", "The app loads the saved model artifact and uses the same program-to-bridge-course mapping used in training.", "content/refresh-cw.svg")
    c4, c5 = st.columns(2)
    with c4: icon_card("Privacy by Design", "Learner inputs stay within the session, aligned with Rwanda's Law No. 058/2021 on personal data protection.", "content/lock-keyhole.svg")
    with c5: icon_card("Advisory Boundary", "Outputs support exploration and counseling; they are not admission decisions.", "content/triangle-alert.svg", "#B45309")
    st.markdown('<div class="warning-box"><b>Important:</b> This is a capstone prototype and not an official government system.</div>', unsafe_allow_html=True)

elif selected_page == "Responsible Use":
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
    c1, c2, c3 = st.columns(3)
    with c1: icon_card("Advisory Only", "Recommendations support decision-making; they never replace official admission requirements.", "content/shield-check.svg")
    with c2: icon_card("Human Oversight", "Every recommendation is designed for review with advisors, parents, or guardians.", "content/user-check.svg")
    with c3: icon_card("Transparent Reasoning", "Each output includes an explanation of the profile factors behind it.", "content/info.svg")
    st.markdown('<div class="warning-box"><b>Important Disclaimer:</b> The recommendation is not an official MINEDUC, REB, RTB, or university admission decision. It does not guarantee admission into any program.</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>Ethical design:</b> The system avoids unnecessary personal identifiers and includes an explanation layer to support transparency, fairness, and human review.</div>', unsafe_allow_html=True)

elif selected_page == "Feedback":
    banner(
        FEEDBACK_PHOTO,
        "Prototype Evaluation",
        "System Feedback",
        "This section collects feedback on usefulness, explanation clarity, and usability for prototype evaluation.",
    )

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