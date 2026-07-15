import base64
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# LOCAL PROJECT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "academic_pathway_model.joblib"

STATIC_DIR = BASE_DIR / "static"
BRANDING_DIR = STATIC_DIR / "branding"
ICONS_DIR = STATIC_DIR / "icons"
PHOTOS_DIR = STATIC_DIR / "images"

PORTAL_LOGO = BRANDING_DIR / "portal_logo.png"
PORTAL_MARK = BRANDING_DIR / "portal_mark.svg"
RWANDA_MAP_WATERMARK = BRANDING_DIR / "rwanda_map_watermark.svg"

NAVIGATION_ICONS = {
    "Home": ICONS_DIR / "navigation" / "home.svg",
    "Get Recommendation": ICONS_DIR / "navigation" / "recommendation.svg",
    "Advisor Dashboard": ICONS_DIR / "navigation" / "advisor-dashboard.svg",
    "Methodology": ICONS_DIR / "navigation" / "methodology.svg",
    "Data & Governance": ICONS_DIR / "navigation" / "data-governance.svg",
    "Responsible Use": ICONS_DIR / "navigation" / "responsible-use.svg",
    "Feedback": ICONS_DIR / "navigation" / "feedback.svg",
}

METRIC_ICONS = {
    "survey": ICONS_DIR / "metrics" / "survey-responses.svg",
    "program": ICONS_DIR / "metrics" / "program-guidance.svg",
    "career": ICONS_DIR / "metrics" / "career-guidance.svg",
    "bridge": ICONS_DIR / "metrics" / "bridge-course.svg",
    "prototype": ICONS_DIR / "metrics" / "prototype-testing.svg",
}

WORKFLOW_ICONS = {
    "learner": ICONS_DIR / "workflow" / "learner-profile.svg",
    "processing": ICONS_DIR / "workflow" / "data-processing.svg",
    "eligibility": ICONS_DIR / "workflow" / "eligibility-filter.svg",
    "prediction": ICONS_DIR / "workflow" / "ml-prediction.svg",
    "bridge": ICONS_DIR / "workflow" / "bridge-mapping.svg",
    "advisor": ICONS_DIR / "workflow" / "advisor-review.svg",
}

ACTION_ICONS = {
    "recommendation": ICONS_DIR / "actions" / "get-recommendation.svg",
    "dashboard": ICONS_DIR / "actions" / "open-dashboard.svg",
    "methodology": ICONS_DIR / "actions" / "system-methodology.svg",
    "responsible": ICONS_DIR / "actions" / "open-responsible-use.svg",
    "arrow": ICONS_DIR / "actions" / "arrow-right.svg",
    "download": ICONS_DIR / "actions" / "download-report.svg",
}

CONTENT_ICONS = {
    "bell": ICONS_DIR / "content" / "bell.svg",
    "help": ICONS_DIR / "content" / "circle-help.svg",
    "user": ICONS_DIR / "content" / "circle-user-round.svg",
    "clipboard": ICONS_DIR / "content" / "clipboard-check.svg",
    "database": ICONS_DIR / "content" / "database.svg",
    "file": ICONS_DIR / "content" / "file-text.svg",
    "info": ICONS_DIR / "content" / "info.svg",
    "lock": ICONS_DIR / "content" / "lock-keyhole.svg",
    "menu": ICONS_DIR / "content" / "menu.svg",
    "message": ICONS_DIR / "content" / "message-square-text.svg",
    "refresh": ICONS_DIR / "content" / "refresh-cw.svg",
    "shield": ICONS_DIR / "content" / "shield-check.svg",
    "star": ICONS_DIR / "content" / "star.svg",
    "warning": ICONS_DIR / "content" / "triangle-alert.svg",
    "review": ICONS_DIR / "content" / "user-check.svg",
}

HERO_PHOTO = PHOTOS_DIR / "hero_graduate_rear_view.jpg"
LEARNER_PHOTO = PHOTOS_DIR / "learner_profile_faceless.jpg"
NOTETAKING_PHOTO = PHOTOS_DIR / "bridge_course_notes_faceless.jpg"
COLLABORATION_PHOTO = PHOTOS_DIR / "student_collaboration_faceless.jpg"
ADVISOR_PHOTO = PHOTOS_DIR / "advisor_review_faceless.jpg"
METHODOLOGY_PHOTO = PHOTOS_DIR / "methodology_teacher_rear_view.jpg"
FEEDBACK_PHOTO = PHOTOS_DIR / "feedback_writing_faceless.jpg"

# Supports the accidental double extension visible in the current folder.
ANALYTICS_PHOTO_CANDIDATES = [
    PHOTOS_DIR / "analytics_laptop_graphs.png",
]
ANALYTICS_PHOTO = next(
    (path for path in ANALYTICS_PHOTO_CANDIDATES if path.exists()),
    ANALYTICS_PHOTO_CANDIDATES[0],
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Rwanda Academic Guidance Portal",
    page_icon=str(PORTAL_LOGO) if PORTAL_LOGO.exists() else None,
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


def asset_data_uri(asset_path, color=None):
    """Return a browser-safe data URI for a local SVG or raster image.

    Raster MIME types are detected from the actual file bytes rather than only
    from the filename. This keeps downloaded photos working even when Windows
    saved them with a misleading extension such as ``.jpg`` for a PNG file.
    """
    path = Path(asset_path)
    if not path.exists() or not path.is_file():
        return ""

    raw = path.read_bytes()
    if not raw:
        return ""

    if path.suffix.lower() == ".svg" or raw.lstrip().startswith(b"<svg"):
        try:
            svg = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ""
        if color:
            svg = svg.replace("currentColor", color)
        encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{encoded}"

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif raw.startswith((b"GIF87a", b"GIF89a")):
        mime_type = "image/gif"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        mime_type = "image/webp"
    else:
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = mime_types.get(path.suffix.lower(), "application/octet-stream")

    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def asset_img(asset_path, alt="", css_class="asset-icon", color=None):
    uri = asset_data_uri(asset_path, color=color)
    if not uri:
        return '<span class="asset-fallback" aria-hidden="true"></span>'
    return f'<img class="{css_class}" src="{uri}" alt="{alt}">'


def page_intro(title, subtitle, photo_path=None, icon_path=None, eyebrow="Rwanda Academic Guidance Portal"):
    photo_uri = asset_data_uri(photo_path) if photo_path else ""
    icon = asset_img(icon_path, title, "page-intro-icon", "#1558C0") if icon_path else ""
    media = (
        f'<img class="page-intro-photo" src="{photo_uri}" alt="{title}">'
        if photo_uri
        else '<div class="page-intro-photo page-intro-placeholder"></div>'
    )
    st.markdown(
        f"""
        <section class="page-intro">
            <div class="page-intro-copy">
                <div class="page-intro-eyebrow">{eyebrow}</div>
                <div class="page-intro-heading-row">{icon}<h1>{title}</h1></div>
                <p>{subtitle}</p>
            </div>
            <div class="page-intro-media">{media}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note, icon_path, accent="#1558C0", icon_background="#EAF2FF"):
    icon = asset_img(icon_path, label, "metric-svg", accent)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon" style="background:{icon_background};">{icon}</div>
            <div class="metric-content">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_step(number, title, body, icon_path, accent="#1558C0", icon_background="#EAF2FF"):
    """Return compact HTML so Markdown never interprets the markup as a code block."""
    icon = asset_img(icon_path, title, "workflow-svg", accent)
    return (
        f'<div class="workflow-step">'
        f'<div class="workflow-icon" style="background:{icon_background};">{icon}</div>'
        f'<div class="workflow-number">STEP {int(number):02d}</div>'
        f'<div class="workflow-title">{title}</div>'
        f'<div class="workflow-body">{body}</div>'
        f'</div>'
    )


def internal_page_url(page_name):
    """Build a local query-string URL for the custom single-page navigation."""
    return f"?page={quote(page_name)}"


def quick_action_link(title, copy, icon_path, accent, background, destination):
    """Render one clickable quick-action card without a separate button."""
    icon = asset_img(icon_path, title, "quick-action-svg", accent)
    arrow = asset_img(ACTION_ICONS["arrow"], "Open", "quick-action-arrow", "#64748B")
    href = internal_page_url(destination)
    st.markdown(
        f'<a class="quick-action-link" href="{href}" target="_self">'
        f'<span class="quick-action-icon" style="background:{background};">{icon}</span>'
        f'<span class="quick-action-copy"><strong>{title}</strong><span>{copy}</span></span>'
        f'<span class="quick-action-open">{arrow}</span>'
        f'</a>',
        unsafe_allow_html=True,
    )


def info_card(title, body, icon_path, accent="#1558C0", background="#EAF2FF", class_name="info-card"):
    icon = asset_img(icon_path, title, "info-card-svg", accent)
    st.markdown(
        f"""
        <div class="{class_name}">
            <div class="info-card-icon" style="background:{background};">{icon}</div>
            <div>
                <div class="info-card-title">{title}</div>
                <div class="info-card-body">{body}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(label, value, text, color="green"):
    st.markdown(
        f"""
        <div class="result-card result-{color}">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
            <div class="result-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def progress_card(label, value, note, color_class="blue"):
    bounded_value = max(0, min(100, int(value)))
    st.markdown(
        f"""
        <div class="progress-card">
            <div class="progress-top"><span>{label}</span><strong>{bounded_value}%</strong></div>
            <div class="progress-track"><div class="progress-fill progress-{color_class}" style="width:{bounded_value}%"></div></div>
            <div class="progress-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigate_to(page_name):
    st.session_state.selected_page = page_name


# =========================================================
# CUSTOM CSS
# =========================================================
portal_mark_uri = asset_data_uri(PORTAL_MARK, color="#0B2A4A")
map_uri = asset_data_uri(RWANDA_MAP_WATERMARK, color="#00A1DE")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {{
    --navy:#061B3A;
    --navy-2:#082D5F;
    --blue:#1558C0;
    --rwanda-blue:#00A1DE;
    --green:#10935D;
    --gold:#D99100;
    --purple:#7047EB;
    --teal:#0E9F9A;
    --canvas:#F5F8FC;
    --line:#E3EAF3;
    --muted:#64748B;
}}

html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
.stApp {{ background:var(--canvas); }}
.block-container {{ max-width:1480px; padding:1.05rem 1.55rem 2.4rem; }}
[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}

/* Hide Streamlit's automatic multipage navigation.
   The portal uses the custom institutional navigation below. */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"] {{
    display:none !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top:.55rem !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,#031A3A 0%,#062D62 55%,#031A3A 100%);
    border-right:1px solid rgba(255,255,255,.08);
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top:.6rem; }}
[data-testid="stSidebar"] * {{ color:#FFFFFF; }}
.sidebar-brand {{ text-align:center; padding:8px 4px 24px; border-bottom:1px solid rgba(255,255,255,.13); }}
.sidebar-mark {{ width:76px; height:76px; margin:0 auto 12px; border-radius:50%; padding:11px; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16); display:flex; align-items:center; justify-content:center; }}
.sidebar-mark img {{ width:100%; height:100%; object-fit:contain; }}
.sidebar-title {{ font-size:21px; line-height:1.27; font-weight:850; letter-spacing:-.025em; }}
.sidebar-subtitle {{ color:#C8DBF5 !important; margin-top:7px; font-size:12px; line-height:1.5; }}
.sidebar-section-title {{ color:#9CBCE5 !important; font-size:10px; font-weight:850; letter-spacing:.11em; text-transform:uppercase; margin:19px 0 9px; }}
.sidebar-box {{ border-top:1px solid rgba(255,255,255,.12); padding:14px 2px 2px; font-size:11px; line-height:1.8; color:#D5E4F7 !important; }}
.sidebar-info-row {{ display:flex; justify-content:space-between; gap:10px; }}
.sidebar-info-row span:first-child {{ color:#AFC7E5 !important; }}
.sidebar-context {{ margin-top:23px; padding-top:17px; border-top:1px solid rgba(255,255,255,.12); }}
.context-flag {{ width:35px; height:22px; border-radius:3px; overflow:hidden; display:grid; grid-template-rows:2fr 1fr 1fr; margin-bottom:8px; box-shadow:0 2px 6px rgba(0,0,0,.18); }}
.context-flag span:nth-child(1) {{ background:#00A1DE; }}
.context-flag span:nth-child(2) {{ background:#FAD201; }}
.context-flag span:nth-child(3) {{ background:#20603D; }}
.context-title {{ font-size:11px; font-weight:800; }}
.context-copy {{ color:#BFD1EA !important; font-size:10.5px; line-height:1.55; margin-top:6px; }}

.sidebar-nav {{ display:flex; flex-direction:column; gap:5px; }}
.sidebar-nav-item {{
    display:flex;
    align-items:center;
    gap:12px;
    min-height:40px;
    padding:9px 12px;
    border:1px solid transparent;
    border-radius:8px;
    text-decoration:none !important;
    background:transparent;
    transition:background .16s ease,border-color .16s ease,transform .16s ease;
}}
.sidebar-nav-item:hover {{
    background:rgba(255,255,255,.075);
    transform:translateX(1px);
}}
.sidebar-nav-item.active {{
    background:linear-gradient(90deg,rgba(38,112,229,.92),rgba(33,91,190,.82));
    border-color:rgba(255,255,255,.14);
    box-shadow:0 5px 14px rgba(0,0,0,.18);
}}
.sidebar-nav-icon {{ width:18px; height:18px; flex:0 0 18px; }}
.sidebar-nav-label {{ color:#FFFFFF !important; font-size:12px; font-weight:650; line-height:1.2; }}


/* Header */
.portal-header {{
    min-height:58px;
    background:#FFFFFF;
    border:1px solid var(--line);
    border-radius:12px;
    padding:10px 16px;
    margin-bottom:14px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    box-shadow:0 4px 18px rgba(15,23,42,.045);
}}
.header-brand {{ display:flex; align-items:center; gap:12px; }}
.header-menu {{ width:19px; opacity:.7; }}
.header-mark {{ width:38px; height:38px; object-fit:contain; }}
.header-copy strong {{ display:block; color:var(--navy); font-size:12px; }}
.header-copy span {{ color:var(--blue); font-size:11px; font-weight:700; }}
.header-actions {{ display:flex; align-items:center; gap:11px; }}
.header-action-icon {{ width:18px; height:18px; opacity:.72; }}
.header-user {{ display:flex; align-items:center; gap:9px; padding-left:12px; border-left:1px solid var(--line); }}
.header-avatar {{ width:32px; height:32px; border-radius:50%; background:#EAF1F8; display:flex; align-items:center; justify-content:center; }}
.header-avatar img {{ width:18px; height:18px; }}
.header-user-copy strong {{ display:block; color:#1E293B; font-size:10.5px; }}
.header-user-copy span {{ color:#64748B; font-size:9.5px; }}

/* Shared cards */
.page-intro {{
    min-height:154px;
    background:linear-gradient(115deg,#FFFFFF 0%,#F6FAFF 58%,#EFF7FF 100%);
    border:1px solid #DDE8F5;
    border-radius:14px;
    padding:24px 28px;
    margin-bottom:18px;
    display:grid;
    grid-template-columns:minmax(0,1fr) 250px;
    gap:24px;
    align-items:center;
    overflow:hidden;
    box-shadow:0 7px 24px rgba(15,23,42,.05);
}}
.page-intro-eyebrow {{ color:var(--blue); font-size:10px; font-weight:850; text-transform:uppercase; letter-spacing:.1em; margin-bottom:7px; }}
.page-intro-heading-row {{ display:flex; align-items:center; gap:10px; }}
.page-intro-heading-row h1 {{ margin:0; color:var(--navy); font-size:27px; line-height:1.15; letter-spacing:-.035em; }}
.page-intro-copy p {{ margin:10px 0 0; color:#5C6B7E; font-size:13px; line-height:1.65; max-width:820px; }}
.page-intro-icon {{ width:30px; height:30px; }}
.page-intro-media {{ height:116px; border-radius:12px; overflow:hidden; position:relative; background:#EAF2FA; }}
.page-intro-photo {{ width:100%; height:100%; object-fit:cover; object-position:center 34%; display:block; }}
.page-intro-placeholder {{ background:linear-gradient(135deg,#D9EAF9,#F4F8FC); }}

.hero-card {{
    position:relative;
    min-height:330px;
    display:grid;
    grid-template-columns:minmax(0,1.72fr) minmax(320px,.78fr);
    gap:42px;
    align-items:center;
    padding:38px 40px;
    background:linear-gradient(115deg,#FFFFFF 0%,#F5F9FF 62%,#EAF3FC 100%);
    border:1px solid #D7E5F5;
    border-radius:20px;
    box-shadow:0 12px 34px rgba(15,23,42,.07);
    overflow:hidden;
}}
.hero-watermark {{
    position:absolute;
    right:25%;
    top:-24%;
    width:520px;
    height:520px;
    opacity:.065;
    z-index:0;
    pointer-events:none;
}}
.hero-copy {{ position:relative; z-index:2; }}
.hero-kicker {{
    color:#0D5ED7;
    font-size:12px;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.11em;
    margin-bottom:17px;
}}
.hero-title {{
    max-width:900px;
    color:var(--navy);
    font-size:42px;
    font-weight:900;
    line-height:1.12;
    letter-spacing:-.045em;
}}
.hero-subtitle {{
    color:#1D5BD1;
    font-size:18px;
    font-weight:750;
    line-height:1.45;
    margin-top:18px;
}}
.hero-body {{
    max-width:960px;
    color:#52627A;
    font-size:15.5px;
    line-height:1.9;
    margin-top:18px;
}}
.hero-badges {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:18px; }}
.badge {{
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:9px 14px;
    border-radius:999px;
    font-size:11px;
    font-weight:750;
    border:1px solid transparent;
}}
.badge img {{ width:14px; height:14px; }}
.badge-blue {{ background:#EDF4FF; color:#2759BE; border-color:#C9DCF9; }}
.badge-gold {{ background:#FFF9EA; color:#A65D00; border-color:#F1D58E; }}
.badge-green {{ background:#ECFAF2; color:#15784A; border-color:#C7EAD5; }}
.hero-media {{
    position:relative;
    z-index:2;
    height:280px;
    border-radius:24px;
    overflow:hidden;
    background:#EAF2FA;
    box-shadow:0 18px 38px rgba(15,23,42,.13);
}}
.hero-media::before {{
    content:"";
    position:absolute;
    inset:0;
    z-index:1;
    background:linear-gradient(90deg,rgba(255,255,255,.04),rgba(3,24,52,.06));
    pointer-events:none;
}}
.hero-media img.hero-photo {{
    width:100%;
    height:100%;
    object-fit:cover;
    object-position:center 30%;
    filter:grayscale(100%) contrast(.94);
    display:block;
    position:relative;
    z-index:0;
}}

.metric-card, .panel, .info-card, .result-card, .progress-card {{
    background:#FFFFFF;
    border:1px solid var(--line);
    border-radius:12px;
    box-shadow:0 5px 17px rgba(15,23,42,.045);
}}
.metric-card {{ min-height:109px; padding:15px; display:flex; gap:12px; align-items:flex-start; }}
.metric-icon {{ min-width:43px; width:43px; height:43px; border-radius:50%; display:flex; align-items:center; justify-content:center; }}
.metric-svg {{ width:23px; height:23px; }}
.metric-label {{ color:#4B5565; font-size:10px; font-weight:750; line-height:1.35; }}
.metric-value {{ color:var(--navy); font-size:23px; font-weight:850; letter-spacing:-.035em; line-height:1.1; margin-top:4px; }}
.metric-note {{ color:#7B8796; font-size:9px; line-height:1.45; margin-top:6px; }}

.panel {{ padding:17px; }}
.panel-title {{ color:var(--navy); font-size:13px; font-weight:850; margin-bottom:4px; }}
.panel-subtitle {{ color:#7A8795; font-size:9.5px; line-height:1.5; margin-bottom:12px; }}
.workflow-grid {{
    display:grid;
    grid-template-columns:repeat(5,minmax(0,1fr));
    column-gap:20px;
    row-gap:18px;
    align-items:start;
    padding:2px 4px 0;
}}
.workflow-step {{
    position:relative;
    text-align:center;
    padding:4px 6px 0;
}}
.workflow-step:nth-child(-n+4)::after {{
    content:"→";
    position:absolute;
    right:-16px;
    top:31px;
    color:#97A9BF;
    font-size:22px;
    font-weight:400;
}}
.workflow-step:nth-child(5)::after {{
    content:"";
}}
.workflow-step:nth-child(6) {{
    grid-column:3;
    margin-top:3px;
}}
.workflow-icon {{
    width:62px;
    height:62px;
    margin:0 auto 10px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    border:1px solid #BBD7FF;
    box-shadow:0 2px 8px rgba(21,88,192,.04);
}}
.workflow-svg {{ width:29px; height:29px; }}
.workflow-number {{
    color:#0D5ED7;
    font-size:10px;
    font-weight:900;
    letter-spacing:.12em;
    text-transform:uppercase;
}}
.workflow-title {{
    color:#152137;
    font-size:14px;
    font-weight:850;
    margin-top:8px;
    line-height:1.35;
}}
.workflow-body {{
    max-width:165px;
    margin:8px auto 0;
    color:#6A7B92;
    font-size:12px;
    line-height:1.55;
}}

.quick-action-link {{ display:flex; gap:10px; align-items:center; margin-bottom:9px; padding:11px 12px; border:1px solid var(--line); border-radius:9px; background:#FFFFFF; text-decoration:none !important; box-shadow:0 3px 12px rgba(15,23,42,.035); transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease; }}
.quick-action-link:hover {{ transform:translateY(-1px); border-color:#BFD4F3; box-shadow:0 7px 18px rgba(15,23,42,.08); }}
.quick-action-icon {{ width:31px; height:31px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:0 0 31px; }}
.quick-action-svg {{ width:17px; height:17px; }}
.quick-action-copy {{ flex:1; min-width:0; }}
.quick-action-copy strong {{ display:block; color:#1F2C3D; font-size:10px; }}
.quick-action-copy span {{ display:block; color:#7B8795; font-size:8.5px; margin-top:2px; }}
.quick-action-open {{ margin-left:auto; display:flex; align-items:center; justify-content:center; }}
.quick-action-arrow {{ width:15px; height:15px; }}

.info-card {{ padding:15px; display:flex; gap:12px; min-height:113px; }}
.info-card-icon {{ min-width:40px; width:40px; height:40px; border-radius:9px; display:flex; align-items:center; justify-content:center; }}
.info-card-svg {{ width:21px; height:21px; }}
.info-card-title {{ color:#18263A; font-size:11px; font-weight:800; margin-top:1px; }}
.info-card-body {{ color:#6B7788; font-size:9.5px; line-height:1.6; margin-top:5px; }}

.result-card {{ padding:17px; margin-bottom:11px; }}
.result-green {{ background:linear-gradient(135deg,#F0FBF4,#FFFFFF); border-color:#CAEED7; }}
.result-blue {{ background:linear-gradient(135deg,#F1F6FF,#FFFFFF); border-color:#D4E2FA; }}
.result-purple {{ background:linear-gradient(135deg,#F6F2FF,#FFFFFF); border-color:#E4D9FB; }}
.result-gold {{ background:linear-gradient(135deg,#FFF9EB,#FFFFFF); border-color:#F5E4B7; }}
.result-label {{ font-size:9px; color:#667385; font-weight:850; text-transform:uppercase; letter-spacing:.07em; }}
.result-value {{ color:var(--navy); font-size:17px; line-height:1.3; font-weight:850; margin-top:5px; }}
.result-text {{ color:#657285; font-size:9.5px; line-height:1.55; margin-top:6px; }}

.progress-card {{ padding:13px 15px; margin-bottom:10px; }}
.progress-top {{ display:flex; justify-content:space-between; gap:12px; color:#344256; font-size:10px; font-weight:700; }}
.progress-track {{ height:6px; background:#EDF1F6; border-radius:999px; overflow:hidden; margin-top:8px; }}
.progress-fill {{ height:100%; border-radius:999px; }}
.progress-blue {{ background:#2367D8; }}
.progress-green {{ background:#13A361; }}
.progress-purple {{ background:#7547E8; }}
.progress-gold {{ background:#E79A12; }}
.progress-note {{ color:#8290A0; font-size:8.5px; margin-top:6px; }}

.notice {{ border-radius:9px; padding:11px 13px; font-size:9.5px; line-height:1.55; margin-top:11px; }}
.notice-blue {{ background:#EEF4FF; color:#2250A3; border:1px solid #D3E1FB; }}
.notice-gold {{ background:#FFF8E9; color:#895A08; border:1px solid #F2E1B6; }}
.notice-green {{ background:#EFFAF3; color:#17683F; border:1px solid #CFEBD9; }}
.notice-red {{ background:#FFF2F0; color:#9D392D; border:1px solid #F4D0CB; }}
.section-heading {{ color:var(--navy); font-size:14px; font-weight:850; margin:16px 0 10px; }}

/* Streamlit controls */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:var(--line) !important; border-radius:12px !important; background:#FFFFFF; box-shadow:0 5px 17px rgba(15,23,42,.04); }}
.stSelectbox label, .stSlider label, .stTextArea label {{ color:#344256 !important; font-size:10px !important; font-weight:750 !important; }}
div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button {{
    border-radius:8px !important;
    border:0 !important;
    min-height:38px;
    background:linear-gradient(135deg,#1F68DA,#124FAE) !important;
    color:#FFFFFF !important;
    font-size:10px !important;
    font-weight:750 !important;
    box-shadow:0 6px 14px rgba(30,91,190,.18) !important;
}}
div.stButton > button:hover, div.stFormSubmitButton > button:hover, div.stDownloadButton > button:hover {{ transform:translateY(-1px); }}

.footer {{ margin-top:18px; padding:13px 2px 4px; border-top:1px solid var(--line); display:flex; justify-content:space-between; gap:20px; color:#768294; font-size:8.5px; line-height:1.55; }}
.footer strong {{ color:#4A586A; }}

@media(max-width:1100px) {{
    .hero-card {{ grid-template-columns:1fr; min-height:auto; }}
    .hero-media {{ height:245px; }}
    .hero-watermark {{ right:-5%; top:2%; width:420px; height:420px; }}
    .page-intro {{ grid-template-columns:1fr; }}
    .page-intro-media {{ height:150px; }}
    .workflow-grid {{ grid-template-columns:repeat(3,1fr); row-gap:28px; }}
    .workflow-step:nth-child(3)::after,
    .workflow-step:nth-child(5)::after {{ display:none; }}
    .workflow-step:nth-child(6) {{ grid-column:2; }}
}}
@media(max-width:720px) {{
    .block-container {{ padding:.8rem .75rem 1.5rem; }}
    .portal-header {{ align-items:flex-start; }}
    .header-actions {{ display:none; }}
    .hero-card {{ padding:26px 22px; gap:24px; }}
    .hero-title {{ font-size:31px; }}
    .hero-subtitle {{ font-size:16px; }}
    .hero-body {{ font-size:14px; line-height:1.75; }}
    .hero-media {{ height:220px; }}
    .workflow-grid {{ grid-template-columns:repeat(2,1fr); row-gap:26px; }}
    .workflow-step:nth-child(even)::after {{ display:none; }}
    .workflow-step:nth-child(6) {{ grid-column:1 / -1; width:50%; justify-self:center; }}
    .footer {{ display:block; }}
}}
</style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR AND HEADER
# =========================================================
valid_pages = list(NAVIGATION_ICONS.keys())
query_page = st.query_params.get("page", "Home")
if isinstance(query_page, list):
    query_page = query_page[0] if query_page else "Home"
selected_page = query_page if query_page in valid_pages else "Home"

with st.sidebar:
    nav_items = []
    for page_name, icon_path in NAVIGATION_ICONS.items():
        active_class = " active" if page_name == selected_page else ""
        icon = asset_img(icon_path, page_name, "sidebar-nav-icon", "#DCEAFF")
        nav_items.append(
            f'<a class="sidebar-nav-item{active_class}" href="{internal_page_url(page_name)}" target="_self">'
            f'{icon}<span class="sidebar-nav-label">{page_name}</span></a>'
        )

    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-mark">{asset_img(PORTAL_MARK, 'Portal mark', 'sidebar-logo', '#FFFFFF')}</div>
            <div class="sidebar-title">Rwanda Academic<br>Guidance Portal</div>
            <div class="sidebar-subtitle">Decision-Support System</div>
        </div>
        <div class="sidebar-section-title">Main Navigation</div>
        <nav class="sidebar-nav">{''.join(nav_items)}</nav>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-section-title">System Information</div>
        <div class="sidebar-box">
            <div class="sidebar-info-row"><span>Model Version</span><strong>v1.0.0</strong></div>
            <div class="sidebar-info-row"><span>Dataset Version</span><strong>2026.06</strong></div>
            <div class="sidebar-info-row"><span>Model Type</span><strong>{ARTIFACT['model_name']}</strong></div>
            <div class="sidebar-info-row"><span>Use</span><strong>Advisory</strong></div>
        </div>
        <div class="sidebar-context">
            <div class="context-flag"><span></span><span></span><span></span></div>
            <div class="context-title">Rwanda Education Context</div>
            <div class="context-copy">Aligned with academic guidance, TVET progression, bridge-course readiness, and responsible AI principles.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

header_menu = asset_img(CONTENT_ICONS["menu"], "Menu", "header-menu", "#334155")
header_logo = asset_img(PORTAL_MARK, "Portal mark", "header-mark", "#0B2A4A")
header_bell = asset_img(CONTENT_ICONS["bell"], "Notifications", "header-action-icon", "#334155")
header_help = asset_img(CONTENT_ICONS["help"], "Help", "header-action-icon", "#334155")
header_user = asset_img(CONTENT_ICONS["user"], "Portal user", "header-action-icon", "#475569")

st.markdown(
    f"""
    <div class="portal-header">
        <div class="header-brand">
            {header_menu}
            {header_logo}
            <div class="header-copy"><strong>Rwanda Academic Guidance Portal</strong><span>Academic Decision-Support Prototype</span></div>
        </div>
        <div class="header-actions">
            {header_bell}{header_help}
            <div class="header-user"><div class="header-avatar">{header_user}</div><div class="header-user-copy"><strong>Portal User</strong><span>Learner / Advisor</span></div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# PAGES
# =========================================================
if selected_page == "Home":
    hero_photo_uri = asset_data_uri(HERO_PHOTO)
    hero_photo = f'<img class="hero-photo" src="{hero_photo_uri}" alt="Graduate viewed from behind">' if hero_photo_uri else ""
    map_image = f'<img class="hero-watermark" src="{map_uri}" alt="" aria-hidden="true">' if map_uri else ""
    prototype_icon = asset_img(CONTENT_ICONS["clipboard"], "Prototype", "badge-icon", "#2759BE")
    advisory_icon = asset_img(CONTENT_ICONS["warning"], "Advisory", "badge-icon", "#A66600")
    explainable_icon = asset_img(CONTENT_ICONS["shield"], "Explainable", "badge-icon", "#15784A")

    st.markdown(
        f"""
        <section class="hero-card">
            {map_image}
            <div class="hero-copy">
                <div class="hero-kicker">Decision-Support Prototype • Explainable Guidance • Rwanda Context</div>
                <div class="hero-title">Welcome to the Rwanda Academic Guidance Portal</div>
                <div class="hero-subtitle">Academic Pathway, Program Category, and Bridge Course Recommendation System</div>
                <div class="hero-body">A professional decision-support prototype that helps Rwandan learners explore suitable academic program categories, preparatory bridge courses, and alternative pathways based on academic background, interests, digital skills, and career goals.</div>
                <div class="hero-badges">
                    <span class="badge badge-blue">{prototype_icon} Research Prototype</span>
                    <span class="badge badge-gold">{advisory_icon} Advisory Use Only</span>
                    <span class="badge badge-green">{explainable_icon} Explainable Recommendation</span>
                </div>
            </div>
            <div class="hero-media">{hero_photo}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Survey Responses", "105", "Collected from students and advisors", METRIC_ICONS["survey"], "#2465D4", "#EDF3FF")
    with c2:
        metric_card("Program Recommendation Requested", "53.3%", "56 out of 105 respondents", METRIC_ICONS["program"], "#10935D", "#ECF9F1")
    with c3:
        metric_card("Career Guidance Requested", "55.2%", "58 out of 105 respondents", METRIC_ICONS["career"], "#D99100", "#FFF7E6")
    with c4:
        metric_card("Willing to Use Bridge Courses", "88.6%", "93 out of 105 respondents", METRIC_ICONS["bridge"], "#7047EB", "#F3EFFF")
    with c5:
        metric_card("Willing to Test Prototype", "76.2%", "80 out of 105 respondents", METRIC_ICONS["prototype"], "#0E9F9A", "#EAF9F8")

    left, right = st.columns([3.15, 1.15])
    with left:
        steps = [
            workflow_step("1", "Learner Profile", "Education type, pathway, strengths, interests, and career direction.", WORKFLOW_ICONS["learner"], "#2465D4", "#EDF3FF"),
            workflow_step("2", "Data Processing", "The profile is structured into the model input features.", WORKFLOW_ICONS["processing"], "#0F172A", "#F4F6F8"),
            workflow_step("3", "Eligibility Filter", "Strong TVET, subject, interest, and career signals are checked first.", WORKFLOW_ICONS["eligibility"], "#2465D4", "#EDF3FF"),
            workflow_step("4", "ML Prediction", "The saved model predicts a program category where needed.", WORKFLOW_ICONS["prediction"], "#10935D", "#ECF9F1"),
            workflow_step("5", "Bridge Mapping", "The bridge course is mapped from the recommended program category.", WORKFLOW_ICONS["bridge"], "#7047EB", "#F3EFFF"),
            workflow_step("6", "Advisor Review", "The recommendation is explained for advisor discussion.", WORKFLOW_ICONS["advisor"], "#0E9F9A", "#EAF9F8"),
        ]
        st.markdown(
            f'<div class="panel"><div class="panel-title">How It Works</div><div class="panel-subtitle">The system follows a simple, explainable guidance workflow.</div><div class="workflow-grid">{"".join(steps)}</div><div class="notice notice-blue"><strong>Important:</strong> Recommendations must be confirmed against official admission requirements from relevant institutions.</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="panel-title" style="margin:4px 0 11px;">Quick Actions</div>', unsafe_allow_html=True)
        quick_actions = [
            ("Get Your Recommendation", "Start a new guidance session", ACTION_ICONS["recommendation"], "#2465D4", "#EDF3FF", "Get Recommendation", "qa_recommendation"),
            ("Advisor Dashboard", "View evaluation indicators", ACTION_ICONS["dashboard"], "#10935D", "#ECF9F1", "Advisor Dashboard", "qa_dashboard"),
            ("How the System Works", "Review the methodology", ACTION_ICONS["methodology"], "#7047EB", "#F3EFFF", "Methodology", "qa_methodology"),
            ("Responsible Use", "Read important guidance", ACTION_ICONS["responsible"], "#D99100", "#FFF7E6", "Responsible Use", "qa_responsible"),
        ]
        for title, copy, icon_path, accent, background, destination, _key in quick_actions:
            quick_action_link(title, copy, icon_path, accent, background, destination)

elif selected_page == "Get Recommendation":
    page_intro(
        "Get Your Recommendation",
        "Provide the learner's academic background, strengths, interests, and career direction. The system will generate an explainable program-category recommendation and aligned bridge-course guidance.",
        HERO_PHOTO,
        NAVIGATION_ICONS["Get Recommendation"],
    )

    left, right = st.columns([1, 1.22])
    with left:
        with st.container(border=True):
            st.markdown("#### 1. Learner Academic Profile")
            st.caption("Complete the profile carefully. Each field contributes to the recommendation.")

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
                interest_options = [default_interest] + [item for item in RWANDA_INTEREST_AREAS if item != default_interest]
                interest_area = st.selectbox("Interest Area", interest_options, key="tvet_interest_area")

                default_career = TVET_TRADE_TO_CAREER_CLUSTER.get(stream_or_trade, "Business Administration and Management")
                career_options = [default_career] + [item for item in PROGRAM_CATEGORY_OPTIONS if item != default_career]
                career_cluster = st.selectbox("Career Cluster", career_options, key="tvet_career_cluster")

            average_score_range = st.selectbox("Average Score Range", ["50–59%", "60–69%", "70–79%", "80–89%", "90–100%"], key="score_range")
            digital_skill_level = st.selectbox("Digital Skill Level", ["Beginner", "Intermediate", "Advanced"], key="digital_skill")
            submitted = st.button("Generate Academic Guidance Report", key="submit_recommendation", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("#### Your Guidance Report")
            st.caption("The output combines the trained model with transparent bridge-course and alternative-pathway guidance.")

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
                except Exception as error:
                    st.error("The learner profile could not be processed. Confirm that the model artifact matches the dashboard input features.")
                    st.exception(error)
                    st.stop()

                explanation = build_explanation(profile, recommended_program, recommended_bridge_course, alternative_pathway, source)
                result_card("Recommended Program Category", recommended_program, "The most suitable academic direction based on the submitted learner profile.", "green")
                result_card("Recommended Bridge Course", recommended_bridge_course, "Preparatory learning intended to strengthen readiness for the recommended category.", "blue")
                result_card("Alternative Academic Pathway", alternative_pathway, "A second route to discuss with an academic advisor.", "purple")
                result_card("Recommendation Source", source, "The recommendation shows whether the eligibility layer or saved ML model produced the category.", "gold")

                st.markdown('<div class="section-heading">Why This Recommendation?</div>', unsafe_allow_html=True)
                st.markdown(explanation)

                report = make_guidance_report(profile, recommended_program, recommended_bridge_course, alternative_pathway, source, explanation)
                st.download_button("Download Guidance Report", data=report, file_name="rwanda_academic_guidance_report.md", mime="text/markdown", use_container_width=True)
            else:
                result_card("Recommended Program Category", "Awaiting learner profile", "Complete the profile and generate the report to view the recommendation.", "green")
                result_card("Recommended Bridge Course", "Awaiting recommendation", "The bridge course will appear after the profile is processed.", "blue")
                result_card("Explainability Summary", "Awaiting profile", "The explanation will summarize the factors supporting the recommendation.", "purple")

elif selected_page == "Advisor Dashboard":
    page_intro(
        "Advisor Dashboard",
        "Review prototype indicators, feedback quality, and usability measures intended to support advisor and stakeholder discussion.",
        ANALYTICS_PHOTO,
        NAVIGATION_ICONS["Advisor Dashboard"],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Feedback", "105", "Prototype responses", METRIC_ICONS["survey"], "#2465D4", "#EDF3FF")
    with c2:
        metric_card("Average Usefulness", "4.3 / 5", "Rating", CONTENT_ICONS["star"], "#10935D", "#ECF9F1")
    with c3:
        metric_card("Average Clarity", "4.4 / 5", "Rating", CONTENT_ICONS["info"], "#0E9F9A", "#EAF9F8")
    with c4:
        metric_card("Average Usability", "4.2 / 5", "Rating", CONTENT_ICONS["user"], "#7047EB", "#F3EFFF")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Evaluation Overview</div><div class="panel-subtitle">Prototype experience indicators</div>', unsafe_allow_html=True)
        progress_card("Recommendation usefulness", 86, "Equivalent to an average rating of 4.3 out of 5.", "blue")
        progress_card("Explanation clarity", 88, "Equivalent to an average rating of 4.4 out of 5.", "green")
        progress_card("Interface usability", 84, "Equivalent to an average rating of 4.2 out of 5.", "purple")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Most Common Learner Needs</div><div class="panel-subtitle">Primary survey indicators</div>', unsafe_allow_html=True)
        progress_card("Career guidance", 55, "58 of 105 respondents", "blue")
        progress_card("Program recommendation", 53, "56 of 105 respondents", "green")
        progress_card("Bridge-course willingness", 89, "93 of 105 respondents", "purple")
        progress_card("Prototype testing", 76, "80 of 105 respondents", "gold")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="notice notice-blue"><strong>Advisor note:</strong> These indicators support prototype evaluation and stakeholder discussion. They are not official national statistics.</div>', unsafe_allow_html=True)

elif selected_page == "Methodology":
    page_intro(
        "System Methodology",
        "Understand how learner information moves through eligibility checks, machine-learning prediction, bridge-course mapping, explainability, and human review.",
        METHODOLOGY_PHOTO,
        NAVIGATION_ICONS["Methodology"],
    )

    steps = [
        workflow_step("1", "Learner Profile", "The learner submits academic and career information.", WORKFLOW_ICONS["learner"], "#2465D4", "#EDF3FF"),
        workflow_step("2", "Validation", "Input values are structured for analysis.", WORKFLOW_ICONS["processing"], "#0F172A", "#F4F6F8"),
        workflow_step("3", "Rule Filter", "Strong eligibility signals are reviewed first.", WORKFLOW_ICONS["eligibility"], "#2465D4", "#EDF3FF"),
        workflow_step("4", "Prediction", "The saved model predicts a program category.", WORKFLOW_ICONS["prediction"], "#10935D", "#ECF9F1"),
        workflow_step("5", "Bridge Mapping", "A relevant preparatory course is attached.", WORKFLOW_ICONS["bridge"], "#7047EB", "#F3EFFF"),
        workflow_step("6", "Human Review", "The output is discussed with an advisor.", WORKFLOW_ICONS["advisor"], "#0E9F9A", "#EAF9F8"),
    ]
    st.markdown(f'<div class="panel"><div class="panel-title">System Architecture Overview</div><div class="panel-subtitle">Hybrid recommendation workflow</div><div class="workflow-grid">{"".join(steps)}</div></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("Input Features", "Education type, pathway, stream or trade, strongest area, support need, interest area, average score range, digital skill level, and career cluster.", CONTENT_ICONS["file"], "#2465D4", "#EDF3FF")
    with c2:
        info_card("Prediction Approach", "Rule-based eligibility guidance is combined with the saved scikit-learn prediction pipeline.", WORKFLOW_ICONS["prediction"], "#10935D", "#ECF9F1")
    with c3:
        info_card("Output", "Program category, bridge course, alternative pathway, and a human-readable explanation.", CONTENT_ICONS["clipboard"], "#7047EB", "#F3EFFF")

    c4, c5, c6 = st.columns(3)
    with c4:
        info_card("Explainability", "The recommendation explains how the learner profile connects to the suggested direction.", CONTENT_ICONS["info"], "#0E9F9A", "#EAF9F8")
    with c5:
        info_card("Model Artifact", f"Model file: academic_pathway_model.joblib. Model type: {ARTIFACT['model_name']}.", CONTENT_ICONS["database"], "#2465D4", "#EDF3FF")
    with c6:
        info_card("Human Oversight", "The recommendation remains advisory and must be reviewed against official requirements.", CONTENT_ICONS["review"], "#D99100", "#FFF7E6")

elif selected_page == "Data & Governance":
    page_intro(
        "Data & Governance",
        "Review the prototype's data sources, privacy controls, governance principles, feature use, and model-update approach.",
        COLLABORATION_PHOTO,
        NAVIGATION_ICONS["Data & Governance"],
    )

    governance_cards = [
        ("Data Sources", "Synthetic learner profiles, survey responses, education-domain mappings, and the final model-training dataset.", CONTENT_ICONS["database"], "#2465D4", "#EDF3FF"),
        ("Student Profile Data", "The system uses academic background, strengths, support needs, interests, scores, and digital skills.", CONTENT_ICONS["user"], "#10935D", "#ECF9F1"),
        ("Prototype Dataset", "The learner data is intended for research and prototype evaluation rather than official placement decisions.", CONTENT_ICONS["file"], "#7047EB", "#F3EFFF"),
        ("Privacy & Data Minimization", "National ID, exact address, phone number, and other unnecessary identifiers are not required.", CONTENT_ICONS["lock"], "#0E9F9A", "#EAF9F8"),
        ("Human Review", "Recommendations should be reviewed with learners, guardians, advisors, schools, and relevant institutions.", CONTENT_ICONS["review"], "#D99100", "#FFF7E6"),
        ("Model & Rule Updates", "Model artifacts and eligibility mappings should be reviewed when policies, curricula, or admission requirements change.", CONTENT_ICONS["refresh"], "#2465D4", "#EDF3FF"),
    ]
    for row_start in range(0, len(governance_cards), 3):
        cols = st.columns(3)
        for col, item in zip(cols, governance_cards[row_start:row_start + 3]):
            with col:
                info_card(*item)

    st.markdown('<div class="notice notice-gold"><strong>Important:</strong> The platform is a capstone prototype, not an official government admissions or placement system.</div>', unsafe_allow_html=True)

elif selected_page == "Responsible Use":
    page_intro(
        "Responsible Use",
        "Understand the system's advisory role, human-oversight requirement, fairness principles, and limitations before using a recommendation.",
        ADVISOR_PHOTO,
        NAVIGATION_ICONS["Responsible Use"],
    )

    c1, c2 = st.columns(2)
    with c1:
        info_card("Advisory Use Only", "The system supports academic exploration and discussion. It does not make official admission or placement decisions.", CONTENT_ICONS["shield"], "#10935D", "#ECF9F1")
        info_card("Privacy & Fairness", "The system minimizes personal data and should be reviewed for consistent treatment across learner profiles.", CONTENT_ICONS["lock"], "#7047EB", "#F3EFFF")
    with c2:
        info_card("Human Oversight Required", "Recommendations should be reviewed with advisors, institutions, parents, guardians, or school leadership.", CONTENT_ICONS["review"], "#2465D4", "#EDF3FF")
        info_card("Not an Admission Decision", "A recommendation does not guarantee admission, scholarship selection, placement, or eligibility.", CONTENT_ICONS["warning"], "#D99100", "#FFF7E6")

    st.markdown('<div class="notice notice-red"><strong>Disclaimer:</strong> This academic prototype is not an official MINEDUC, REB, RTB, university, or scholarship decision. Always confirm current entry requirements with the relevant institution.</div>', unsafe_allow_html=True)

elif selected_page == "Feedback":
    page_intro(
        "Share Your Feedback",
        "Rate the recommendation's usefulness, explanation clarity, and interface usability. Your feedback supports prototype evaluation and improvement.",
        FEEDBACK_PHOTO,
        NAVIGATION_ICONS["Feedback"],
    )

    left, right = st.columns([1.15, .85])
    with left:
        with st.form("feedback_form"):
            user_role = st.selectbox("Your Role", ["Student", "Academic Advisor", "Teacher", "Parent/Guardian", "Education Stakeholder", "Other"])
            usefulness = st.slider("How useful is the recommendation output?", 1, 5, 4)
            clarity = st.slider("How clear is the explanation?", 1, 5, 4)
            usability = st.slider("How easy is the system to use?", 1, 5, 4)
            comments = st.text_area("Additional Comments", placeholder="Share suggestions to improve the recommendation, explanation, or interface.")
            feedback_submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

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
            st.markdown('<div class="notice notice-green"><strong>Thank you.</strong> Your feedback has been recorded successfully.</div>', unsafe_allow_html=True)

    with right:
        info_card("What Your Feedback Supports", "Recommendation relevance, explanation clarity, interface usability, and priorities for future prototype improvement.", CONTENT_ICONS["message"], "#2465D4", "#EDF3FF")
        info_card("Privacy Note", "Do not include sensitive personal information in the open comment field.", CONTENT_ICONS["lock"], "#10935D", "#ECF9F1")
        info_card("Evaluation Use", "Feedback is used for capstone evaluation and interface improvement rather than official learner assessment.", CONTENT_ICONS["clipboard"], "#7047EB", "#F3EFFF")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        <div><strong>Rwanda Academic Guidance Portal — Decision-Support Prototype</strong><br>Supporting informed academic and career exploration.</div>
        <div>Privacy Guidance &nbsp; | &nbsp; Responsible Use &nbsp; | &nbsp; Feedback</div>
    </div>
    """,
    unsafe_allow_html=True,
)

