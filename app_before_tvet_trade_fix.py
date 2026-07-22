import base64
import re
from functools import lru_cache
from urllib.parse import quote
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import numpy as np
import shap

# =========================================================
# LOCAL PROJECT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "academic_pathway_model_v2.joblib"

IMAGE_DIR = BASE_DIR / "static" / "images"
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


# Relevant interest areas offered for each TVET sector.
# The trade-aligned interest is always displayed first, while learners can
# explore other closely related directions without creating unrelated profiles.
TVET_SECTOR_TO_INTEREST_AREAS = {
    "ICT and Multimedia Sector": [
        "Networking and Cloud Infrastructure",
        "Cybersecurity",
        "Software Engineering and Development",
        "Data Science, AI, and Machine Learning",
        "Multimedia and Digital Content Production",
        "UI/UX Design and Digital Product Design",
    ],
    "Construction and Building Services Sector": [
        "Construction and Technical Services",
        "Science, Engineering, and Mathematics",
        "Manufacturing and Industrial Production",
        "UI/UX Design and Digital Product Design",
        "Business Management and Entrepreneurship",
    ],
    "Hospitality and Tourism Sector": [
        "Hospitality, Tourism, and Service Sector",
        "Business Management and Entrepreneurship",
        "Communication, Marketing, and Public Relations",
        "Languages, Translation, and Interpretation",
        "International Relations and Diplomacy",
    ],
    "Energy and Technical Services Sector": [
        "Science, Engineering, and Mathematics",
        "Construction and Technical Services",
        "Manufacturing and Industrial Production",
        "Networking and Cloud Infrastructure",
        "Data Science, AI, and Machine Learning",
    ],
    "Manufacturing, Mining, and Transport Sector": [
        "Transport and Logistics",
        "Manufacturing and Industrial Production",
        "Science, Engineering, and Mathematics",
        "Construction and Technical Services",
        "Business Management and Entrepreneurship",
    ],
    "Agriculture and Food Processing Sector": [
        "Agriculture, Food Processing, and Environment",
        "Science, Engineering, and Mathematics",
        "Business Management and Entrepreneurship",
        "Hospitality, Tourism, and Service Sector",
        "Data Science, AI, and Machine Learning",
    ],
    "Business and Arts/Crafts Sector": [
        "Finance, Accounting, and Banking",
        "Business Management and Entrepreneurship",
        "Arts, Media, and Creative Industries",
        "Communication, Marketing, and Public Relations",
        "UI/UX Design and Digital Product Design",
    ],
}


def get_tvet_interest_options(tvet_sector, stream_or_trade):
    """Return relevant TVET interest options with the trade suggestion first."""

    default_interest = TVET_TRADE_TO_INTEREST_AREA.get(
        stream_or_trade,
        "Science, Engineering, and Mathematics",
    )

    sector_options = TVET_SECTOR_TO_INTEREST_AREAS.get(
        tvet_sector,
        RWANDA_INTEREST_AREAS,
    )

    return list(dict.fromkeys([default_interest, *sector_options]))


def get_tvet_career_options(interest_area, stream_or_trade):
    """Return relevant career options for the selected TVET interest area."""

    default_career = TVET_TRADE_TO_CAREER_CLUSTER.get(
        stream_or_trade,
        "Business Administration and Management",
    )

    related_careers = INTEREST_AREA_TO_CAREER_CLUSTERS.get(
        interest_area,
        PROGRAM_CATEGORY_OPTIONS,
    )

    # Keep the trade-aligned career first only when it is compatible with
    # the chosen interest. Otherwise show the interest-related careers only.
    if default_career in related_careers:
        return list(dict.fromkeys([default_career, *related_careers]))

    return list(dict.fromkeys(related_careers))

# =========================================================
# LOAD MODEL ARTIFACT
# =========================================================
@st.cache_resource
def load_model_artifact():
    """Load and validate the final ML-first deployment bundle."""

    if not MODEL_PATH.exists():
        st.error(
            "The final model file was not found at:\n\n"
            f"{MODEL_PATH}\n\n"
            "Place academic_pathway_model_v2.joblib "
            "inside the models folder."
        )
        st.stop()

    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as error:
        st.error(
            "The model artifact could not be loaded. "
            "Confirm that requirements.txt uses "
            "scikit-learn==1.6.1."
        )
        st.exception(error)
        st.stop()

    required_keys = {
        "model_pipeline",
        "label_encoder",
        "feature_columns",
        "bridge_course_mapping",
        "score_range_order",
        "digital_skill_order",
        "shap_background_profiles",
        "metadata",
    }

    if not isinstance(artifact, dict):
        st.error(
            "The saved model must be a dictionary-based "
            "deployment bundle."
        )
        st.stop()

    missing_keys = required_keys - set(artifact.keys())

    if missing_keys:
        st.error(
            "The model bundle is missing required components: "
            + ", ".join(sorted(missing_keys))
        )
        st.stop()

    model_pipeline = artifact["model_pipeline"]
    label_encoder = artifact["label_encoder"]
    feature_columns = artifact["feature_columns"]
    bridge_course_mapping = artifact[
        "bridge_course_mapping"
    ]
    shap_background_profiles = artifact[
        "shap_background_profiles"
    ]
    metadata = artifact["metadata"]

    # Retrieve the fitted preprocessing and SVM components
    fitted_preprocessor = model_pipeline.named_steps[
        "preprocessor"
    ]

    fitted_classifier = model_pipeline.named_steps[
        "classifier"
    ]

    # Transform the saved SHAP background sample
    transformed_background = fitted_preprocessor.transform(
        shap_background_profiles
    )

    if hasattr(transformed_background, "toarray"):
        transformed_background = (
            transformed_background.toarray()
        )

    transformed_background = np.asarray(
        transformed_background
    )

    # Store the average transformed training profile so the app can
    # calculate feature-level linear-model contributions without requiring
    # an additional explainability package at deployment time.
    transformed_background_mean = transformed_background.mean(axis=0)

    transformed_feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    # Link one-hot encoded columns to the original features
    original_feature_groups = {}

    for feature_index, transformed_name in enumerate(
        transformed_feature_names
    ):
        clean_name = transformed_name.split("__", 1)[-1]

        matched_feature = None

        for original_feature in sorted(
            feature_columns,
            key=len,
            reverse=True
        ):
            if (
                clean_name == original_feature
                or clean_name.startswith(
                    f"{original_feature}_"
                )
            ):
                matched_feature = original_feature
                break

        if matched_feature is not None:
            original_feature_groups.setdefault(
                matched_feature,
                []
            ).append(feature_index)

    return {
        "model_pipeline": model_pipeline,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "bridge_course_mapping": bridge_course_mapping,
        "score_range_order": artifact[
            "score_range_order"
        ],
        "digital_skill_order": artifact[
            "digital_skill_order"
        ],
        "shap_background_profiles":
            shap_background_profiles,
        "transformed_background_mean": transformed_background_mean,
        "transformed_feature_names":
            transformed_feature_names,
        "original_feature_groups":
            original_feature_groups,
        "metadata": metadata,
    }


ARTIFACT = load_model_artifact()

MODEL = ARTIFACT["model_pipeline"]
LABEL_ENCODER = ARTIFACT["label_encoder"]
MODEL_INPUT_FEATURES = ARTIFACT["feature_columns"]

MODEL_BROAD_CATEGORY_TO_BRIDGE_COURSE = ARTIFACT[
    "bridge_course_mapping"
]

# Exact program-level bridge courses already defined above.
PROGRAM_CATEGORY_TO_BRIDGE_COURSE = dict(
    DEFAULT_PROGRAM_CATEGORY_TO_BRIDGE_COURSE
)

MODEL_BACKGROUND_MEAN = np.asarray(
    ARTIFACT["transformed_background_mean"]
)

TRANSFORMED_FEATURE_NAMES = ARTIFACT[
    "transformed_feature_names"
]

ORIGINAL_FEATURE_GROUPS = ARTIFACT[
    "original_feature_groups"
]

MODEL_METADATA = ARTIFACT["metadata"]

# Genuine SHAP explainer for the trained linear SVM.
MODEL_SHAP_BACKGROUND = MODEL.named_steps[
    "preprocessor"
].transform(
    ARTIFACT["shap_background_profiles"]
)

if hasattr(MODEL_SHAP_BACKGROUND, "toarray"):
    MODEL_SHAP_BACKGROUND = (
        MODEL_SHAP_BACKGROUND.toarray()
    )

MODEL_SHAP_BACKGROUND = np.asarray(
    MODEL_SHAP_BACKGROUND
)


@st.cache_resource
def build_shap_explainer():
    """Create one reusable SHAP explainer for the fitted SVM."""

    return shap.LinearExplainer(
        MODEL.named_steps["classifier"],
        MODEL_SHAP_BACKGROUND,
    )


SHAP_EXPLAINER = build_shap_explainer()

# Alternative pathways aligned with the 16 final model categories
PROGRAM_CATEGORY_TO_ALTERNATIVE_PATHWAY = {
    "Medicine and Surgery":
        "Biomedical Laboratory Sciences, Nursing and Midwifery, "
        "Pharmacy, or additional science preparation.",

    "Nursing and Midwifery":
        "Medicine and Surgery, Biomedical Laboratory Sciences, "
        "community health, or health-science preparation.",

    "Pharmacy and Pharmaceutical Sciences":
        "Biomedical Laboratory Sciences, Biotechnology, "
        "Medicine, or chemistry-focused preparation.",

    "Biomedical Laboratory Sciences":
        "Biotechnology, Pharmacy, Nursing, "
        "or laboratory-skills preparation.",

    "Biotechnology and Applied Biosciences":
        "Biomedical Laboratory Sciences, Agriculture, "
        "Environmental Studies, or laboratory preparation.",

    "Data Science, Statistics and Analytics":
        "Computing and Information Technology, Economics, "
        "or mathematics and statistics preparation.",

    "Computing, Software and Information Technology":
        "Data Science and Analytics, Engineering Systems, "
        "or programming and digital-skills preparation.",

    "Engineering, Construction and Technical Systems":
        "Computing and Information Technology, Environmental Studies, "
        "or mathematics, physics and technical-drawing preparation.",

    "Business, Finance and Economics":
        "Law and Governance, Data Science and Analytics, "
        "or business and financial-skills preparation.",

    "Law, Governance and International Relations":
        "Business and Economics, Social Sciences, "
        "or academic-writing and governance preparation.",

    "Social Sciences, Psychology and Community Development":
        "Education, Law and Governance, "
        "or research-methods and communication preparation.",

    "Education and Teacher Training":
        "Languages and Communication, Social Sciences, "
        "or teaching-methods preparation.",

    "Languages, Communication and Media":
        "Education, Law and International Relations, "
        "or communication and academic-writing preparation.",

    "Tourism and Hospitality Management":
        "Business and Management, Languages and Communication, "
        "or customer-service and hospitality preparation.",

    "Agriculture, Food and Environmental Studies":
        "Engineering and Technical Systems, Biotechnology, "
        "or environmental and agricultural-skills preparation.",

    "Creative Arts, Fashion and Digital Design":
        "Languages, Communication and Media, Computing, "
        "or portfolio and digital-design preparation."
}

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


MODEL_STREAM_OR_TRADE_MAP = {
    "Arts and Humanities": "General Arts",
    "Languages": "Language Stream",
    "Stream 1": "Stream 1",
    "Stream 2": "Stream 2",

    "Software Development": "Software Development",
    "Networking and Internet Technologies":
        "Networking and Telecommunication",
    "Computer Systems and Architecture":
        "Networking and Telecommunication",
    "Multimedia Production":
        "Multimedia and Digital Design",
    "Software Programming and Embedded Systems":
        "Software Development",

    "Building Construction": "Masonry / Construction",
    "Public Works / Road Construction":
        "Masonry / Construction",
    "Plumbing Technology": "Domestic Plumbing / Water",
    "Land Surveying": "Masonry / Construction",
    "Interior Design / Painting and Decoration":
        "Masonry / Construction",

    "Culinary Arts": "Culinary Arts / Front Office",
    "Food and Beverage Operations":
        "Culinary Arts / Front Office",
    "Front Office and Housekeeping Operations":
        "Culinary Arts / Front Office",
    "Tourism": "Culinary Arts / Front Office",

    "Electrical Technology / Electrical Installation":
        "Electrical Technology",
    "Industrial Electricity": "Electrical Technology",
    "Renewable Energy Technology": "Electrical Technology",
    "Electronics and Telecommunication":
        "Networking and Telecommunication",

    "Automobile Technology / Automobile Mechanic":
        "Welding / Mechanical",
    "Auto Electricity and Electronic Systems":
        "Welding / Mechanical",
    "Manufacturing and Production Technology":
        "Welding / Mechanical",
    "Welding and Fabrication": "Welding / Mechanical",

    "Crop Production": "Crop Production / Animal Health",
    "Animal Health / Livestock Farming":
        "Crop Production / Animal Health",
    "Food Processing": "Food Processing",
    "Forestry and Wood Technology / Carpentry":
        "Masonry / Construction",

    "Accounting": "Accountancy",
    "Fashion Design and Tailoring / Garment Making":
        "Tailoring / Fashion Design",
    "Fine and Plastic Arts":
        "Multimedia and Digital Design",
    "Hairdressing and Beauty Therapy / Cosmetology":
        "Tailoring / Fashion Design",
}


MODEL_INTEREST_AREA_MAP = {
    "Agriculture, Food Processing, and Environment":
        "Technical Trades",
    "Arts, Media, and Creative Industries":
        "Humanities & Business",
    "Business Management and Entrepreneurship":
        "Humanities & Business",
    "Communication, Marketing, and Public Relations":
        "Languages & Communications",
    "Construction and Technical Services":
        "Technical Trades",
    "Cybersecurity": "STEM Fields",
    "Data Science, AI, and Machine Learning":
        "STEM Fields",
    "Education": "Humanities & Business",
    "Finance, Accounting, and Banking":
        "Humanities & Business",
    "Hospitality, Tourism, and Service Sector":
        "Technical Trades",
    "International Relations and Diplomacy":
        "Languages & Communications",
    "Languages, Translation, and Interpretation":
        "Languages & Communications",
    "Law, Governance, and Public Administration":
        "Humanities & Business",
    "Manufacturing and Industrial Production":
        "Technical Trades",
    "Medicine and Health Sciences": "STEM Fields",
    "Multimedia and Digital Content Production":
        "Languages & Communications",
    "Networking and Cloud Infrastructure":
        "STEM Fields",
    "Psychology, Counseling, and Social Sciences":
        "Humanities & Business",
    "Science, Engineering, and Mathematics":
        "STEM Fields",
    "Software Engineering and Development":
        "STEM Fields",
    "Transport and Logistics": "Technical Trades",
    "UI/UX Design and Digital Product Design":
        "Languages & Communications",
}


MODEL_CAREER_CLUSTER_MAP = {
    "Medicine and Surgery": "Medicine & Health",
    "Nursing and Midwifery": "Medicine & Health",
    "Pharmacy and Pharmaceutical Sciences":
        "Medicine & Health",
    "Biomedical Laboratory Sciences":
        "Medicine & Health",
    "Biotechnology and Applied Biosciences":
        "Research and Basic Sciences",

    "Civil Engineering and Construction Technology":
        "Engineering and Technology",
    "Electrical Engineering and Power Systems":
        "Engineering and Technology",
    "Electrical Technology and Power Systems":
        "Engineering and Infrastructure",
    "Mechanical and Manufacturing Engineering":
        "Engineering and Infrastructure",
    "Mechanical Fabrication and Welding Technology":
        "Engineering and Infrastructure",
    "Water, Sanitation and Building Services Technology":
        "Engineering and Infrastructure",

    "Data Science and Analytics": "Technology and Data",
    "Statistics and Applied Mathematics":
        "Research and Basic Sciences",
    "Computer Science and Information Systems":
        "Technology and Data",
    "Software Engineering and Application Development":
        "Technology and Data",
    "Information Technology, Networking and Information Security":
        "Information and Communication Technology (ICT)",
    "Information Technology and Systems Administration":
        "Information and Communication Technology (ICT)",
    "Computer Engineering and Embedded Systems":
        "Information and Communication Technology (ICT)",

    "Finance and Banking": "Business & Finance",
    "Accounting and Finance": "Business & Finance",
    "Economics and Development Finance":
        "Business & Finance",
    "Business Administration and Management":
        "Business & Finance",

    "Geography, GIS and Environmental Planning":
        "Tourism & Environment",
    "Urban and Regional Planning":
        "Tourism & Environment",
    "Crop Science and Agribusiness":
        "Agriculture and Food Processing",
    "Crop Production and Agribusiness":
        "Agriculture and Food Processing",
    "Food Science and Processing Technology":
        "Agriculture and Food Processing",
    "Environmental Science and Sustainability":
        "Tourism & Environment",

    "Public Administration and Governance":
        "Law & Governance",
    "Law and Legal Studies": "Law & Governance",
    "International Relations and Diplomacy":
        "International Relations",

    "Psychology and Counselling Studies":
        "Social Sciences",
    "Sociology and Social Sciences":
        "Social Sciences",
    "Social Work and Community Development":
        "Social Sciences",
    "Development Studies and Community Development":
        "Social Sciences",

    "Education in Arts and Humanities": "Education",
    "Education in Languages": "Education",

    "Translation and Interpretation":
        "International Relations",
    "English, Literature and Language Studies":
        "Media & Communication",
    "Journalism and Media Studies":
        "Media & Communication",
    "Communication and Public Relations":
        "Media & Communication",

    "Hospitality Management and Culinary Arts":
        "Hospitality, Tourism, and Service Sector",
    "Hospitality Management and Food and Beverage Services":
        "Hospitality, Tourism, and Service Sector",
    "Hospitality Management and Room Division":
        "Hospitality, Tourism, and Service Sector",
    "Tourism and Travel Management":
        "Hospitality, Tourism, and Service Sector",

    "Multimedia, Graphic Design and Digital Media Production":
        "Creative Industries",
    "Fashion Design and Garment Production":
        "Business, Art, and Craft",
}

MODEL_BEST_SUBJECTS = {
    "Biology",
    "Chemistry",
    "Economics",
    "English",
    "French",
    "Geography",
    "History",
    "Kiswahili",
    "Literature",
    "Mathematics",
    "Physics",
    "Practical Workshop / Execution",
    "Religious Studies",
}

MODEL_WEAKEST_SUBJECTS = {
    "Academic Theory",
    "Advanced Calculus",
    "Biology",
    "Chemistry",
    "Geography",
    "History",
    "Kinyarwanda",
    "Literature",
    "Mathematics",
    "Physics",
    "Research Writing",
}


BROAD_CATEGORY_TO_SPECIFIC_PROGRAMS = {
    "Agriculture, Food and Environmental Studies": [
        "Crop Science and Agribusiness",
        "Crop Production and Agribusiness",
        "Food Science and Processing Technology",
        "Environmental Science and Sustainability",
        "Geography, GIS and Environmental Planning",
        "Urban and Regional Planning",
    ],
    "Biomedical Laboratory Sciences": [
        "Biomedical Laboratory Sciences",
    ],
    "Biotechnology and Applied Biosciences": [
        "Biotechnology and Applied Biosciences",
    ],
    "Business, Finance and Economics": [
        "Finance and Banking",
        "Accounting and Finance",
        "Economics and Development Finance",
        "Business Administration and Management",
    ],
    "Computing, Software and Information Technology": [
        "Computer Science and Information Systems",
        "Software Engineering and Application Development",
        "Information Technology, Networking and Information Security",
        "Information Technology and Systems Administration",
        "Computer Engineering and Embedded Systems",
    ],
    "Creative Arts, Fashion and Digital Design": [
        "Multimedia, Graphic Design and Digital Media Production",
        "Fashion Design and Garment Production",
    ],
    "Data Science, Statistics and Analytics": [
        "Data Science and Analytics",
        "Statistics and Applied Mathematics",
    ],
    "Education and Teacher Training": [
        "Education in Arts and Humanities",
        "Education in Languages",
    ],
    "Engineering, Construction and Technical Systems": [
        "Civil Engineering and Construction Technology",
        "Electrical Engineering and Power Systems",
        "Electrical Technology and Power Systems",
        "Mechanical and Manufacturing Engineering",
        "Mechanical Fabrication and Welding Technology",
        "Water, Sanitation and Building Services Technology",
    ],
    "Languages, Communication and Media": [
        "Translation and Interpretation",
        "English, Literature and Language Studies",
        "Journalism and Media Studies",
        "Communication and Public Relations",
    ],
    "Law, Governance and International Relations": [
        "Public Administration and Governance",
        "Law and Legal Studies",
        "International Relations and Diplomacy",
    ],
    "Medicine and Surgery": [
        "Medicine and Surgery",
    ],
    "Nursing and Midwifery": [
        "Nursing and Midwifery",
    ],
    "Pharmacy and Pharmaceutical Sciences": [
        "Pharmacy and Pharmaceutical Sciences",
    ],
    "Social Sciences, Psychology and Community Development": [
        "Psychology and Counselling Studies",
        "Sociology and Social Sciences",
        "Social Work and Community Development",
        "Development Studies and Community Development",
    ],
    "Tourism and Hospitality Management": [
        "Hospitality Management and Culinary Arts",
        "Hospitality Management and Food and Beverage Services",
        "Hospitality Management and Room Division",
        "Tourism and Travel Management",
    ],
}

SPECIFIC_PROGRAM_TO_BROAD_CATEGORY = {
    program: broad_category
    for broad_category, programs in (
        BROAD_CATEGORY_TO_SPECIFIC_PROGRAMS.items()
    )
    for program in programs
}

BROAD_CATEGORY_DEFAULT_PROGRAM = {
    broad_category: programs[0]
    for broad_category, programs in (
        BROAD_CATEGORY_TO_SPECIFIC_PROGRAMS.items()
    )
}

# These families allow the model's broad category and an explicitly selected
# closely related program to work together without replacing the SVM decision.
RELATED_BROAD_CATEGORY_FAMILIES = [
    {
        "Medicine and Surgery",
        "Nursing and Midwifery",
        "Pharmacy and Pharmaceutical Sciences",
        "Biomedical Laboratory Sciences",
        "Biotechnology and Applied Biosciences",
    },
    {
        "Computing, Software and Information Technology",
        "Data Science, Statistics and Analytics",
        "Engineering, Construction and Technical Systems",
    },
    {
        "Languages, Communication and Media",
        "Creative Arts, Fashion and Digital Design",
    },
    {
        "Law, Governance and International Relations",
        "Social Sciences, Psychology and Community Development",
        "Education and Teacher Training",
    },
]

MODEL_BEST_SUBJECTS = {
    "Biology",
    "Chemistry",
    "Economics",
    "English",
    "French",
    "Geography",
    "History",
    "Kiswahili",
    "Literature",
    "Mathematics",
    "Physics",
    "Practical Workshop / Execution",
    "Religious Studies",
}

MODEL_WEAKEST_SUBJECTS = {
    "Academic Theory",
    "Advanced Calculus",
    "Biology",
    "Chemistry",
    "Geography",
    "History",
    "Kinyarwanda",
    "Literature",
    "Mathematics",
    "Physics",
    "Research Writing",
}

FEATURE_DISPLAY_NAMES = {
    "EducationType": "education type",
    "Pathway": "academic pathway",
    "Stream_or_Trade": "stream or TVET trade",
    "BestSubject": "strongest subject or competency",
    "WeakestSubject": "subject or competency needing support",
    "InterestArea": "interest area",
    "AverageScoreRange": "average score range",
    "DigitalSkillLevel": "digital skill level",
    "CareerCluster": "career direction",
}


def broad_categories_are_related(first_category, second_category):
    """Return True when two broad categories belong to one related family."""

    if first_category == second_category:
        return True

    return any(
        first_category in family and second_category in family
        for family in RELATED_BROAD_CATEGORY_FAMILIES
    )


def normalize_model_subject(value, column_name, education_type):
    """Translate dashboard subject labels into the model vocabulary."""

    value = str(value).strip()

    if education_type == "TVET":
        return (
            "Practical Workshop / Execution"
            if column_name == "BestSubject"
            else "Academic Theory"
        )

    subject_map = {
        "Entrepreneurship": "Economics",
        "Psychology": "Literature",
        "Kinyarwanda": (
            "Kinyarwanda"
            if column_name == "WeakestSubject"
            else "English"
        ),
    }

    value = subject_map.get(value, value)

    if column_name == "BestSubject":
        return (
            value
            if value in MODEL_BEST_SUBJECTS
            else "Mathematics"
        )

    return (
        value
        if value in MODEL_WEAKEST_SUBJECTS
        else "Research Writing"
    )


def prepare_profile_for_model(student_profile):
    """
    Translate the user-facing profile into the exact vocabulary used
    during SVM training. The original dashboard values remain unchanged.
    """

    profile = dict(student_profile)
    education_type = str(
        profile.get("EducationType", "General Education")
    ).strip()
    original_stream_or_trade = str(
        profile.get("Stream_or_Trade", "")
    ).strip()

    if education_type == "TVET":
        profile["Pathway"] = "TVET Route"
        profile["Stream_or_Trade"] = (
            MODEL_STREAM_OR_TRADE_MAP.get(
                original_stream_or_trade,
                "Welding / Mechanical",
            )
        )
        profile["BestSubject"] = "Practical Workshop / Execution"
        profile["WeakestSubject"] = "Academic Theory"
        profile["InterestArea"] = "Technical Trades"

        selected_specific_program = str(
            profile.get("CareerCluster", "")
        ).strip()
        trade_aligned_program = TVET_TRADE_TO_CAREER_CLUSTER.get(
            original_stream_or_trade,
            selected_specific_program,
        )
        program_for_model = (
            selected_specific_program
            if selected_specific_program in MODEL_CAREER_CLUSTER_MAP
            else trade_aligned_program
        )
        profile["CareerCluster"] = MODEL_CAREER_CLUSTER_MAP.get(
            program_for_model,
            "Engineering and Infrastructure",
        )

    else:
        profile["Stream_or_Trade"] = (
            MODEL_STREAM_OR_TRADE_MAP.get(
                original_stream_or_trade,
                original_stream_or_trade,
            )
        )
        profile["InterestArea"] = (
            MODEL_INTEREST_AREA_MAP.get(
                str(profile.get("InterestArea", "")).strip(),
                "Humanities & Business",
            )
        )
        profile["CareerCluster"] = (
            MODEL_CAREER_CLUSTER_MAP.get(
                str(profile.get("CareerCluster", "")).strip(),
                "Business & Finance",
            )
        )
        profile["BestSubject"] = normalize_model_subject(
            profile.get("BestSubject", ""),
            "BestSubject",
            education_type,
        )
        profile["WeakestSubject"] = normalize_model_subject(
            profile.get("WeakestSubject", ""),
            "WeakestSubject",
            education_type,
        )

    return profile


def prepare_student_dataframe(student_profile):
    """Create the exact model input columns in the saved pipeline order."""

    model_profile = prepare_profile_for_model(student_profile)
    student_df = pd.DataFrame([model_profile])

    missing_features = [
        feature
        for feature in MODEL_INPUT_FEATURES
        if feature not in student_df.columns
    ]

    if missing_features:
        raise ValueError(
            "The learner profile is missing required features: "
            + ", ".join(missing_features)
        )

    return student_df[MODEL_INPUT_FEATURES].copy()


def format_course_list(courses):
    """Convert a bridge-course list into natural, readable text."""

    if courses is None:
        return "Academic Writing, Digital Literacy, and Study Skills"

    if isinstance(courses, str):
        return courses

    courses = list(courses)

    if not courses:
        return "Academic Writing, Digital Literacy, and Study Skills"

    if len(courses) == 1:
        return courses[0]

    if len(courses) == 2:
        return f"{courses[0]} and {courses[1]}"

    return ", ".join(courses[:-1]) + f", and {courses[-1]}"


def get_model_ranking(student_profile):
    """Return the SVM broad-category prediction and ranked decision scores."""

    student_df = prepare_student_dataframe(student_profile)
    predicted_class_id = int(MODEL.predict(student_df)[0])
    predicted_broad_category = LABEL_ENCODER.inverse_transform(
        [predicted_class_id]
    )[0]

    decision_scores = np.asarray(
        MODEL.decision_function(student_df)[0]
    )
    ranked_class_ids = np.argsort(decision_scores)[::-1]
    ranked_broad_categories = LABEL_ENCODER.inverse_transform(
        ranked_class_ids
    ).tolist()

    return {
        "student_df": student_df,
        "predicted_class_id": predicted_class_id,
        "predicted_broad_category": predicted_broad_category,
        "decision_scores": decision_scores,
        "ranked_class_ids": ranked_class_ids,
        "ranked_broad_categories": ranked_broad_categories,
    }


def get_preferred_specific_program(student_profile):
    """Read the learner's explicit program direction from the profile."""

    selected_program = student_profile.get("CareerCluster")

    if (
        selected_program
        and selected_program in SPECIFIC_PROGRAM_TO_BROAD_CATEGORY
    ):
        return selected_program

    if student_profile.get("EducationType") == "TVET":
        return TVET_TRADE_TO_CAREER_CLUSTER.get(
            student_profile.get("Stream_or_Trade"),
            selected_program,
        )

    return selected_program


def refine_specific_program(student_profile, model_ranking):
    """
    Refine the SVM broad field to a specific program without replacing
    the model. A learner-selected program is used only when its broad field
    is model-supported or closely related to the top SVM field.
    """

    predicted_broad_category = model_ranking[
        "predicted_broad_category"
    ]
    ranked_broad_categories = model_ranking[
        "ranked_broad_categories"
    ]
    preferred_program = get_preferred_specific_program(student_profile)
    preferred_broad_category = (
        SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(preferred_program)
    )

    if preferred_program in SPECIFIC_PROGRAM_TO_BROAD_CATEGORY:
        if (
            preferred_broad_category in ranked_broad_categories[:3]
            or broad_categories_are_related(
                preferred_broad_category,
                predicted_broad_category,
            )
        ):
            return preferred_program

    interest_area = student_profile.get("InterestArea")
    interest_candidates = INTEREST_AREA_TO_CAREER_CLUSTERS.get(
        interest_area,
        [],
    )

    for candidate in interest_candidates:
        candidate_broad_category = (
            SPECIFIC_PROGRAM_TO_BROAD_CATEGORY.get(candidate)
        )
        if candidate_broad_category == predicted_broad_category:
            return candidate

    return BROAD_CATEGORY_DEFAULT_PROGRAM.get(
        predicted_broad_category,
        preferred_program or predicted_broad_category,
    )


def get_program_alternative(program, model_ranking):
    """Return a related alternative pathway for advisor discussion."""

    if program in DEFAULT_ALTERNATIVE_PATHWAY:
        return DEFAULT_ALTERNATIVE_PATHWAY[program]

    for broad_category in model_ranking["ranked_broad_categories"][1:]:
        alternative_program = BROAD_CATEGORY_DEFAULT_PROGRAM.get(
            broad_category
        )
        if alternative_program and alternative_program != program:
            return alternative_program

    return "A related diploma, foundation, or advisor-recommended pathway."


def get_grouped_model_explanation(student_profile, model_ranking):
    """
    Aggregate the trained linear model's feature contributions back into
    the nine original learner-profile fields.

    This calculation uses genuine SHAP values from the fitted SVM and
    groups encoded contributions into the original learner-profile fields.
    """

    student_df = model_ranking["student_df"]
    fitted_preprocessor = MODEL.named_steps["preprocessor"]
    transformed_profile = fitted_preprocessor.transform(student_df)

    if hasattr(transformed_profile, "toarray"):
        transformed_profile = transformed_profile.toarray()

    transformed_profile = np.asarray(
        transformed_profile
    )

    predicted_class_id = model_ranking[
        "predicted_class_id"
    ]

    shap_result = SHAP_EXPLAINER(
        transformed_profile
    )

    shap_values = np.asarray(
        shap_result.values
    )

    if shap_values.ndim == 3:
        # Multiclass LinearSVC output:
        # learner x encoded feature x class
        encoded_contributions = shap_values[
            0,
            :,
            predicted_class_id,
        ]

    elif shap_values.ndim == 2:
        encoded_contributions = shap_values[0]

    else:
        return []

    explanation_rows = []

    for feature in MODEL_INPUT_FEATURES:
        encoded_indices = ORIGINAL_FEATURE_GROUPS.get(feature, [])
        if not encoded_indices:
            continue

        contribution = float(
            encoded_contributions[encoded_indices].sum()
        )

        display_value = student_profile.get(
            feature,
            "Not recorded",
        )

        explanation_rows.append(
            {
                "feature": feature,
                "display_name": FEATURE_DISPLAY_NAMES.get(
                    feature,
                    feature,
                ),
                "display_value": display_value,
                "contribution": contribution,
                "absolute_contribution": abs(contribution),
            }
        )

    return sorted(
        explanation_rows,
        key=lambda row: row["absolute_contribution"],
        reverse=True,
    )


def recommend_student(student_profile):
    """
    Generate an ML-first hierarchical recommendation.

    The SVM ranks one of 16 broad fields. The learner's explicit career
    direction or TVET trade then refines that model-supported field to a
    specific program, after which the related bridge course is retrieved.
    """

    model_ranking = get_model_ranking(student_profile)
    recommended_program = refine_specific_program(
        student_profile,
        model_ranking,
    )

    bridge_courses = PROGRAM_CATEGORY_TO_BRIDGE_COURSE.get(
        recommended_program
    )

    if bridge_courses is None:
        bridge_courses = MODEL_BROAD_CATEGORY_TO_BRIDGE_COURSE.get(
            model_ranking["predicted_broad_category"],
            [
                "Academic Writing",
                "Digital Literacy",
                "Study Skills",
            ],
        )

    recommended_bridge_course = format_course_list(bridge_courses)
    alternative_pathway = get_program_alternative(
        recommended_program,
        model_ranking,
    )
    recommendation_source = (
        "Learner profile and academic pathway guidance"
    )

    return (
        recommended_program,
        recommended_bridge_course,
        alternative_pathway,
        recommendation_source,
    )


def build_explanation(
    profile,
    program,
    bridge,
    alternative,
    source,
):
    """Create a clear explanation for learners and academic advisors."""

    def clean_value(value, fallback="Not specified"):
        text = str(value or "").strip()
        return text if text else fallback

    education_type = clean_value(profile.get("EducationType"))
    pathway = clean_value(profile.get("Pathway"))
    stream_or_trade = clean_value(profile.get("Stream_or_Trade"))
    strongest_area = clean_value(profile.get("BestSubject"))
    support_area = clean_value(profile.get("WeakestSubject"))
    interest_area = clean_value(profile.get("InterestArea"))
    career_direction = clean_value(profile.get("CareerCluster"))
    score_range = clean_value(profile.get("AverageScoreRange"))

    model_ranking = get_model_ranking(profile)
    explanation_rows = get_grouped_model_explanation(
        profile,
        model_ranking,
    )
    most_influential = explanation_rows[:3]

    if most_influential:
        influence_items = [
            f"**{row['display_name']}**"
            for row in most_influential
        ]
        if len(influence_items) == 1:
            influence_list = influence_items[0]
        elif len(influence_items) == 2:
            influence_list = (
                f"{influence_items[0]} and {influence_items[1]}"
            )
        else:
            influence_list = (
                f"{influence_items[0]}, {influence_items[1]}, "
                f"and {influence_items[2]}"
            )

        influence_reason = (
            f"The parts of your profile that had the greatest influence "
            f"on this guidance were your {influence_list}."
        )
    else:
        influence_reason = (
            "Your education background, strengths, interests, and career "
            "direction were considered together when preparing this guidance."
        )

    if education_type == "TVET":
        recommendation_reason = (
            f"**{program}** is recommended because your TVET training in "
            f"**{stream_or_trade}** provides a relevant foundation for this "
            f"academic direction. Your strongest competency, "
            f"**{strongest_area}**, also supports further study in this area."
        )

        expected_interest = clean_value(
            TVET_TRADE_TO_INTEREST_AREA.get(stream_or_trade, ""),
            "",
        )
        expected_career = clean_value(
            TVET_TRADE_TO_CAREER_CLUSTER.get(stream_or_trade, ""),
            "",
        )

        interest_is_different = (
            expected_interest
            and interest_area.casefold() != expected_interest.casefold()
        )
        career_is_different = (
            expected_career
            and career_direction.casefold() != expected_career.casefold()
        )

        if interest_is_different or career_is_different:
            preference_reason = (
                f"You also selected **{interest_area}** as an interest area and "
                f"**{career_direction}** as a career direction. These choices "
                f"may differ from your current TVET trade. This recommendation "
                f"therefore gives greater attention to the training and practical "
                f"skills you already have. An academic advisor can help you plan "
                f"a transition when the new direction is your main goal."
            )
        else:
            preference_reason = (
                f"Your interest in **{interest_area}** and your preferred career "
                f"direction, **{career_direction}**, are also consistent with "
                f"this recommendation."
            )
    else:
        recommendation_reason = (
            f"**{program}** is recommended because your background in "
            f"**{pathway}**, particularly **{stream_or_trade}**, provides a "
            f"relevant starting point for this direction. Your strongest subject, "
            f"**{strongest_area}**, also supports the knowledge and skills "
            f"commonly required in this field."
        )
        preference_reason = (
            f"Your interest in **{interest_area}** and your preferred career "
            f"direction, **{career_direction}**, were also considered when "
            f"identifying this recommendation."
        )

    if (
        strongest_area.casefold() == support_area.casefold()
        and strongest_area != "Not specified"
    ):
        support_reason = (
            f"You selected **{strongest_area}** as both your strongest area and "
            f"the area where you need support. Review this choice to make sure "
            f"your profile accurately reflects your learning needs."
        )
    else:
        support_reason = (
            f"You identified **{support_area}** as an area needing support. "
            f"Strengthening this area may improve your overall preparation for "
            f"further study."
        )

    preparation_reason = (
        f"To prepare for **{program}**, the recommended bridge courses are "
        f"**{bridge}**. These courses can help strengthen the knowledge and "
        f"practical skills commonly needed in this program."
    )

    if score_range == "50–59%":
        readiness_reason = (
            f"Your average score range is **50–59%**. This recommendation should "
            f"be treated as a possible pathway rather than a confirmed admission "
            f"match. Completing the suggested bridge preparation and speaking "
            f"with an academic advisor will be especially important."
        )
    elif score_range == "60–69%":
        readiness_reason = (
            f"Your average score range is **60–69%**. Additional preparation in "
            f"the recommended bridge areas may strengthen your readiness and "
            f"improve your options."
        )
    else:
        readiness_reason = (
            f"Your average score range of **{score_range}** provides a useful "
            f"starting point for this direction. Official admission requirements "
            f"should still be confirmed with the institution."
        )

    return (
        f"{recommendation_reason}\n\n"
        f"{preference_reason}\n\n"
        f"{influence_reason}\n\n"
        f"{support_reason}\n\n"
        f"{preparation_reason}\n\n"
        f"{readiness_reason}\n\n"
        f"**Another pathway to consider:** {alternative}\n\n"
        f"This recommendation is intended to support exploration and discussion "
        f"with an academic advisor. It is not an official admission decision."
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
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 14px;
    margin-bottom: 6px;
    border: none;
    border-radius: 9px;
    background: transparent;
    color: #D9E8FA !important;
    cursor: pointer;
}

/* Force unselected navigation text to remain visible */
section[data-testid="stSidebar"]
div[role="radiogroup"] > label p,

section[data-testid="stSidebar"]
div[role="radiogroup"] > label span,

section[data-testid="stSidebar"]
div[role="radiogroup"] > label div {
    color: #D9E8FA !important;
    -webkit-text-fill-color: #D9E8FA !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.08);
}

/* Active navigation item */
section[data-testid="stSidebar"]
div[role="radiogroup"] > label:has(input:checked) {
    background: #4F63D9 !important;
    color: #FFFFFF !important;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.14);
}

section[data-testid="stSidebar"]
div[role="radiogroup"] > label:has(input:checked) p,

section[data-testid="stSidebar"]
div[role="radiogroup"] > label:has(input:checked) span,

section[data-testid="stSidebar"]
div[role="radiogroup"] > label:has(input:checked) div {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Hide Streamlit radio circles */
section[data-testid="stSidebar"]
div[role="radiogroup"] > label > div:first-of-type {
    display: none;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] > label p {
    margin: 0;
    font-size: 14px;
    font-weight: 700;
}

/* Space reserved for each SVG icon */
section[data-testid="stSidebar"]
div[role="radiogroup"] > label::before {
    content: "";
    width: 19px;
    height: 19px;
    flex: none;
    background-position: center;
    background-repeat: no-repeat;
    background-size: contain;
}
"""]

for _idx, _fname in enumerate(NAV_ICON_FILES, start=1):
    _pale = _nav_icon_data_uri(_fname, "#D9E8FA")
    _white = _nav_icon_data_uri(_fname, "#FFFFFF")
    if _pale:
        _nav_rules.append(f'div[role="radiogroup"] > label:nth-child({_idx})::before {{ background-image:url("{_pale}"); }}')
    if _white:
        _nav_rules.append(f'div[role="radiogroup"] > label:nth-child({_idx}):has(input:checked)::before {{ background-image:url("{_white}"); }}')
render_html("<style>" + "\n".join(_nav_rules) + "</style>")

# Rwanda map watermark and sidebar text visibility
_wm_uri = watermark_data_uri()

watermark_css = ""

if _wm_uri:
    watermark_css = f"""
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

    .hero-card {{
        position: relative;
        overflow: hidden;
    }}

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

    .hero-grid {{
        position: relative;
        z-index: 1;
    }}
    """

st.markdown(
    f"""
    <style>
    {watermark_css}

    /* Sidebar information-card visibility */
    section[data-testid="stSidebar"] .sidebar-box,
    section[data-testid="stSidebar"] .sidebar-box p,
    section[data-testid="stSidebar"] .sidebar-box span,
    section[data-testid="stSidebar"] .sidebar-box div,
    section[data-testid="stSidebar"] .sidebar-box b {{
        color: #F8FAFC !important;
        -webkit-text-fill-color: #F8FAFC !important;
        opacity: 1 !important;
    }}

    section[data-testid="stSidebar"] .sidebar-section-title {{
        color: #A7C8FF !important;
        -webkit-text-fill-color: #A7C8FF !important;
        opacity: 1 !important;
    }}

    section[data-testid="stSidebar"] .sidebar-subtitle {{
        color: #CFE3FF !important;
        -webkit-text-fill-color: #CFE3FF !important;
        opacity: 1 !important;
    }}
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
        <b>Model:</b> {MODEL_METADATA['model_name']}<br>
        <b>Model file:</b> academic_pathway_model_v2.joblib<br>
        <b>Test accuracy:</b> {MODEL_METADATA['accuracy'] * 100:.2f}%<br>
        <b>Macro F1:</b> {MODEL_METADATA['macro_f1'] * 100:.2f}%<br>
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

                interest_options = get_tvet_interest_options(
                    tvet_sector,
                    stream_or_trade,
                )
                interest_area = st.selectbox(
                    "Interest Area",
                    interest_options,
                    key=f"tvet_interest_{tvet_sector}_{stream_or_trade}",
                    help=(
                        "The trade-aligned interest is shown first. "
                        "You may select another closely related direction."
                    ),
                )

                career_options = get_tvet_career_options(
                    interest_area,
                    stream_or_trade,
                )
                career_cluster = st.selectbox(
                    "Career Cluster",
                    career_options,
                    key=(
                        f"tvet_career_{tvet_sector}_"
                        f"{stream_or_trade}_{interest_area}"
                    ),
                    help=(
                        "Career options are limited to programs related to "
                        "the selected interest area."
                    ),
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

                result_card(
                    "Recommended Academic Program",
                    recommended_program,
                    (
                        "This program is aligned with the learner’s education "
                        "background, strengths, interests, and career direction."
                    ),
                    "green",
                )
                result_card(
                    "Recommended Bridge Courses",
                    recommended_bridge_course,
                    (
                        "These courses can help the learner strengthen readiness "
                        "for the recommended program."
                    ),
                    "blue",
                )
                result_card(
                    "Alternative Academic Pathway",
                    alternative_pathway,
                    (
                        "This is another related option to explore with an "
                        "academic advisor."
                    ),
                    "purple",
                )
                result_card(
                    "How This Guidance Was Prepared",
                    source,
                    (
                        "The guidance considers the learner’s education background, "
                        "strengths, interests, career direction, and readiness."
                    ),
                    "gold",
                )

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
                result_card("Recommended Bridge Course", "Awaiting recommendation", "These preparatory courses are aligned with the recommended academic program category.", "blue")
                result_card("Explainability Summary", "Awaiting profile", "The explanation will show the model-ranked field and the profile inputs that most influenced it.", "purple")

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