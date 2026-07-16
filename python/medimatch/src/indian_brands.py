"""
indian_brands.py — US Generic → Indian Brand Name Mapping
==========================================================
Maps commonly used US generic drug names to their popular Indian
brand name equivalents for display in the MediMatch UI.
"""

import re
from typing import List

# ---------------------------------------------------------------------------
# Mapping: US generic (lowercase) → popular Indian brand names
# ---------------------------------------------------------------------------
INDIAN_BRANDS_MAP = {
    "acetaminophen": ["Dolo-650", "Crocin", "Calpol"],
    "acyclovir": ["Acivir", "Zovirax"],
    "adapalene": ["Adaferin", "Deriva"],
    "albuterol": ["Asthalin", "Ventorlin"],
    "alprazolam": ["Alprax", "Restyl"],
    "amikacin": ["Mikacin", "Amicin"],
    "amlodipine": ["Amlong", "Amlokind"],
    "amoxicillin": ["Novamox", "Moxikind"],
    "anastrozole": ["Altraz", "Armotraz"],
    "aripiprazole": ["Arip MT", "Asprito"],
    "aspirin": ["Ecosprin", "Disprin"],
    "atorvastatin": ["Atorva", "Lipikind"],
    "azithromycin": ["Azithral", "Azee"],
    "brimonidine": ["Alphagan", "Brimocom"],
    "budesonide": ["Budecort", "Foracort"],
    "carbamazepine": ["Tegretol", "Mazetol"],
    "cefixime": ["Zifi", "Taxim-O"],
    "ceftriaxone": ["Monocef", "Oframax"],
    "cetirizine": ["Cetzine", "Okacet"],
    "ciprofloxacin": ["Cifran", "Ciplox"],
    "citalopram": ["Cilentra", "Celapram"],
    "clindamycin": ["Dalacin", "Clindac A"],
    "clonazepam": ["Clonotril", "Lonazep"],
    "clopidogrel": ["Clavix", "Plavix"],
    "dapagliflozin": ["Forxiga", "Oxra"],
    "dexamethasone": ["Dexona", "Decdan"],
    "diclofenac": ["Voveran", "Reactin"],
    "doxycycline": ["Doxicip", "Tetradox"],
    "duloxetine": ["Duvanta", "Symbal"],
    "dutasteride": ["Dutas", "Veltride"],
    "empagliflozin": ["Jardiance", "Gibtulio"],
    "enalapril": ["Envas", "Nuril"],
    "enoxaparin": ["Clexane", "Lonopin"],
    "escitalopram": ["Nexito", "Cipralex"],
    "esomeprazole": ["Nexpro", "Sompraz"],
    "fentanyl": ["Durogesic", "Fendrop"],
    "finasteride": ["Finpecia", "Finast"],
    "fluconazole": ["Zocon", "Forcan"],
    "fluoxetine": ["Fludac", "Prodep"],
    "fluticasone": ["Flomist", "Flohale"],
    "furosemide": ["Lasix", "Frusenex"],
    "gabapentin": ["Gabapin", "Neurontin"],
    "glimepiride": ["Glimy", "Amaryl"],
    "glipizide": ["Glynase", "Glucotrol"],
    "hydrochlorothiazide": ["Aquazide", "Hydride"],
    "hydroxychloroquine": ["HCQS", "Plaquenil"],
    "ibuprofen": ["Brufen", "Combiflam"],
    "insulin aspart": ["Novorapid", "Novolog"],
    "insulin glargine": ["Basalog", "Lantus"],
    "ipratropium": ["Ipravent", "Duolin"],
    "isotretinoin": ["Sotret", "Acutret"],
    "ketoconazole": ["Nizral", "Ketomac"],
    "lansoprazole": ["Lanzol", "Lanzap"],
    "latanoprost": ["Latoprost", "Xalatan"],
    "letrozole": ["Letroz", "Femara"],
    "levetiracetam": ["Levipil", "Keppra"],
    "levocetirizine": ["Levocet", "Teczine"],
    "levothyroxine": ["Thyronorm", "Eltroxin"],
    "liraglutide": ["Victoza", "Saxenda"],
    "lisinopril": ["Listril", "Lipril"],
    "lithium": ["Lithosun", "Intalith"],
    "losartan": ["Losacar", "Repace"],
    "metformin": ["Glycomet", "Gluconorm"],
    "methotrexate": ["Folitrax", "Neotrexate"],
    "methylprednisolone": ["Medrol", "Zempred"],
    "metoprolol": ["Metolar", "Starpress"],
    "metronidazole": ["Flagyl", "Metrogyl"],
    "minoxidil": ["Mintop", "Tugain"],
    "montelukast": ["Montair", "Telekast"],
    "morphine": ["Morphine Sulphate", "Morcontin"],
    "mupirocin": ["T-Bact", "Supirocin"],
    "olanzapine": ["Oleanz", "Olan"],
    "omeprazole": ["Omez", "Ocid"],
    "ondansetron": ["Emeset", "Zofran"],
    "pantoprazole": ["Pan-D", "Pantocid"],
    "paroxetine": ["Pari", "Paxidep"],
    "phenytoin": ["Eptoin", "Dilantin"],
    "pioglitazone": ["Pioz", "Piomed"],
    "prednisolone": ["Wysolone", "Omnacortil"],
    "pregabalin": ["Pregalin", "Pregeb"],
    "quetiapine": ["Qutan", "Seroquel"],
    "rabeprazole": ["Rablet", "Rabicip"],
    "ramipril": ["Cardace", "Ramistar"],
    "ranitidine": ["Aciloc", "Rantac"],
    "risperidone": ["Sizodon", "Risperdal"],
    "rosuvastatin": ["Rosuvas", "Rozavel"],
    "sertraline": ["Daxid", "Sertima"],
    "sildenafil": ["Manforce", "Silagra"],
    "simvastatin": ["Simvotin", "Zocor"],
    "sitagliptin": ["Januvia", "Istamet"],
    "spironolactone": ["Aldactone", "Spiroton"],
    "sucralfate": ["Sucrafil", "Sulcrate"],
    "tadalafil": ["Megalis", "Tadalis"],
    "tamoxifen": ["Cytotam", "Nolvadex"],
    "tamsulosin": ["Veltam", "Urimax"],
    "telmisartan": ["Telma", "Telmikind"],
    "terbinafine": ["Terbinaforce", "Sebifin"],
    "theophylline": ["Deriphyllin", "Theo Asthalin"],
    "timolol": ["Glucomol", "Timolet"],
    "topiramate": ["Topamac", "Epitop"],
    "tramadol": ["Ultracet", "Contramal"],
    "tretinoin": ["Retino-A", "A-Ret"],
    "valproic acid": ["Valparin", "Encorate"],
    "valsartan": ["Valzaar", "Diovan"],
    "venlafaxine": ["Venlor", "Veniz"],
    "warfarin": ["Warf", "Uniwarfin"],
}


def get_indian_names(drug_name: str) -> List[str]:
    """
    Given a drug name (generic or combination like 'Acetaminophen / hydrocodone'),
    returns a flat list of popular Indian brand name equivalents.

    Uses case-insensitive partial matching so it works on combination drugs too.

    Args:
        drug_name: The drug name string from the dataset.

    Returns:
        A list of Indian brand name strings (may be empty if none found).
    """
    found_brands: List[str] = []

    # Normalise and split combination drugs on "/" or ","
    components = [c.strip().lower() for c in re.split(r"[/,]", drug_name)]

    for component in components:
        for generic, brands in INDIAN_BRANDS_MAP.items():
            if generic in component:
                for brand in brands:
                    if brand not in found_brands:
                        found_brands.append(brand)

    return found_brands
