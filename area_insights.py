# --------------------------- FULL APP (combined, Census page updated) ---------------------------
import streamlit as st
from pymongo import MongoClient
import pandas as pd
import numpy as np
import re
import plotly.express as px
from typing import Any, Set, Dict, Optional
import math
import traceback
import json
import ast

# --------------------------- Config & Constants ---------------------------
MONGO_URI = ""
DB_NAME = ""
POSTCODE_COLLECTION = "postcodes_data"
RENTAL_COLLECTION = "rental_data"
SALE_COLLECTION = "sales_data"
CRIME_COLLECTION = "crime_data"

SAMPLE_LIMIT = 6000
VERIFY_PING = False

KNOWN_CENSUS_COLLECTIONS = [
    "census_2021_accommodation_type",
    "census_2021_car_or_van_availability",
    "census_2021_central_heating",
    "census_2021_country_of_birth",
    "census_2021_distance_travelled_to_work",
    "census_2021_highest_level_of_qualification",
    "census_2021_household_size",
    "census_2021_occupancy_rating_rooms",
    "census_2021_occupancy_rating_bedrooms",
    "census_2021_length_of_residence",
    "census_2021_number_of_bedrooms",
    "census_2021_second_address_indicator",
    "census_2021_occupation",
    "census_2021_sex"
]
COLLECTION_DISPLAY_MAP = {
    "census_2021_accommodation_type": ("Accommodation Type","Accommodation type"),
    "census_2021_car_or_van_availability": ("Car or Van Availability", "Number of cars or vans"),
    "census_2021_central_heating": ("Central Heating", "Type of central heating"),
    "census_2021_country_of_birth": ("Country of Birth", "Country of birth"),
    "census_2021_distance_travelled_to_work": ("Distance Travelled to Work", "Distance travelled"),
    "census_2021_highest_level_of_qualification": ("Highest Level of Qualification", "Qualification"),
    "census_2021_household_size": ("Household Size", "Household size"),
    "census_2021_occupancy_rating_rooms": ("Occupancy Rating Rooms", "Rating rooms"),
    "census_2021_occupancy_rating_bedrooms": ("Occupancy Rating Bedrooms", "Rating Bedrooms"),
    "census_2021_length_of_residence": ("Length of Residence", "Length of residence"),
    "census_2021_number_of_bedrooms": ("Number of Bedrooms", "Bedrooms"),
    "census_2021_second_address_indicator": ("Second address indicator", "Second address"),
    "census_2021_occupation": ("Occupation", "Occupation"),
    "census_2021_sex": ("Sex", "Sex")
}
FIELD_NAME_OVERRIDES: Dict[str, Dict[str, str]] = {
    "census_2021_accommodation_type": {
       
        "Accommodation type: Detached":"Detached",
        "Accommodation type: A caravan or other mobile or temporary structure":"caravan",
        "Accommodation type: In a commercial building, for example, in an office building, hotel or over a shop":"Commercial",
        "Accommodation type: In a purpose-built block of flats or tenement":"Tenement",
        "Accommodation type: Part of a converted or shared house, including bedsits":"Shared house",
        "Accommodation type: Part of another converted building, for example, former school, church or warehouse":"Converted building",
        "Accommodation type: Semi-detached": "Semi-detached",
        "Accommodation type: Terraced":"Terraced"
    },
    "census_2021_car_or_van_availability": {
        "Number of cars or vans: 1 car or van in household":"1 car/van",
        "Number of cars or vans: 2 cars or vans in household":"2 cars/vans",
        "Number of cars or vans: 3 or more cars or vans in household":"3 or more cars/vans",
        "Number of cars or vans: No cars or vans in household":"No cars/vans"
    },
    "census_2021_central_heating": {
        "Type of central heating in household: District or communal heat networks only":"Communal heat",
        "Type of central heating in household: Electric only":"Electric",
        "Type of central heating in household: Mains gas only":"Main gas",
        "Type of central heating in household: No central heating":"No central heating",
        "Type of central heating in household: Oil only":"Oil",
        "Type of central heating in household: Other central heating only":"Others",
        "Type of central heating in household: Renewable energy only":"Renewable energy",
        "Type of central heating in household: Solid fuel only":"Solid fuel",
        "Type of central heating in household: Tank or bottled gas only":"Tank gas",
        "Type of central heating in household: Wood only":"Wood"
    },
    "census_2021_country_of_birth": {
        "Country of birth: Africa; measures: Value":"Africa",
        "Country of birth: Antarctica and Oceania (including Australasia) and Other; measures: Value":"Antarctica & Oceania",
        "Country of birth: British Overseas ; measures: Value":"British Overseas",
        "Country of birth: Europe: EU countries: All other EU countries; measures: Value":"EU countries",
        "Country of birth: Europe: Non-EU countries: All other non-EU countries; measures: Value":"Non-EU countries",
        "Country of birth: Europe: United Kingdom; measures: Value":"United Kingdom",
        "Country of birth: Europe: measures: Value":"Europe",
        "Country of birth: Middle East and Asia: measures: Value":"Middle East & Asia",
        "Country of birth: The Americas and the Caribbean: measures: Value":"Americas & Caribbean"
    },
    "census_2021_distance_travelled_to_work": {
        "Distance travelled to work: 10km to less than 20km":"10-20km",
        "Distance travelled to work: 20km to less than 30km":"20-30km",
        "Distance travelled to work: 2km to less than 5km":"2-5km",
        "Distance travelled to work: 30km to less than 40km":"30-40km",
        "Distance travelled to work: 40km to less than 60km":"40-60km",
        "Distance travelled to work: 5km to less than 10km":"5-10km",
        "Distance travelled to work: 60km and over":"60km and above",
        "Distance travelled to work: Less than 2km":"Less than 2km",
        "Distance travelled to work: Works mainly from home":"WFH"
    },
    "census_2021_highest_level_of_qualification": {
        "Highest level of qualification: Apprenticeship":"Apprenticeship",
        "Highest level of qualification: Level 1 and entry level qualifications":"Level 1 qualifications",
        "Highest level of qualification: Level 2 qualifications":"Level 2 qualifications",
        "Highest level of qualification: Level 3 qualifications":"Level 3 qualifications",
        "Highest level of qualification: Level 4 qualifications and above":"Level 4 qualifications & above",
        "Highest level of qualification: No qualifications":"No qualifications",
        "Highest level of qualification: Other qualifications":"Others"
    },
    "census_2021_household_size": {
        "Household size: 0 people in household; measures: Value":"0 people",
        "Household size: 1 person in household; measures: Value":"1 people",
        "Household size: 2 people in household; measures: Value":"2 peoples",
        "Household size: 3 people in household; measures: Value":"3 peoples",
        "Household size: 4 people in household; measures: Value":"4 peoples",
        "Household size: 5 people in household; measures: Value":"5 peoples",
        "Household size: 6 people in household; measures: Value":"6 peoples",
        "Household size: 7 people in household; measures: Value":"7 peoples",
        "Household size: 8 or more people in household; measures: Value": "8 or more peoples"
    },
    "census_2021_occupancy_rating_rooms":{
        "Occupancy rating for rooms: Occupancy rating of rooms: +1":"+1 rating",
        "Occupancy rating for rooms: Occupancy rating of rooms: +2 or more":" +2 or more ratings",
        "Occupancy rating for rooms: Occupancy rating of rooms: -1":" -1 ratings",
        "Occupancy rating for rooms: Occupancy rating of rooms: -2 or less": "-2 or less ratings",
        "Occupancy rating for rooms: Occupancy rating of rooms: 0":"0 ratings"
    },
    "census_2021_occupancy_rating_bedrooms":{
        "Occupancy rating for bedrooms: Occupancy rating of bedrooms: +1":"+1 rating",
        "Occupancy rating for bedrooms: Occupancy rating of bedrooms: +2 or more":" +2 or more ratings",
        "Occupancy rating for bedrooms: Occupancy rating of bedrooms: -1":" -1 ratings",
        "Occupancy rating for bedrooms: Occupancy rating of bedrooms: -2 or less":"-2 or less ratings",
        "Occupancy rating for bedrooms: Occupancy rating of bedrooms: 0":"0 ratings"
    },   
    "census_2021_length_of_residence": {
        "Length of residence in the UK: 10 years or more; measures: Value":"10 years or more",
        "Length of residence in the UK: 2 years or more, but less than 5 years; measures: Value":"2-5 years",
        "Length of residence in the UK: 5 years or more, but less than 10 years; measures: Value":"5-10 years",
        "Length of residence in the UK: Born in the UK; measures: Value":"Born in UK",
        "Length of residence in the UK: Less than 2 years; measures: Value":"2 years less"
    },
    "census_2021_number_of_bedrooms": {
        "Number of bedrooms: 1 bedroom":"1 bedroom",
        "Number of bedrooms: 2 bedrooms":"2 bedrooms",
        "Number of bedrooms: 3 bedrooms":"3 bedrooms",
        "Number of bedrooms: 4 or more bedrooms":"4 or more bedrooms"
    },
    "census_2021_second_address_indicator":{
        "Second address indicator: No second address":"No second address",
        "Second address indicator: Second address is in the UK": "Second address is in the UK",
        "Second address indicator: Second address is outside the UK":"Second address is outside the UK"
    },
    "census_2021_occupation": {
        "Occupation (current): 1 Managers, directors and senior officials":"Managers/directors",
        "Occupation (current): 2 Professional occupations": "Professional",
        "Occupation (current): 3 Associate professional and technical occupations":"Associate professional",
        "Occupation (current): 4 Administrative and secretarial occupations": "Administrative",
        "Occupation (current): 5 Skilled trades occupations": "Skilled trades",
        "Occupation (current): 6 Caring, leisure and other service occupations":"Caring/leisure",
        "Occupation (current): 7 Sales and customer service occupations":"customer service",
        "Occupation (current): 8 Process, plant and machine operatives":"Machine operatives",
        "Occupation (current): 9 Elementary occupations": "Elementary"
    },
    "census_2021_sex": {
        "Sex: Female; measures: Value":"Female",
        "Sex: Male; measures: Value":"Male"
    }
}

house_property_type_filter = [
    "semi detached villa", "semi detached", "semi detached house", "terrace", "terraced", "end terrace", "end terraced",
    "end of terrace", "end of terraced", "end of terrace house", "end of terraced house", "end terrace house",
    "end terraced house", "terraced house", "terrace house", "terraced villa", "detached house", "detached villa",
    "link detached house", "link detached", "detached", "house", "houses", "manor house", "cluster house", "mews",
    "cottage", "smallholding", "equestrian", "leisure", "villa"
]
flat_property_type_filter = [
    "apartment", "block of apartments", "block of flats", "flat", "flat share", "ground flat", "ground floor flat",
    "ground floor studio flat", "ground maisonette", "studio apartment", "studio flat", "serviced apartment",
    "triplex", "studio", "serviced apartments", "serviced studio apartment", "duplex", "maisonette", "penthouse"
]
_house_set = set([s.lower() for s in house_property_type_filter])
_flat_set = set([s.lower() for s in flat_property_type_filter])

# --------------------------- Streamlit Page Config ---------------------------
st.set_page_config(page_title="Area Insights", layout="wide")

@st.cache_resource
def _create_client(uri):
    return MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)

def get_db_safe(uri=MONGO_URI, db_name=DB_NAME, verify_ping=VERIFY_PING):
    try:
        client = _create_client(uri)
        if verify_ping:
            client.admin.command("ping")
        return client[db_name], None
    except Exception as e:
        return None, str(e)

db_obj, db_err = get_db_safe()
if db_obj is None:
    st.error(f"Database connection error: {db_err}")
    st.stop()

# --------------------------- Utility Functions ---------------------------
def normalize_str(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def parse_numeric(val):
    try:
        if val is None:
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        s = str(val).strip()
        s = s.replace(",", "").replace("%", "").strip()
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        if s == "" or re.match(r"^[^\d\.\-]+$", s):
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def remove_prefix_before_colon(s: str) -> str:
    if ":" in s:
        return s.split(":", 1)[1].strip()
    return s.strip()

def shorten_field_name_default(s: str, word_limit=5):
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    words = s.split()
    return " ".join(words[:word_limit])

def find_geography_field(cols):
    cols_lower = [c.lower().strip() for c in cols]
    candidates = [
        "geography code", "geography_code", "oa11_code", "oa11",
        "output_area", "output area", "census output area", "census_output_area",
        "lower layer super output area", "lsoa21 code", "lsoa21", "geography"
    ]
    for cand in candidates:
        if cand in cols_lower:
            return cols[cols_lower.index(cand)]
    for idx, c in enumerate(cols_lower):
        if ("oa" in c and "code" in c) or ("census" in c and "output" in c) or ("lsoa" in c):
            return cols[idx]
    return None

@st.cache_data(ttl=600)
def get_distinct_from_postcodes(field: str, match_filter: Optional[dict] = None):
    coll_postcodes = db_obj[POSTCODE_COLLECTION]
    try:
        results = coll_postcodes.distinct(field, filter=match_filter or {})
        cleaned = [r for r in results if r is not None and str(r).strip() != ""]
        return sorted(cleaned)
    except Exception:
        pipeline = []
        if match_filter:
            pipeline.append({"$match": match_filter})
        pipeline.append({"$group": {"_id": f"${field}"}})
        rows = [doc["_id"] for doc in coll_postcodes.aggregate(pipeline, allowDiskUse=True) if doc["_id"] is not None]
        return sorted(rows)

@st.cache_data(ttl=600)
def load_postcodes(limit=20000):
    coll_postcodes = db_obj[POSTCODE_COLLECTION]
    cur = coll_postcodes.find({}, limit=limit)
    docs = list(cur)
    if not docs:
        return pd.DataFrame()
    return pd.json_normalize(docs)

def build_geo_values_from_postcodes(selected_districts, selected_outcodes, selected_postcodes, postcode_df, 
                                    district_col, postcode_district_col, postcode_col, census_geo_col):
    coll_postcodes = db_obj[POSTCODE_COLLECTION]
    query = {}
    
    if selected_districts and district_col:
        query[district_col] = {"$in": selected_districts}
    
    if selected_outcodes:
        if postcode_district_col and postcode_district_col in postcode_df.columns:
            query[postcode_district_col] = {"$in": selected_outcodes}
        else:
            for fld in ["Postcode district", "postcode_district", "outcode", "Outcode"]:
                if fld in postcode_df.columns:
                    query[fld] = {"$in": selected_outcodes}
                    break
    
    if selected_postcodes and postcode_col and postcode_col in postcode_df.columns:
        query[postcode_col] = {"$in": selected_postcodes}

    possible_geo_fields = []
    if census_geo_col:
        possible_geo_fields.append(census_geo_col)
    possible_geo_fields.extend(["Census output area 2021", "census_output_area", "geography code", 
                                "geography_code", "oa11", "oa11_code", "lsoa21", "geography"])
    
    seen = set()
    possible_geo_fields = [x for x in possible_geo_fields if not (x in seen or seen.add(x))]

    chosen_geo_field = None
    for cand in possible_geo_fields:
        if cand in postcode_df.columns:
            chosen_geo_field = cand
            break

    if not chosen_geo_field:
        for cand in possible_geo_fields:
            doc = coll_postcodes.find_one({cand: {"$exists": True}}, projection={cand: 1})
            if doc:
                chosen_geo_field = cand
                break

    if not chosen_geo_field:
        chosen_geo_field = possible_geo_fields[0]

    geo_set = set()
    cursor = None
    try:
        cursor = coll_postcodes.find(query, projection={chosen_geo_field: 1}, no_cursor_timeout=True)
        for doc in cursor:
            v = doc.get(chosen_geo_field)
            if v is None:
                for k in doc.keys():
                    kl = k.lower()
                    if kl in ("census output area 2021", "census_output_area", "geography code", 
                             "geography_code", "oa11", "oa11_code", "lsoa21", "geography"):
                        v = doc.get(k)
                        break
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    if item is not None:
                        geo_set.add(str(item).strip())
            else:
                geo_set.add(str(v).strip())
    except Exception:
        pass
    finally:
        try:
            if cursor is not None:
                cursor.close()
        except Exception:
            pass

    return set(x.strip().lower() for x in geo_set if x and str(x).strip() != "")

def compute_total_counts_preserve_order_stream(coll_name: str, geo_set: Set[str], overrides_map: Dict[str, str]):
    coll = db_obj[coll_name]
    geo_set_normalized = set([g.strip().lower() for g in (geo_set or set())])

    sample_doc = coll.find_one()
    coll_cols = list(sample_doc.keys()) if sample_doc else []
    geo_field_in_coll = find_geography_field(coll_cols) if coll_cols else None

    cursor = None
    totals: Dict[str, float] = {}
    ordered_fields = []
    first_doc = None

    try:
        cursor = coll.find({}, no_cursor_timeout=True)
        for doc in cursor:
            doc_geo_val = None
            if geo_field_in_coll and geo_field_in_coll in doc:
                doc_geo_val = doc.get(geo_field_in_coll)
            else:
                for k in doc.keys():
                    kl = k.lower().strip()
                    if kl in ("geography code", "geography_code", "census output area 2021", 
                             "census_output_area", "oa11", "oa11_code", "lsoa21", "geography"):
                        doc_geo_val = doc.get(k)
                        break
            if doc_geo_val is None:
                for k, v in doc.items():
                    if isinstance(v, (list, tuple)) and v:
                        for item in v:
                            if item is not None and str(item).strip().lower() in geo_set_normalized:
                                doc_geo_val = item
                                break
                        if doc_geo_val:
                            break

            if geo_set_normalized:
                if doc_geo_val is None or str(doc_geo_val).strip().lower() not in geo_set_normalized:
                    continue

            if first_doc is None:
                first_doc = doc
                geo_key_candidates = [k for k in first_doc.keys() if "geograph" in k.lower() or "oa" in k.lower() 
                                     or "output" in k.lower() or "lsoa" in k.lower()]
                exclude_keys = set(["_id"] + geo_key_candidates)
                if overrides_map:
                    ordered_fields = [k for k in overrides_map.keys() if k not in exclude_keys]
                else:
                    ordered_fields = [k for k in first_doc.keys() if k not in exclude_keys]
                for fld in ordered_fields:
                    totals[fld] = 0.0

            if not ordered_fields:
                continue

            for fld in ordered_fields:
                if fld not in doc:
                    continue
                v = doc.get(fld)
                n = parse_numeric(v)
                if not np.isnan(n):
                    totals[fld] = totals.get(fld, 0.0) + float(n)

    finally:
        try:
            if cursor is not None:
                cursor.close()
        except Exception:
            pass

    rows = []
    for fld in ordered_fields:
        raw_total = float(totals.get(fld, 0.0))
        rounded_total = int(round(raw_total))
        rows.append({"field_name": fld, "total_count": rounded_total})
    return pd.DataFrame(rows)

def df_to_styled_html_compact(df: pd.DataFrame, col_label: str, val_label: str, scroll=False) -> str:
    css = f"""
    <style>
      .table-container {{
        max-height: none !important;
        overflow-y: visible !important;
        padding:12px;
        margin-bottom:18px;
        background:#fff;
      }}
      .ai-table {{
        border-collapse:collapse;
        width:100%;
        font-size:15px;
        table-layout: fixed;
      }}
      .ai-table th {{
        border:1px solid #dcdcdc;
        text-align:left;
        padding:10px 12px;
        background:#f6f8fa;
        font-size:15px;
        color: #000000;
        font-weight:700;
      }}
      .ai-table td {{
        border:1px solid #efefef;
        text-align:left;
        padding:10px 12px;
        font-size:14px;
        color: #111111;
        font-weight:400;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .ai-table td:first-child {{
        font-size:13px;
        font-weight:400;
      }}
      .ai-table td, .ai-table th {{
        line-height:1.4;
      }}
    </style>
    """
    html = f"<div class='table-container'><table class='ai-table'><thead><tr><th style='text-align:left'>{col_label}</th><th style='text-align:left'>{val_label}</th></tr></thead><tbody>"
    for _, row in df.iterrows():
        val = row.iloc[1]
        if pd.isna(val):
            val_str = ""
        else:
            try:
                val_str = f"{int(val):,d}"
            except Exception:
                val_str = str(val)
        html += f"<tr><td style='font-weight:400'>{row.iloc[0]}</td><td style='text-align:left'>{val_str}</td></tr>"
    html += "</tbody></table></div>"
    return css + html

@st.cache_data(ttl=600)
def cached_compute_totals(coll_name: str, geo_values_serializable, overrides_keys_tuple):
    geo_set = set(geo_values_serializable) if geo_values_serializable else set()
    overrides_map = FIELD_NAME_OVERRIDES.get(coll_name, {}) or {}
    return compute_total_counts_preserve_order_stream(coll_name, geo_set, overrides_map)

def _round_up_to_nice(x: float) -> int:
    if x <= 0:
        return 0
    exp = math.floor(math.log10(x))
    base = 10 ** exp
    candidates = [1, 2, 2.5, 5, 10]
    for c in candidates:
        if base * c >= x:
            return int(math.ceil(base * c))
    return int(math.ceil(base * 10))

def compute_y_ticks(max_val: int, desired_ticks=5):
    if max_val <= 0:
        return [0, 1]
    top = _round_up_to_nice(max_val)
    raw_ticks = [0]
    for i in range(1, desired_ticks):
        frac = i / (desired_ticks - 1)
        raw_ticks.append(int(round(top * frac)))
    ticks = sorted(set(raw_ticks))
    if ticks[-1] != top:
        ticks[-1] = top
    final_ticks = []
    prev = -1
    for t in ticks:
        if t <= prev:
            t = prev + max(1, int(round(top * 0.05)))
        final_ticks.append(t)
        prev = t
    return final_ticks

# --------------------------- Sales/Rental helpers ---------------------------
def normalize_text(x):
    try: 
        return str(x).strip().lower()
    except: 
        return ""

def normalize_property_type(x):
    s = normalize_text(x)
    if not s: 
        return "unknown"
    if "flat" in s: 
        return "flat"
    if "apartment" in s: 
        return "apartment"
    if "terrace" in s or "terraced" in s or "townhouse" in s: 
        return "terraced"
    if "detached" in s and "semi" not in s: 
        return "detached"
    if "semi" in s: 
        return "semi detached"
    if "bungalow" in s or "bunglow" in s: 
        return "bungalow"
    if "studio" in s: 
        return "studio"
    return "other"

def extract_bedrooms(val):
    if pd.isna(val): 
        return np.nan
    if isinstance(val, (int, float)):
        try:
            return int(val)
        except:
            return np.nan
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else np.nan

def build_mongo_filter(postcodes, outcodes, districts):
    clauses = []
    if postcodes:
        clauses.append({
            "$expr": {
                "$in": [
                    {"$toUpper": {"$trim": {"input": "$postcode"}}},
                    [x.strip().upper() for x in postcodes]
                ]
            }
        })
    if outcodes:
        clauses.append({
            "$expr": {
                "$in": [
                    {"$toUpper": {"$trim": {"input": "$outcode"}}},
                    [x.strip().upper() for x in outcodes]
                ]
            }
        })
    if districts:
        clauses.append({
            "$expr": {
                "$in": [
                    {"$toUpper": {"$trim": {"input": "$district"}}},
                    [x.strip().upper() for x in districts]
                ]
            }
        })
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def load_collection_sample(collection_name, mongo_filter=None, limit=SAMPLE_LIMIT):
    coll = db_obj[collection_name]
    try:
        cursor = coll.find(mongo_filter if mongo_filter else {}).limit(limit)
        df = pd.json_normalize(list(cursor))
    except Exception as e:
        raise RuntimeError(f"Error reading {collection_name}: {e}")
    return df

def metrics_by_property_and_bedrooms(df, price_field="price"):
    if df.empty: 
        return pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    df["price"] = pd.to_numeric(df[price_field], errors="coerce") if price_field in df.columns else np.nan
    df["_ptype_raw"] = df["type"].astype(str) if "type" in df.columns else "unknown"
    df["_ptype"] = df["_ptype_raw"].map(normalize_property_type)
    bedroom_cols = [c for c in df.columns if "bed" in c.lower()]
    if bedroom_cols:
        df["_bedrooms"] = df[bedroom_cols[0]].apply(extract_bedrooms)
    else:
        df["_bedrooms"] = np.nan
    canonical = ["flat","apartment","terraced","detached","semi detached","bungalow","studio","other","unknown"]
    agg_type = df.groupby("_ptype").agg(
        count=("price","count"),
        avg_price=("price","mean"),
        median_price=("price","median"),
        min_price=("price","min"),
        max_price=("price","max")
    ).reset_index().rename(columns={"_ptype":"property_type"})
    agg_type["sort_order"] = agg_type["property_type"].apply(lambda x: canonical.index(x) if x in canonical else len(canonical))
    agg_type = agg_type.sort_values(["sort_order","count"], ascending=[True,False]).drop(columns=["sort_order"])
    df_bed = df.dropna(subset=["_bedrooms"])
    if not df_bed.empty:
        df_bed["_bedrooms_int"] = df_bed["_bedrooms"].astype(int)
        agg_bed = df_bed.groupby("_bedrooms_int").agg(
            count=("price","count"),
            avg_price=("price","mean"),
            median_price=("price","median"),
            min_price=("price","min"),
            max_price=("price","max")
        ).reset_index().rename(columns={"_bedrooms_int":"bedrooms"}).sort_values("bedrooms")
    else:
        agg_bed = pd.DataFrame(columns=["bedrooms","count","avg_price","median_price","min_price","max_price"])
    for col in ["avg_price","median_price","min_price","max_price"]:
        if col in agg_type.columns:
            agg_type[col] = agg_type[col].round(2)
        if col in agg_bed.columns:
            agg_bed[col] = agg_bed[col].round(2)
    return agg_type, agg_bed

def metrics_top_agents(df, agent_col="agent_name", price_field="price", top_n=5, by="count"):
    if df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    if agent_col not in df2.columns:
        df2[agent_col] = np.nan
    df2[agent_col] = df2[agent_col].fillna("Unknown Agent").astype(str).str.strip()
    df2["_price_num"] = pd.to_numeric(df2[price_field], errors="coerce") if price_field in df2.columns else np.nan
    agg = df2.groupby(agent_col).agg(
        listings_count=("_price_num","count"),
        avg_price=("_price_num","mean"),
        median_price=("_price_num","median"),
        min_price=("_price_num","min"),
        max_price=("_price_num","max")
    ).reset_index().rename(columns={agent_col:"agent_name"})
    for col in ["avg_price","median_price","min_price","max_price"]:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0)
    sort_key = "listings_count" if by == "count" else "avg_price"
    agg_sorted = agg.sort_values(sort_key, ascending=False).head(top_n).reset_index(drop=True)
    for col in ["avg_price","median_price","min_price","max_price"]:
        if col in agg_sorted.columns:
            agg_sorted[col] = agg_sorted[col].round(2)
    return agg_sorted

def majority_property_type(df, field="type", top_n_display=10):
    if df.empty:
        return None, pd.DataFrame(columns=["property_type_mapped", "count"])
    if "type" not in df.columns:
        return None, pd.DataFrame(columns=["property_type_mapped", "count"])

    def _map_type(raw):
        s = normalize_text(raw)
        if not s:
            return "unknown"
        if s in _house_set:
            return "house"
        if s in _flat_set:
            return "flat"
        for h in _house_set:
            if h in s:
                return "house"
        for f in _flat_set:
            if f in s:
                return "flat"
        return normalize_property_type(s)

    mapped = df["type"].fillna("Unknown").astype(str).apply(_map_type)
    counts = mapped.value_counts(dropna=False).rename_axis("property_type_mapped").reset_index(name="count")
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)
    majority_value = counts.iloc[0]["property_type_mapped"] if not counts.empty else None
    counts_display = counts.head(top_n_display).copy()
    return majority_value, counts_display

def metrics_tax_band(df, tax_field="tax_band", top_n_display=20):
    if df.empty or tax_field not in df.columns:
        return pd.DataFrame(columns=["tax_band", "count"])
    tb = df[tax_field].fillna("Unknown").astype(str).str.strip()
    counts = tb.value_counts(dropna=False).rename_axis("tax_band").reset_index(name="count")
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)
    return counts.head(top_n_display).copy()

def _is_truthy_company_owned(val):
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("true", "yes", "y", "1", "t"):
        return True
    try:
        if float(s) == 1.0:
            return True
    except:
        pass
    return False

def compute_company_owned_stats(df, company_field="company_owned", group_by=None):
    if df is None or df.empty or company_field not in df.columns:
        return 0, 0, 0.0, pd.DataFrame()

    df2 = df.copy()
    df2["_company_owned_bool"] = df2[company_field].apply(_is_truthy_company_owned).astype(bool)

    overall_total = len(df2)
    overall_company_owned = int(df2["_company_owned_bool"].sum())
    overall_pct = round((overall_company_owned / overall_total * 100.0) if overall_total > 0 else 0.0, 2)

    grouped_df = pd.DataFrame()
    if group_by and group_by in df2.columns:
        grouped_total = df2.groupby(group_by).size().reset_index(name="total_count")
        grouped_company = df2.groupby(group_by)["_company_owned_bool"].sum().reset_index(name="company_owned_count")
        grouped = pd.merge(grouped_total, grouped_company, on=group_by, how="left")
        grouped["company_owned_count"] = grouped["company_owned_count"].fillna(0).astype(int)
        grouped["company_owned_pct"] = grouped.apply(
            lambda r: round((r["company_owned_count"] / r["total_count"] * 100.0) if r["total_count"] > 0 else 0.0, 2),
            axis=1
        )
        grouped_df = grouped.sort_values("total_count", ascending=False).reset_index(drop=True)

    return overall_total, overall_company_owned, overall_pct, grouped_df

def metrics_number_of_floors(df, floors_field="number_of_floors", top_n_display=30):
    if df.empty or floors_field not in df.columns:
        return pd.DataFrame(columns=["number_of_floors", "count"])
    def _extract_floor(x):
        if pd.isna(x):
            return "Unknown"
        if isinstance(x, (int, float)):
            try:
                return int(x)
            except:
                return "Unknown"
        s = str(x).strip()
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
        return s or "Unknown"
    floors = df[floors_field].apply(_extract_floor)
    counts = floors.value_counts(dropna=False).rename_axis("number_of_floors").reset_index(name="count")
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)
    return counts.head(top_n_display).copy()

# --------------------------- Price history helpers ---------------------------
def _parse_price_history(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        prices = []
        for item in raw:
            if isinstance(item, (int, float)):
                prices.append(float(item))
            elif isinstance(item, dict):
                for k in ("price","amount","value"):
                    if k in item and item[k] is not None:
                        try:
                            prices.append(float(item[k]))
                            break
                        except:
                            continue
            else:
                try:
                    val = float(str(item).replace(",","").strip())
                    prices.append(val)
                except:
                    continue
        return prices
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return _parse_price_history(parsed)
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            return _parse_price_history(parsed)
        except Exception:
            pass
        parts = [p.strip() for p in s.split(",") if p.strip()]
        prices = []
        for p in parts:
            p_clean = re.sub(r"[^\d\.\-]", "", p)
            if not p_clean:
                continue
            try:
                prices.append(float(p_clean))
            except:
                continue
        return prices
    try:
        return [float(raw)]
    except:
        return []

def compute_price_drop_levels(df, history_field="price_history", price_field="price"):
    if df.empty:
        return df.copy()
    df2 = df.copy()
    df2["_current_price"] = pd.to_numeric(df2[price_field], errors="coerce") if price_field in df2.columns else np.nan
    hist_prices_list = []
    hist_max = []
    for raw in df2.get(history_field, pd.Series([None]*len(df2))):
        prices = _parse_price_history(raw)
        hist_prices_list.append(prices)
        if prices:
            try:
                mx = max([p for p in prices if (p is not None and not np.isnan(p))])
            except:
                mx = np.nan
        else:
            mx = np.nan
        hist_max.append(mx)
    df2["_hist_prices"] = hist_prices_list
    df2["hist_max_price"] = pd.to_numeric(pd.Series(hist_max), errors="coerce")
    def _pct_drop(row):
        hp = row.get("hist_max_price", np.nan)
        cur = row.get("_current_price", np.nan)
        if pd.isna(hp) or hp == 0 or pd.isna(cur):
            return np.nan
        try:
            return float((hp - cur) / hp * 100.0)
        except:
            return np.nan
    df2["pct_drop"] = df2.apply(_pct_drop, axis=1)
    def _level(pct):
        if pd.isna(pct):
            return "Unknown / insufficient data"
        try:
            if pct < 1.0:
                return "No drop (<1%)"
            if 1.0 <= pct < 5.0:
                return "Small (1–5%)"
            if 5.0 <= pct < 15.0:
                return "Moderate (5–15%)"
            if 15.0 <= pct < 30.0:
                return "Large (15–30%)"
            if pct >= 30.0:
                return "Severe (>=30%)"
            return "Unknown / insufficient data"
        except:
            return "Unknown / insufficient data"
    df2["price_drop_level"] = df2["pct_drop"].apply(_level)
    df2["pct_drop"] = df2["pct_drop"].round(2)
    return df2

def _render_metric_pies(agg_df, name_col, metrics, title_prefix):
    if agg_df.empty:
        st.warning(f"No data to show for {title_prefix}.")
        return
    charts_per_row = 3
    pies = []
    for metric in metrics:
        if metric not in agg_df.columns:
            continue
        vals = agg_df[metric].fillna(0)
        if vals.sum() == 0:
            pies.append((metric, None))
            continue
        fig = px.pie(agg_df, names=name_col, values=metric, 
                    title=f"{title_prefix} — {metric.replace('_',' ').title()}")
        pies.append((metric, fig))
    i = 0
    while i < len(pies):
        col1, col2, col3 = st.columns(charts_per_row)
        cols = [col1, col2, col3]
        for col_idx in range(charts_per_row):
            if i >= len(pies):
                break
            metric, fig = pies[i]
            with cols[col_idx]:
                if fig is None:
                    st.info(f"No meaningful values for **{metric.replace('_',' ').title()}**.")
                else:
                    st.plotly_chart(fig, use_container_width=True)
            i += 1

def query_lsoa_codes_from_postcodes(postcodes, outcodes, districts, 
                                   lsoa_field_name_candidates=("LSOA Code","LSOA code","lsoa_code")):
    coll = db_obj[POSTCODE_COLLECTION]
    query_clauses = []
    if postcodes:
        query_clauses.append({"Postcode": {"$in": [p.strip() for p in postcodes]}})
    if outcodes:
        query_clauses.append({"Postcode district": {"$in": [o.strip() for o in outcodes]}})
    if districts:
        query_clauses.append({"District": {"$in": [d.strip() for d in districts]}})
    final_query = {"$and": query_clauses} if query_clauses else {}
    projection = {k: 1 for k in lsoa_field_name_candidates}
    try:
        cursor = coll.find(final_query, projection=projection).limit(10000)
    except Exception as e:
        raise RuntimeError(f"Error querying {POSTCODE_COLLECTION} for LSOA codes: {e}")
    lsoas = set()
    for doc in cursor:
        for cname in lsoa_field_name_candidates:
            if cname in doc and doc[cname]:
                lsoas.add(str(doc[cname]).strip())
    return sorted(list(lsoas))

def load_crime_by_lsoas(lsoa_list, lsoa_field_candidates=("LSOA Code","LSOA code","lsoa_code"), limit=SAMPLE_LIMIT):
    coll = db_obj[CRIME_COLLECTION]
    if not lsoa_list:
        return pd.DataFrame()
    or_clauses = []
    for f in lsoa_field_candidates:
        or_clauses.append({f: {"$in": lsoa_list}})
    query = {"$or": or_clauses} if len(or_clauses) > 1 else or_clauses[0]
    try:
        cursor = coll.find(query).limit(limit)
        df = pd.json_normalize(list(cursor))
    except Exception as e:
        raise RuntimeError(f"Error reading {CRIME_COLLECTION}: {e}")
    return df

def compute_crime_levels(crime_df, crime_type_field="Crime type", 
                        lsoa_field_candidates=("LSOA Code","LSOA code","lsoa_code"), 
                        thresholds=(10,50,200)):
    if crime_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if crime_type_field in crime_df.columns:
        crime_type_counts = crime_df[crime_type_field].fillna("Unknown").astype(str).str.strip().value_counts().rename_axis(crime_type_field).reset_index(name="count")
    else:
        crime_type_counts = pd.DataFrame(columns=[crime_type_field,"count"])

    lsoa_col = None
    for c in lsoa_field_candidates:
        if c in crime_df.columns:
            lsoa_col = c
            break
    if lsoa_col is None:
        for c in crime_df.columns:
            if c.lower() == "lsoa code".lower() or c.lower() == "lsoa_code":
                lsoa_col = c
                break

    if lsoa_col is None:
        lsoa_counts_df = pd.DataFrame(columns=["lsoa","count","crime_level"])
        return crime_type_counts, lsoa_counts_df

    lsoa_counts = crime_df[lsoa_col].fillna("Unknown").astype(str).str.strip().value_counts().rename_axis("lsoa").reset_index(name="count")
    low_th, med_th, high_th = thresholds
    def _crime_level(cnt):
        try:
            cnt = int(cnt)
        except:
            return "Unknown"
        if cnt <= low_th:
            return "Low"
        if low_th < cnt <= med_th:
            return "Moderate"
        if med_th < cnt <= high_th:
            return "High"
        return "Severe"
    lsoa_counts["crime_level"] = lsoa_counts["count"].apply(_crime_level)
    lsoa_counts_df = lsoa_counts.sort_values("count", ascending=False).reset_index(drop=True)
    return crime_type_counts, lsoa_counts_df

@st.cache_data(ttl=600)
def load_postcode_distincts(max_distinct=5000, batch_size=1000):
    coll = db_obj[POSTCODE_COLLECTION]
    postcodes_set, outcodes_set, districts_set = set(), set(), set()
    projection = {"Postcode": 1, "Postcode district": 1, "District": 1}
    try:
        cursor = coll.find({}, projection=projection, batch_size=batch_size)
        for doc in cursor:
            if "Postcode" in doc and doc["Postcode"]: 
                postcodes_set.add(str(doc["Postcode"]).strip())
            if "Postcode district" in doc and doc["Postcode district"]: 
                outcodes_set.add(str(doc["Postcode district"]).strip())
            if "District" in doc and doc["District"]: 
                districts_set.add(str(doc["District"]).strip())
            if len(postcodes_set) >= max_distinct and len(outcodes_set) >= max_distinct and len(districts_set) >= max_distinct:
                break
    except Exception as e:
        return {"postcodes": [], "outcodes": [], "districts": [], "error": str(e)}

    def _limit_sort(s):
        lst = sorted(list(s))
        return lst[:max_distinct] if len(lst) > max_distinct else lst

    return {
        "postcodes": _limit_sort(postcodes_set),
        "outcodes": _limit_sort(outcodes_set),
        "districts": _limit_sort(districts_set),
        "error": None
    }

# --------------------------- UI Start ---------------------------
st.title("Area Insights")

# Load filter options
dist_res = load_postcode_distincts()
if dist_res["error"]:
    st.error("Error loading filters: " + str(dist_res["error"]))
    st.stop()

# Load postcode dataframe for census
postcode_df = load_postcodes()
postcode_columns_lower = [c.lower() for c in postcode_df.columns]

def pick_column(possible_names):
    for cand in possible_names:
        for col in postcode_df.columns:
            if col.lower().strip() == cand.lower().strip():
                return col
    return None

district_col = pick_column(["district", "district name", "local_authority", "local authority", "localauthority"])
postcode_col = pick_column(["postcode", "postcode_full", "postcode_full_clean", "postcode_full_cleaned", "postcode_full_cleaned_ups"])
postcode_district_col = pick_column(["postcode district", "postcode_district", "outcode", "outcode_clean", "outcode_standard"])
census_geo_col = pick_column(["census output area 2021", "census_output_area", "geography code", "geography_code", "oa11", "oa11_code", "lsoa21"])

# Filters section
st.markdown("---")
st.subheader("Filters")

col1, col2, col3 = st.columns(3)

# Initialize session state for filters if not exists
if 'selected_districts' not in st.session_state:
    st.session_state.selected_districts = []
if 'selected_outcodes' not in st.session_state:
    st.session_state.selected_outcodes = []
if 'selected_postcodes' not in st.session_state:
    st.session_state.selected_postcodes = []
if 'filter_applied' not in st.session_state:
    st.session_state.filter_applied = False

with col1:
    district_options = dist_res["districts"]
    selected_districts = st.multiselect("District", options=district_options, default=st.session_state.selected_districts, key="district_filter")
    st.session_state.selected_districts = selected_districts

with col2:
    outcode_options = []
    if selected_districts:
        try:
            match = {}
            if district_col:
                match[district_col] = {"$in": selected_districts}
            outcode_field = postcode_district_col or "Postcode district"
            if outcode_field:
                vals = get_distinct_from_postcodes(outcode_field, match_filter=match)
                if vals:
                    outcode_options = vals
        except Exception:
            pass
    else:
        outcode_options = dist_res["outcodes"]
    
    selected_outcodes = st.multiselect("Outcode", options=outcode_options, default=st.session_state.selected_outcodes, key="outcode_filter")
    st.session_state.selected_outcodes = selected_outcodes

# Postcode filter (cascades based on district and outcode, multi-select)
with col3:
    postcode_options = []
    if postcode_col:
        try:
            match = {}
            if selected_districts and district_col:
                match[district_col] = {"$in": selected_districts}
            if selected_outcodes:
                outcode_field = postcode_district_col or "Postcode district"
                if outcode_field:
                    match[outcode_field] = {"$in": selected_outcodes}
            if match or (not selected_districts and not selected_outcodes):
                postcodes_list = get_distinct_from_postcodes(postcode_col, match_filter=match if match else None)
                if postcodes_list:
                    postcode_options = postcodes_list
        except Exception:
            pass
    
    selected_postcodes = st.multiselect("Postcode", options=postcode_options, default=st.session_state.selected_postcodes, key="postcode_filter")
    st.session_state.selected_postcodes = selected_postcodes

st.markdown("---")
page = st.radio("Select option", ["Sales, Rental, Crime", "Census"], horizontal=True)

st.markdown("---")

if st.button("Apply Filter", type="primary"):
    st.session_state.filter_applied = True
    st.rerun()

if st.session_state.filter_applied:
    try:
        # Convert selections for mongo queries
        postcode_sel = selected_postcodes
        outcode_sel = selected_outcodes
        district_sel = selected_districts
        
        # Build mongo filter for sales/rental/crime
        mongo_filter = build_mongo_filter(postcode_sel, outcode_sel, district_sel)
        
        # Build geo values for census
        geo_values = build_geo_values_from_postcodes(
            selected_districts, selected_outcodes, selected_postcodes, 
            postcode_df, district_col, postcode_district_col, postcode_col, census_geo_col
        )
        
        if page == "Sales, Rental, Crime":
            # ========== SALES SECTION ==========
            st.header(" Sales Metrics")
            sale_df = load_collection_sample(SALE_COLLECTION, mongo_filter)
            st.write(f"**Rows fetched:** {len(sale_df)}")
            
            if not sale_df.empty:
                # By property type
                s_type, s_bed = metrics_by_property_and_bedrooms(sale_df)
                st.subheader("By Property Type")
                _render_metric_pies(s_type, name_col="property_type",
                                   metrics=["count","avg_price","median_price","min_price","max_price"],
                                   title_prefix="Sales — Property Type")
                
                # By bedrooms
                st.subheader("By Bedrooms")
                _render_metric_pies(s_bed, name_col="bedrooms",
                                   metrics=["count","avg_price","median_price","min_price","max_price"],
                                   title_prefix="Sales — Bedrooms")
                
                # Majority property type
                st.subheader("Majority Property Type")
                maj_val, maj_counts_df = majority_property_type(sale_df, field="type", top_n_display=15)
                if maj_val is None:
                    st.info("Column `type` not present in the sales dataset or no values found.")
                else:
                    if not maj_counts_df.empty:
                        left, right = st.columns([1, 1])
                        with left:
                            st.dataframe(maj_counts_df)
                        with right:
                            fig = px.bar(maj_counts_df, x="property_type_mapped", y="count", text="count")
                            fig.update_layout(xaxis_title="Mapped Property Type", yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                
                # Price drop levels
                st.subheader("Price Drop Levels")
                sale_with_drop = compute_price_drop_levels(sale_df, history_field="price_history", price_field="price")
                dist = sale_with_drop["price_drop_level"].value_counts(dropna=False).rename_axis("price_drop_level").reset_index(name="count")
                dist = dist.sort_values("count", ascending=False).reset_index(drop=True)
                if dist.empty:
                    st.info("No price_history data available or unable to compute price drops.")
                else:
                    left, right = st.columns([1, 1])
                    with left:
                        st.dataframe(dist)
                    with right:
                        fig = px.bar(dist, x="price_drop_level", y="count", text="count")
                        fig.update_layout(xaxis_title="Price Drop Level", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Top agents
                st.subheader("Top Agents")
                c1, c2 = st.columns([1, 1])
                with c1:
                    top_n = st.number_input("Top N agents", min_value=1, max_value=20, value=5, step=1, key="top_n_agents")
                with c2:
                    agent_sort_by = st.selectbox("Rank agents by", options=["Listings (count)", "Average price"], 
                                                index=0, key="agent_sort_by")
                by_key = "count" if agent_sort_by == "Listings (count)" else "avg_price"
                agents_df = metrics_top_agents(sale_df, agent_col="agent_name", price_field="price", 
                                              top_n=int(top_n), by=by_key)
                if agents_df.empty:
                    st.info("No agent data available in the filtered sales dataset.")
                else:
                    left, right = st.columns([1, 1])
                    with left:
                        st.dataframe(agents_df)
                    with right:
                        if by_key == "count":
                            fig = px.bar(agents_df, x="agent_name", y="listings_count", text="listings_count")
                        else:
                            fig = px.bar(agents_df, x="agent_name", y="avg_price", text="avg_price")
                        fig.update_layout(xaxis_title="Agent", 
                                        yaxis_title=("Listings" if by_key=="count" else "Average Price"))
                        st.plotly_chart(fig, use_container_width=True)
                
                # Tax band
                st.subheader("Tax Band Metrics")
                tax_counts = metrics_tax_band(sale_df, tax_field="tax_band", top_n_display=50)
                if tax_counts.empty:
                    st.info("No `tax_band` field present or no values found in the sales dataset.")
                else:
                    left, right = st.columns([1, 1])
                    with left:
                        st.dataframe(tax_counts)
                    with right:
                        fig = px.bar(tax_counts, x="tax_band", y="count", text="count")
                        fig.update_layout(xaxis_title="Tax Band", yaxis_title="Count", xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Number of floors
                st.subheader("Number of Floors Metrics")
                floors_counts = metrics_number_of_floors(sale_df, floors_field="number_of_floors", top_n_display=50)
                if floors_counts.empty:
                    st.info("No `number_of_floors` field present or no values found in the sales dataset.")
                else:
                    left, right = st.columns([1, 1])
                    with left:
                        st.dataframe(floors_counts)
                    with right:
                        fig = px.bar(floors_counts, x="number_of_floors", y="count", text="count")
                        fig.update_layout(xaxis_title="Number of Floors", yaxis_title="Count", xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Company owned
                st.subheader("Company-Owned Metrics")
                overall_total, overall_company_owned, overall_pct, grouped_company_df = compute_company_owned_stats(
                    sale_df, company_field="company_owned", group_by="type"
                )
                if overall_total == 0:
                    st.info("No `company_owned` field present or sales dataset is empty.")
                else:
                    comp_df = pd.DataFrame({
                        "status": ["Company owned", "Not company owned"],
                        "count": [overall_company_owned, overall_total - overall_company_owned]
                    })
                    left, right = st.columns([1, 1])
                    if not grouped_company_df.empty:
                        with left:
                            st.dataframe(grouped_company_df.head(50))
                    with right:
                        fig = px.pie(comp_df, names="status", values="count", title="Company Owned vs Not")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No sale data fetched for the selected filters.")
            
            st.markdown("---")
            
            # ========== RENTAL SECTION ==========
            st.header("Rental Metrics")
            rental_df = load_collection_sample(RENTAL_COLLECTION, mongo_filter)
            st.write(f"**Rows fetched:** {len(rental_df)}")
            
            if not rental_df.empty:
                r_type, r_bed = metrics_by_property_and_bedrooms(rental_df)
                st.subheader("By Property Type")
                _render_metric_pies(r_type, name_col="property_type",
                                   metrics=["count","avg_price","median_price","min_price","max_price"],
                                   title_prefix="Rental — Property Type")
                st.subheader("By Bedrooms")
                _render_metric_pies(r_bed, name_col="bedrooms",
                                   metrics=["count","avg_price","median_price","min_price","max_price"],
                                   title_prefix="Rental — Bedrooms")
            else:
                st.warning("No rental data fetched for the selected filters.")
            
            st.markdown("---")
            
            # ========== CRIME SECTION ==========
            st.header("Crime Metrics", info="Crime data is mapped via LSOA codes from the postcode filters.")
            
            try:
                lsoa_codes = query_lsoa_codes_from_postcodes(postcode_sel, outcode_sel, district_sel)
                
                if not lsoa_codes:
                    st.info("No LSOA codes found for selected filters.")
                else:
                    st.subheader("Crime Level Thresholds (Adjustable)")
                    th1_col, th2_col, th3_col = st.columns(3)
                    with th1_col:
                        low_to_mod = st.number_input("Low → Moderate (<=)", min_value=0, value=10, step=1, 
                                                    key="crime_th1")
                    with th2_col:
                        mod_to_high = st.number_input("Moderate → High (<=)", min_value=1, value=50, step=1, 
                                                     key="crime_th2")
                    with th3_col:
                        high_to_severe = st.number_input("High → Severe (<=)", min_value=1, value=200, step=1, 
                                                        key="crime_th3")
                    
                    crime_df = load_crime_by_lsoas(lsoa_codes)
                    st.write(f"**Crime rows fetched:** {len(crime_df)}")
                    
                    if crime_df.empty:
                        st.info("No crime rows found for the matched LSOA codes.")
                    else:
                        crime_type_counts, lsoa_counts_df = compute_crime_levels(
                            crime_df, crime_type_field="Crime type", 
                            thresholds=(low_to_mod, mod_to_high, high_to_severe)
                        )
                        
                        st.subheader("Top Crime Types")
                        if crime_type_counts.empty:
                            st.info("`Crime type` field not present in crime data.")
                        else:
                            top_n_ct = st.number_input("Top N crime types", min_value=1, max_value=50, 
                                                      value=10, step=1, key="top_n_crime_types")
                            ct_show = crime_type_counts.head(int(top_n_ct)).copy()
                            left, right = st.columns([1, 1])
                            with left:
                                st.dataframe(ct_show)
                            with right:
                                fig = px.bar(ct_show, x="Crime type", y="count", title="Top Crime Types", text="count")
                                fig.update_layout(xaxis_title="Crime Type", yaxis_title="Count")
                                st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error computing crime metrics: {str(e)}")
                st.exception(traceback.format_exc())
        
        elif page == "Census":
            # ===========================
            # UPDATED CENSUS PAGE (all sections: table left, chart right)
            # ===========================
            st.header("Census Metrics")
            if not geo_values:
                st.info("No census geography values found for the selected postcodes.")
            # Use the exact confirmed 14 collections
            census_collections = [
                "census_2021_accommodation_type",
                "census_2021_car_or_van_availability",
                "census_2021_central_heating",
                "census_2021_country_of_birth",
                "census_2021_distance_travelled_to_work",
                "census_2021_highest_level_of_qualification",
                "census_2021_household_size",
                "census_2021_occupancy_rating_rooms",
                "census_2021_occupancy_rating_bedrooms",
                "census_2021_length_of_residence",
                "census_2021_number_of_bedrooms",
                "census_2021_second_address_indicator",
                "census_2021_occupation",
                "census_2021_sex"
            ]

            # For each collection, compute totals and present left table + right chart
            for coll_name in census_collections:
                display_title, field_label = COLLECTION_DISPLAY_MAP.get(coll_name, (coll_name, "field_name"))
                # Section header
                st.markdown(f"<div style='font-size:22px;font-weight:800;margin-top:18px;margin-bottom:10px'>{display_title}</div>", 
                           unsafe_allow_html=True)
                with st.spinner(f"Computing totals for {display_title}..."):
                    overrides_keys_tuple = tuple(FIELD_NAME_OVERRIDES.get(coll_name, {}).keys()) if FIELD_NAME_OVERRIDES.get(coll_name) else tuple()
                    # compute totals using existing caching layer (uses compute_total_counts_preserve_order_stream)
                    df_totals = cached_compute_totals(coll_name, tuple(sorted(list(geo_values))), overrides_keys_tuple)
                if df_totals.empty:
                    st.info("No numeric/total data found in this collection for the selected filter.")
                    st.markdown("---")
                    continue

                overrides_map = FIELD_NAME_OVERRIDES.get(coll_name, {}) or {}
                def map_to_short(fld):
                    short = overrides_map.get(fld)
                    if short:
                        return short
                    return shorten_field_name_default(remove_prefix_before_colon(str(fld)), word_limit=5)

                # Apply short labels
                df_totals["short_label"] = df_totals["field_name"].apply(map_to_short)
                df_totals["total_count"] = df_totals["total_count"].apply(lambda x: int(round(x)) if not pd.isna(x) else 0)

                display_df = df_totals[["short_label", "total_count"]].rename(
                    columns={"short_label": field_label, "total_count": "Total count"}
                )

                # Left-right columns for table and chart
                left_col, right_col = st.columns([1, 1], gap="medium")
                with left_col:
                    # Render the compact HTML table (keeps your existing styling)
                    html = df_to_styled_html_compact(display_df, field_label, "Total count", scroll=False)
                    st.markdown(html, unsafe_allow_html=True)

                with right_col:
                    # Prepare chart data - top N categories for readability
                    chart_df = display_df.copy()
                    # ensure numeric
                    chart_df["Total Count"] = pd.to_numeric(
                        chart_df["Total count"].astype(str).str.replace(",", ""), errors="coerce"
                    ).fillna(0).astype(int)
                    chart_df_ordered = chart_df.reset_index(drop=True)
                    chart_df_for_plot = chart_df_ordered.head(15).copy()

                    if not chart_df_for_plot.empty and chart_df_for_plot["Total Count"].sum() > 0:
                        # choose pie for sex, otherwise bar
                        if coll_name == "census_2021_sex":
                            fig = px.pie(
                                chart_df_for_plot,
                                names=field_label,
                                values="Total Count",
                                hole=0.4
                            )
                            fig.update_traces(textinfo="percent+label", textfont=dict(color="black", size=13))
                            fig.update_layout(
                                height=380,
                                margin=dict(l=20, r=20, t=40, b=20),
                                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                            )
                            fig.update_layout(legend=dict(font=dict(size=12, color="#000000")))
                            st.plotly_chart(fig, use_container_width=True, height=380)
                        else:
                            max_val = int(chart_df_for_plot["Total Count"].max())
                            y_ticks = compute_y_ticks(max_val, desired_ticks=5)
                            fig = px.bar(
                                chart_df_for_plot,
                                x=field_label,
                                y="Total Count",
                                orientation='v',
                                text="Total Count"
                            )
                            fig.update_traces(
                                texttemplate='%{text:,d}',
                                textposition='outside',
                                textfont=dict(color="black", size=12),
                                marker_line_width=0,
                                width=0.5
                            )
                            fig.update_yaxes(range=[0, chart_df_for_plot["Total Count"].max() * 1.25])
                            fig.update_layout(margin=dict(t=80, b=140, l=80, r=20), showlegend=False, height=420)
                            fig.update_xaxes(
                                title=dict(text=field_label, font=dict(size=18, color="#000000"), standoff=20),
                                tickfont=dict(color="black", size=12),
                                automargin=True
                            )
                            fig.update_yaxes(
                                title=dict(text="Total Count", font=dict(size=18, color="#000000"), standoff=20),
                                tickfont=dict(color="black", size=12),
                                tickmode='array',
                                tickvals=y_ticks,
                                ticktext=[f"{t:,d}" for t in y_ticks],
                                automargin=True
                            )
                            st.plotly_chart(fig, use_container_width=True, height=420)
                    else:
                        st.info("No meaningful numeric totals to plot for this dataset.")

                st.markdown("---")

        st.success(" Metrics loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(traceback.format_exc())
else:
    st.info("   ")
