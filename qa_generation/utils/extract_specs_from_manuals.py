import pdfplumber
import re
import os
import argparse
import json

# Output file path
OUTPUT_FILE = "utils/operating_ranges_from_manuals.py"

def extract_specs_from_pdf(pdf_path, robot_model):
    """
    Extracts specs from a PDF manual for a specific robot.
    Search for tables containing keywords like "motion", "range", "torque", "speed".
    """
    specs = {
        "joint_limits": {}, # min, max (deg)
        "max_speed": {},    # deg/s
        "max_torque": {},   # Nm (if available)
        "temp_range": {"min": 0, "max": 45} # Default, look for environment specs
    }
    
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text: continue
                
                # Look for Motion Range / Working Range
                # formatting in these manuals can be tricky.
                # Heuristic: look for lines with "Axis" and numbers
                
                # Example regex for Axis 1 +180 -180
                # This is a very simplified extractor. Real manuals require robust parsing.
                # We will check for common patterns.
                
                lines = text.split('\n')
                for line in lines:
                    # Check for Axis limits
                    # "Axis 1 +180 to -180" or similar
                    match = re.search(r"(Axis|Joint)\s+(\d+).+?([+-]?\d+\.?\d*)\s*(deg|°).+?([+-]?\d+\.?\d*)\s*(deg|°)", line, re.IGNORECASE)
                    if match:
                        axis_num = int(match.group(2))
                        val1 = float(match.group(3))
                        val2 = float(match.group(5))
                        specs["joint_limits"][f"joint_{axis_num}"] = {
                            "min": min(val1, val2), 
                            "max": max(val1, val2)
                        }
                        
                    # Check for Max Speed
                    # "Axis 1 150 deg/s"
                    match_speed = re.search(r"(Axis|Joint)\s+(\d+).+?(\d+\.?\d*)\s*(deg/s|°/s)", line, re.IGNORECASE)
                    if match_speed:
                        axis_num = int(match_speed.group(2))
                        val = float(match_speed.group(3))
                        specs["max_speed"][f"joint_{axis_num}"] = val
                        
                    # Check for temperature
                    # "Ambient temperature +5 to +45"
                    match_temp = re.search(r"Ambient temperature.+?([+-]?\d+).+?([+-]?\d+)", line, re.IGNORECASE)
                    if match_temp:
                         specs["temp_range"]["min"] = float(match_temp.group(1))
                         specs["temp_range"]["max"] = float(match_temp.group(2))

        # Fill missing joints if partial extraction
        # For IRB 2600, usually 6 axes
        for i in range(1, 7):
            j = f"joint_{i}"
            if j not in specs["joint_limits"]:
                specs["joint_limits"][j] = {"min": -180, "max": 180} # Fallback
            if j not in specs["max_speed"]:
                specs["max_speed"][j] = 175 # Fallback typical speed
                
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
        
    return specs

def generate_python_file(all_specs):
    content = "# Auto-generated operating ranges from manuals\n\n"
    content += "OPERATING_RANGES = {\n"
    
    for robot, specs in all_specs.items():
        content += f"    '{robot}':,\n"
        content += json.dumps(specs, indent=8).replace("true", "True").replace("false", "False")
        content += ",\n"
        
    content += "}\n"
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.replace(":,", ":") # fix json dump trailing comma issue if any (not needed really)
        f.write(content)
        
    print(f"✓ Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuals-dir", default="manuals")
    args = parser.parse_args()
    
    results = {}
    
    if os.path.exists(args.manuals_dir):
        for root, dirs, files in os.walk(args.manuals_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    # Guess robot model from folder or filename
                    # User structure: manuals/ABB_IRB2600/...
                    robot_name = "Unknown"
                    if "IRB" in root or "IRB" in file:
                        robot_name = "ABB_IRB2600"
                    
                    if robot_name != "Unknown":
                        results[robot_name] = extract_specs_from_pdf(os.path.join(root, file), robot_name)
    
    generate_python_file(results)
