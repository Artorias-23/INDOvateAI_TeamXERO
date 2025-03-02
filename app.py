from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from werkzeug.utils import secure_filename
import os
from PIL import Image
from geopy.geocoders import Nominatim
import piexif
from geopy.distance import geodesic
from supabase import create_client, Client

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])
SUPABASE_URL = "https://grxnfcefkpxdtkzrkqdg.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdyeG5mY2Vma3B4ZHRrenJrcWRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ5NTYxMjUsImV4cCI6MjA1MDUzMjEyNX0.YNpxvWx7413tDbmlX6Xz91LcNvkyR2d9_xVq3Tc6Gkk"  # Replace with your API key

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cpu")
#from pyngrok import ngrok
import os


UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper: Convert DMS (degrees, minutes, seconds) to decimal coordinates
def dms_to_decimal(dms, ref):
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if ref in [b'S', b'W']:
        decimal *= -1
    return decimal
@app.route("/verify_image/", methods=["POST"])
def verify_image():
    if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
    program_name = request.form.get("program_name")

    # Save the uploaded image file locally
    image_file = request.files['image']
    filename = image_file.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    pil=Image.open(image_path)
    image_file.save(image_path)

    # Extract EXIF data using piexif

    exif_data = piexif.load(image_path)

    gps_data = exif_data.get("GPS", {})

    image_lat = dms_to_decimal(gps_data[2], gps_data[1])
    image_lon = dms_to_decimal(gps_data[4], gps_data[3])


    geolocator = Nominatim(user_agent="my_unique_application")
    location_obj = geolocator.reverse((image_lat, image_lon), language="en")
    image_location_address = location_obj.address if location_obj else "Address not found"
    print("image location:",image_location_address)

    supabase_response = supabase.table("incidents").select("location").eq("program_name", program_name).execute()

    print("Supabase Response:", supabase_response)

    record = supabase_response.data[0]
    location_str = record.get("location", "")
    print("Location:",location_str)

    coords_part = location_str.split("|")[0].strip()  # Extract "latitude:longitude"
    coords_part1 = location_str.split("|")[1].strip()
    print("coords :",coords_part1)  
    lat_str, lon_str = coords_part.split(",")
    table_lat = float(lat_str)
    table_lon = float(lon_str)
    '''   image = request.files["image"]
        program_name = request.form.get("program_name")'''

    distance_km = geodesic((table_lat, table_lon), (image_lat, image_lon)).km
    is_authentic = distance_km <= 20

    inputs = caption_processor(pil, "A image consists of:", return_tensors="pt").to("cpu")
    caption = caption_processor.decode(caption_model.generate(**inputs)[0], skip_special_tokens=True)

    # Store verification result in Supabase
    verification_status = "verified" if is_authentic else "non-verified"

    # Update the verification column in Supabase
    update_response = supabase.table("incidents").update({"verification": verification_status}).eq("program_name", program_name).execute()
    print("Supabase Update Response:", update_response)

    result={
         "isAuthentic": is_authentic,
          "caption": caption,
          "image_address":image_location_address
    }
    if not is_authentic:
         result["user_address"]=coords_part1

    '''if(is_authentic):
        result = {"isAuthentic": True, "message": "Location verified successfully"}

    else:
        result = {"isAuthentic": False, "message": "Location mismatch"}'''

    return jsonify(result), 200
        
if __name__ == "__main__":
    app.run(debug=True,port=5000)