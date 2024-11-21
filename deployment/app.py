import os
import threading
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from model import cartoon_and_style_transfer  # Import your model functions
import streamlit as st

# Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODELS_DIR = "/home/suchetan/Desktop/project/assets/models"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MODELS_DIR"] = MODELS_DIR

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Flask Routes
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Cartoonization and Style Transfer API!"})


@app.route("/process", methods=["POST"])
def process_images():
    try:
        if "content_image" not in request.files or "style_image" not in request.files:
            return jsonify({"error": "Both 'content_image' and 'style_image' are required."}), 400

        content_image = request.files["content_image"]
        style_image = request.files["style_image"]

        content_filename = secure_filename(content_image.filename)
        style_filename = secure_filename(style_image.filename)

        content_path = os.path.join(app.config["UPLOAD_FOLDER"], content_filename)
        style_path = os.path.join(app.config["UPLOAD_FOLDER"], style_filename)

        content_image.save(content_path)
        style_image.save(style_path)

        cartoon_output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"cartoon_{content_filename}")
        style_output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"styled_{content_filename}")

        # Call the model function
        cartoonized_image, stylized_image = cartoon_and_style_transfer(
            original_image_path=content_path,
            style_image_path=style_path,
            cartoon_output_path=cartoon_output_path,
            style_output_path=style_output_path,
            models_dir=app.config["MODELS_DIR"]
        )
        return jsonify({
            "message": "Processing successful!",
            "cartoon_image_url": f"/download/{os.path.basename(cartoonized_image)}",
            "stylized_image_url": f"/download/{os.path.basename(stylized_image)}"
        })
    except Exception as e:
        return jsonify({"error occured while generating cartoonized or stylized image": str(e)}), 500


@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found."}), 404
    return send_file(file_path, as_attachment=True)


# Start Flask in a separate thread
def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False)


# Streamlit UI
def main():
    st.title("Cartoonization and Style Transfer")
    st.write(
        "This app allows you to upload a content image and a style image to generate cartoonized and stylized outputs."
    )

    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png","jpeg"])
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "png","jpeg"])

    if content_file and style_file:
        content_path = os.path.join(UPLOAD_FOLDER, content_file.name)
        style_path = os.path.join(UPLOAD_FOLDER, style_file.name)

        # Save the files
        with open(content_path, "wb") as f:
            f.write(content_file.read())
        with open(style_path, "wb") as f:
            f.write(style_file.read())

        # Output paths
        cartoon_output_path = os.path.join(OUTPUT_FOLDER, f"cartoon_{content_file.name}")
        style_output_path = os.path.join(OUTPUT_FOLDER, f"styled_{content_file.name}")

        try:
            # Process the cartoonized image first
            st.write("Cartoonizing the content image... Please wait.")
            cartoonized_image, _ = cartoon_and_style_transfer(
                original_image_path=content_path,
                style_image_path=style_path,
                cartoon_output_path=cartoon_output_path,
                style_output_path=style_output_path,
                models_dir=MODELS_DIR
            )
            st.write("Cartoonizatiion Completed")
            st.success("Cartoonization completed!")
            st.subheader("Cartoonized Image")
            st.image(cartoon_output_path, use_column_width=True)
            print("Cartoonization done")
            # Process the stylized image
            st.write("Applying style transfer... Please wait.")
            _, stylized_image = cartoon_and_style_transfer(
                original_image_path=content_path,
                style_image_path=style_path,
                cartoon_output_path=cartoon_output_path,  # Skip cartoonization this time
                style_output_path=style_output_path,
                models_dir=MODELS_DIR
            )
            st.success("Style transfer completed!")

            # Display stylized image
            st.subheader("Stylized Image")
            st.image(style_output_path, use_column_width=True)

            # Provide download links
            st.download_button(
                label="Download Cartoonized Image",
                data=open(cartoon_output_path, "rb").read(),
                file_name=f"cartoon_{content_file.name}",
                mime="image/jpeg",
            )
            st.download_button(
                label="Download Stylized Image",
                data=open(style_output_path, "rb").read(),
                file_name=f"styled_{content_file.name}",
                mime="image/jpeg",
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    main()

