# from app import app
from flask import request, render_template, Flask
from PIL import Image
from skimage.metrics import structural_similarity
import imutils
import cv2
import os

app = Flask(__name__)

# app.config['INITIAL_FILE_UPLOADS'] = "app/static/uploads"
# app.config['EXISTING_FILE'] = "app/static/original"
# app.config['GENERATED_FILE'] = "app/static/generated"

INITIAL_FILE_UPLOADS = "./static/uploads"
EXISTING_FILE = "./static/original"
GENERATED_FILE = "./static/generated"

# Route to home page
@app.route('/', methods=['GET','POST'])
def index():
    #Execute if request is get
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # Get uploaded images
        file_upload = request.files['file_upload']
        # file_name = file_upload.filename

        # Resize and save the uploaded images
        uploaded_image = Image.open(file_upload).resize((250,160))
        # uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))
        uploaded_image.save(os.path.join(INITIAL_FILE_UPLOADS, 'image.png'))

        # Resize and save the original image to be sure the uploaded and original matches the size
        # original_image = Image.open(os.path.join(app.config['EXISTING_FILE'], 'image.jpg')).resize(250,160)
        original_image = Image.open(os.path.join(EXISTING_FILE, 'original.png')).resize((250,160))
        # original_image.save(os.path.join(app.config['EXISTING_FILE'], 'image.png'))
        original_image.save(os.path.join(EXISTING_FILE, 'resized.png'))

        # Read the images in computer vision library
        # original_image = cv2.imread(os.path.join(app.config['EXISTING_FILE'], 'image.png'))
        original_image = cv2.imread(os.path.join(EXISTING_FILE, 'resized.png'))
        # uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))
        uploaded_image = cv2.imread(os.path.join(INITIAL_FILE_UPLOADS, 'image.png'))

        # Convert image into grayscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

        # Find the structural_similarity_index(SSIM)
        (score, diff) = structural_similarity(original_gray,uploaded_gray, full=True)
        diff = (diff*255).astype('uint8')

        #Find the Threshold and Contours
        thres = cv2.threshold(diff,0,255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(original_image, (x,y), (x+w, y+h), (0,255,0), 3)
            cv2.rectangle(uploaded_image, (x,y), (x+w, y+h), (0,255,0), 3)
    
        # Save all output Images
        # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'],'image_original'), original_image)
        # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'],'image_uploaded'), uploaded_image)
        # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'],'image_diff'), diff)
        # cv2.imwrite(os.path.join(app.config['GENERATED_FILE'],'image_threshold'), thres)
        cv2.imwrite(os.path.join(GENERATED_FILE,'image_original.png'), original_image)
        cv2.imwrite(os.path.join(GENERATED_FILE,'image_uploaded.png'), uploaded_image)
        cv2.imwrite(os.path.join(GENERATED_FILE,'image_diff.png'), diff)
        cv2.imwrite(os.path.join(GENERATED_FILE,'image_threshold.png'), thres)
        return render_template('index.html', pred = str(round(score*100,2))+'%'+'correct')

if __name__ == "__main__":
    app.run(debug = True)