"""
Streamlit web application for colorectal polyp segmentation in clinical use 
@author: ozangokkan
"""
import cv2
import streamlit as st
from PIL import Image
import torchvision.transforms as T
import torch 
import numpy as np
import segmentation_models_pytorch as smp
# import os
import io

best_threshold=0.5 
min_size=5 
image_size=448

def post_process(probability, threshold, min_region_size,size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((size, size), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component > 0
        if p.sum() > min_region_size: # if region of polyp is greater than min_region_size
            predictions[p] = 1
            num += 1
    return predictions, num

    
# #sanity check
# uploaded_file = st.file_uploader("Upload a file")

# if uploaded_file:
#    st.write("Filename: ", uploaded_file.name)




st.title("Application of Polyp Segmentation")
# FRAME_WINDOW = st.image([])
#cam = cv2.VideoCapture(0)# webcam connection
#cam = cv2.VideoCapture(3)# external grabber or usb camera connection
#image read check
# ret, frame = cam.read()
# Blackmagic --ultra studio mini recorder (frame grabber) for video capturing stage using opencv 
# cam = cv2.VideoCapture('decklinksrc mode=7 connection=0 ! videoconvert ! appsink')



img = st.file_uploader("Upload a polyp image for segmentation", type=["png","tif","tiff","jpg", "jpeg"])
img = Image.open(img)

st.image(img)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncoderDecoder(n_class=1).to(device)
model.eval()
state = torch.load("C:/Users/yourpath/model.pth", map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

preprocess = T.Compose([
     T.Resize([448,448]),
     T.ToTensor(),
     T.Normalize(
       mean = [0.485, 0.456, 0.406],
       std = [0.229, 0.224, 0.225]
     )
 ])

x = preprocess(img)
x.shape
alpha = .8
transformed = torch.unsqueeze(torch.tensor(x), 0)
print(transformed.shape)
out = model(transformed)
out = np.squeeze(out)
out = out.detach().numpy()
predict_image, num_predict_ = post_process(out, best_threshold, min_size, image_size)
predicted_mask = predict_image
predicted_mask = Image.fromarray(predicted_mask) 

# img = np.array(img)
# predicted_mask = np.array(predicted_mask)
img = img.resize((448,448))
predicted_mask = predicted_mask.resize((448,448))

predicted_mask = predicted_mask.convert('RGB')


width = predicted_mask.size[0] 
height = predicted_mask.size[1] 
for i in range(0,width):# process all pixels
    for j in range(0,height):
        data = predicted_mask.getpixel((i,j))
        if data[0] > 0 :
            predicted_mask.putpixel((i,j),(0, 128, 0))


overlay = Image.blend(img, predicted_mask, alpha=.4)

overlay = overlay.resize((610,448))


if st.button("Apply"):
    st.image(overlay)  
    st.write('The number of detected polyps :', num_predict_)



# #real-time video streaming, measuring frames per second (FPS) and making polyp segmentation
# cam.set(cv2.CAP_PROP_FPS, 30)
# fps = int(cam.get(5))
# print("fps:", fps)

# if st.button("Apply"):
#     while True:
#         ret, frame = cam.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         prediction = model_prediction(frame, model)
#         st.image(Image.open(prediction))
#         FRAME_WINDOW.image(frame)
#     else:
#         st.write('Stopped')