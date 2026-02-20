import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠")
st.title("🧠 Brain Tumor Classification Web App")
st.write("อัปโหลดภาพ MRI สมอง เพื่อวิเคราะห์และจำแนกประเภทเนื้องอก 4 คลาส ด้วยโมเดล MobileNetV3")

# 2. ฟังก์ชันโหลดโมเดล MobileNetV3
@st.cache_resource
def load_model():
    # โครงสร้างโมเดล MobileNetV3 Large
    model = models.mobilenet_v3_large(weights=None)
    # ปรับ Layer สุดท้ายให้ส่งออก 4 คลาส (glioma, meningioma, notumor, pituitary)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 4)
    
    # โหลด Weights จากไฟล์ที่คุณอัปโหลดมา (ใช้ CPU สำหรับเว็บฟรี)
    checkpoint = torch.load('mobilenetv3_checkpoint_fold2.pt', map_location=torch.device('cpu'), weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

model = load_model()

# 3. กำหนดชื่อคลาส
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 4. ขั้นตอนการแปลงภาพ (Transform)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. UI ส่วนอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ MRI สมอง...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงรูปภาพ
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='รูปภาพที่อัปโหลด', use_column_width=True)
    st.write("กำลังประมวลผล...")
    
    # แปลงภาพเข้าโมเดล
    img_tensor = transform(image).unsqueeze(0)
    
    # ทำนายผล
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    # แสดงผลลัพธ์
    predicted_class = class_names[predicted.item()]
    st.success(f"**ผลการวิเคราะห์:** {predicted_class}")
    st.info(f"**ความมั่นใจ (Confidence):** {confidence.item() * 100:.2f}%")
