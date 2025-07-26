import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import io

# Page configuration
st.set_page_config(
    page_title="Parking Slot Detection & Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .custom-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Upload area styling */
    .stFileUploader > div > div > div {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: 2px dashed #ffffff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(240, 147, 251, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Image containers */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-empty {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .status-occupied {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #2c3e50;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    st.markdown("""
    <div class="custom-header">
        <h1>🚗 Parking Slot Detection Using YOLO & Grad-CAM </h1>
        <p>AI-Powered Smart Parking Management System</p>
    </div>
    """, unsafe_allow_html=True)

def create_metrics(empty_count, occupied_count):
    total_count = empty_count + occupied_count
    occupancy_rate = (occupied_count / total_count * 100) if total_count > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: #27ae60;">{empty_count}</p>
            <p class="metric-label">Empty Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: #e74c3c;">{occupied_count}</p>
            <p class="metric-label">Occupied Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: #3498db;">{total_count}</p>
            <p class="metric-label">Total Slots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: #f39c12;">{occupancy_rate:.1f}%</p>
            <p class="metric-label">Occupancy Rate</p>
        </div>
        """, unsafe_allow_html=True)

def create_pie_chart(empty_count, occupied_count):
    if empty_count + occupied_count == 0:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=['Empty Slots', 'Occupied Slots'],
        values=[empty_count, occupied_count],
        hole=.3,
        marker_colors=['#27ae60', '#e74c3c'],
        textinfo='label+percent',
        textfont_size=14,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'Parking Slot Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        font=dict(size=14),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=20, l=20, r=20),
        height=400
    )
    
    return fig

def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# Load custom CSS
load_css()

# Create header
create_header()

# Sidebar
st.sidebar.markdown("### 🔧 Configuration")
st.sidebar.markdown("---")

# st.sidebar.markdown("#### 🤖 YOLO Model")
# model_file = st.sidebar.file_uploader(
#     "Upload YOLO model (.pt)", 
#     type=["pt"],
#     help="Upload your trained YOLO model file"
# )

st.sidebar.markdown("#### 📸 Input Image")
image_file = st.sidebar.file_uploader(
    "Upload an image", 
    type=["jpg", "jpeg", "png"],
    help="Upload a parking lot image for analysis"
)

st.sidebar.markdown("#### ⚙️ Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05,
    help="Minimum confidence score for detections"
)

show_heatmaps = st.sidebar.checkbox("Show Explainability Heatmaps", value=True)
# show_sam_masks = st.sidebar.checkbox("Show SAM Segmentation", value=False)

# Always use best.pt from directory
MODEL_PATH = "./assets/best.pt"
if 'model' not in st.session_state:
    try:
        st.session_state.model = YOLO(MODEL_PATH)
    except Exception as e:
        st.session_state.model = None
        st.sidebar.error(f"❌ Error loading model: {str(e)}")

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None

# Load model
# if model_file:
#     if st.session_state.model is None:
#         with st.spinner("🔄 Loading YOLO model..."):
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
#                     temp_model_file.write(model_file.read())
#                     model_path = temp_model_file.name
#                 st.session_state.model = YOLO(model_path)
#                 st.sidebar.success("✅ Model loaded successfully!")
#             except Exception as e:
#                 st.sidebar.error(f"❌ Error loading model: {str(e)}")
#                 st.session_state.model = None

# Main content
if image_file and st.session_state.model:
    model = st.session_state.model
    
    # Load and process image
    with st.spinner("🔍 Processing image..."):
        image = Image.open(image_file)
        image_np = np.array(image)
        
        # Run YOLO detection
        results = model(image_np, conf=confidence_threshold)
        
        # Define class names
        vacant_class_names = ["space-empty", "empty", "vacant", "free", "slot-empty", "empty-slot"]
        occupied_class_names = ["space-occupied", "occupied", "car", "vehicle", "slot-occupied"]
        
        # Count detections
        empty_space_count = 0
        occupied_space_count = 0
        
        for result in results:
            for cls in result.boxes.cls:
                class_name = model.names[int(cls)].lower()
                if class_name in vacant_class_names:
                    empty_space_count += 1
                else:
                    occupied_space_count += 1
    
    # Display metrics
    create_metrics(empty_space_count, occupied_space_count)
    
    # Create visualization columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        section_header("📊 Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        section_header("📈 Statistics")
        pie_chart = create_pie_chart(empty_space_count, occupied_space_count)
        if pie_chart:
            st.plotly_chart(pie_chart, use_container_width=True)
    
    # Detection Results Table
    section_header("🔍 Detection Results")
    detection_data = []
    
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            status = "Empty" if class_name.lower() in vacant_class_names else "Occupied"
            detection_data.append({
                "Slot ID": len(detection_data) + 1,
                "Status": status,
                "Class": class_name,
                "Confidence": f"{conf:.3f}",
                "Coordinates": f"({x1}, {y1}, {x2}, {y2})",
                "Area": f"{(x2-x1) * (y2-y1)} px²"
            })
    
    if detection_data:
        df = pd.DataFrame(detection_data)
        st.dataframe(df, use_container_width=True)
        
        # Download button for results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detection Results",
            data=csv,
            file_name=f"parking_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("🔍 No detections found. Try adjusting the confidence threshold.")
    
    # Annotated Image
    section_header("🎯 Annotated Image")
    annotated_image = image_np.copy()
    
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            
            # Color coding: Green for empty, Red for occupied
            color = (0, 255, 0) if class_name.lower() in vacant_class_names else (255, 0, 0)
            
            # Draw rectangle and label
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Add background for text
            label = f"{class_name} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_image, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    st.image(annotated_image, caption="Detection Results", use_column_width=True)
    
    # Explainability Section
    if show_heatmaps:
        section_header("🧠 AI Explainability - Heatmaps & Analysis")
        
        explain_data = []
        
        for i, result in enumerate(results):
            for j, (box, cls, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)].lower()
                
                # Extract crop
                crop = image_np[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Generate saliency map
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                saliency = cv2.Laplacian(gray_crop, cv2.CV_64F)
                heatmap = np.absolute(saliency)
                
                if heatmap.max() > 0:
                    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
                else:
                    heatmap = np.zeros_like(gray_crop)
                
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                density = np.mean(heatmap)
                
                # Generate explanation
                if class_name in vacant_class_names:
                    explanation = "✅ Uniform low-detail region indicates empty space" if density < 20 else "⚠️ High variance may indicate partial obstruction"
                else:
                    explanation = "🚗 High visual complexity indicates vehicle presence" if density > 20 else "🎯 Distinct edges and shapes confirm occupied status"
                
                # Display in columns
                with st.expander(f"🔍 Slot Analysis #{j+1} - {class_name.title()} (Confidence: {conf:.3f})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(crop, caption="Original Crop", use_column_width=True)
                    
                    with col2:
                        st.image(heatmap_color, caption="Saliency Heatmap", use_column_width=True)
                    
                    with col3:
                        st.markdown("**Analysis:**")
                        st.write(explanation)
                        st.metric("Density Score", f"{density:.2f}")
                
                explain_data.append({
                    'Slot_ID': f"#{j+1}",
                    'Status': class_name,
                    'Confidence': round(float(conf), 3),
                    'Density_Score': round(float(density), 2),
                    'Analysis': explanation
                })
        
        if explain_data:
            st.subheader("📊 Analysis Summary")
            st.dataframe(pd.DataFrame(explain_data), use_container_width=True)
    
    # SAM Segmentation (optional)
    # if show_sam_masks:
    #     section_header("🎭 SAM Segmentation Analysis")
    #     st.info("⚠️ SAM segmentation requires the segment-anything model. Install with: pip install segment-anything")
        
    #     # Note: SAM integration would require additional setup
    #     # This is a placeholder for SAM functionality
    #     st.write("SAM segmentation would be implemented here with proper model setup.")

# elif not model_file:
#     st.info("👈 Please upload a YOLO model file (.pt) in the sidebar to get started.")
elif not image_file:
    st.info("👈 Please upload an image file in the sidebar to analyze parking slots.")
else:
    st.info("🔄 Please wait while the model is loading...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p>🚗 Smart Parking Detection System | Built with Streamlit & YOLO</p>
    <p>Analyze parking occupancy with AI-powered computer vision</p>
</div>
""", unsafe_allow_html=True)
