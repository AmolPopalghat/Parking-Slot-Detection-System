# 🚗 Parking Slot Detection & Analysis

A modern, AI-powered parking management system built with Streamlit and YOLO for parking slot detection and analysis.

## ✨ Features

- **Smart Detection**: YOLO-based parking slot detection with customizable confidence thresholds
- **Instant Analysis**: Instant parking occupancy statistics and visualizations
- **AI Explainability**: Saliency heatmaps and detailed explanations for each detection
- **Interactive Dashboard**: Modern, responsive UI with dynamic charts and metrics
- **Export Functionality**: Download detection results as CSV files
- **Flexible Input**: Support for various image formats and YOLO model files

## 🎯 Key Capabilities

### Detection & Analysis
- Automatic detection of empty and occupied parking slots
- Confidence-based filtering for accurate results
- Detailed bounding box coordinates and area calculations

### Visualization
- Interactive pie charts for parking distribution
- Annotated images with color-coded detections
- Saliency heatmaps for AI decision explanation
- Responsive metrics dashboard

### User Experience
- Drag-and-drop file uploads
- Customizable detection parameters
- Downloadable results
- Mobile-friendly responsive design

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Parking-Slot-Detection-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to
   ```bash
   http://localhost:8501
   ```

## 📁 Project Structure

```
Parking-Slot-Detection-System
├── app.py
├── requirements.txt
├── assets/
│   └── best.pt
```

## 🔧 Configuration

### Streamlit Settings
The app uses custom configuration in `.streamlit/config.toml`:
- Upload size limit: 200MB
- Custom theme colors
- Performance optimizations

### Model Requirements
- **Format**: PyTorch (.pt) YOLO models
- **Classes**: Should include parking slot classes (empty/occupied variations)
- **Recommended**: YOLOv8 or newer for best performance

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Maximum size: 200MB

## 🎨 UI Features

### Modern Design
- Gradient backgrounds and smooth animations
- Responsive layout for all screen sizes
- Professional color scheme
- Interactive hover effects

### Dynamic Elements
- Animated loading states
- Interactive charts with Plotly
- Expandable analysis sections

## 📊 Detection Classes

The app automatically recognizes these parking slot classes:

**Empty Slots:**
- `space-empty`

**Occupied Slots:**
- `space-occupied`

## 🔍 Explainability Features

### Saliency Heatmaps
- Visual explanation of AI decision-making
- Laplacian-based edge detection
- Color-coded intensity maps
- Density score calculations

### Analysis Explanations
- Automated reasoning for each detection
- Visual complexity assessment
- Confidence correlation analysis


## 🛠️ Customization

### Adding New Features
1. Modify `app.py` for new functionality
2. Update `requirements.txt` for new dependencies
3. Adjust CSS in the `load_css()` function

### Styling Changes
- Modify the CSS in `load_css()` function
- Customize chart themes in Plotly configurations


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review Streamlit documentation

## 🚀 Future Enhancements

- [ ] Real-time video processing
- [ ] Database integration for historical data
- [ ] Advanced analytics dashboard
- [ ] Multi-camera support
- [ ] API endpoints for external integration
- [ ] Mobile app companion

---

**Built with ❤️ using Streamlit, YOLO, and modern web technologies**
