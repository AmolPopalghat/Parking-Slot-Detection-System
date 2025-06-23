# 🚗 Parking Slot Detection & Analysis

A modern, AI-powered parking management system built with Streamlit and YOLO for real-time parking slot detection and analysis.

## ✨ Features

- **Smart Detection**: YOLO-based parking slot detection with customizable confidence thresholds
- **Real-time Analysis**: Instant parking occupancy statistics and visualizations
- **AI Explainability**: Saliency heatmaps and detailed explanations for each detection
- **Interactive Dashboard**: Modern, responsive UI with dynamic charts and metrics
- **Export Functionality**: Download detection results as CSV files
- **Flexible Input**: Support for various image formats and YOLO model files

## 🎯 Key Capabilities

### Detection & Analysis
- Automatic detection of empty and occupied parking slots
- Confidence-based filtering for accurate results
- Detailed bounding box coordinates and area calculations
- Real-time occupancy rate calculation

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
   cd parking-detection-app
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
   Navigate to `http://localhost:8501`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t parking-detection-app .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 parking-detection-app
   ```

## 📁 Project Structure

```
parking-detection-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md             # Project documentation
└── assets/               # Sample images and models (optional)
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
- Real-time metrics updating
- Animated loading states
- Interactive charts with Plotly
- Expandable analysis sections

## 📊 Detection Classes

The app automatically recognizes these parking slot classes:

**Empty Slots:**
- `space-empty`
- `empty`
- `vacant`
- `free`
- `slot-empty`
- `empty-slot`

**Occupied Slots:**
- `space-occupied`
- `occupied`
- `car`
- `vehicle`
- `slot-occupied`

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

## 🌐 Deployment Options

### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### AWS/GCP/Azure
Use the provided Dockerfile for container deployment

## 🛠️ Customization

### Adding New Features
1. Modify `app.py` for new functionality
2. Update `requirements.txt` for new dependencies
3. Adjust CSS in the `load_css()` function

### Styling Changes
- Modify the CSS in `load_css()` function
- Update colors in `.streamlit/config.toml`
- Customize chart themes in Plotly configurations

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure YOLO model is compatible with ultralytics version
   - Check file format (.pt extension)
   - Verify model classes match expected names

2. **Memory Issues**
   - Reduce image size before processing
   - Adjust confidence threshold to reduce detections
   - Use Docker with increased memory limits

3. **Performance Optimization**
   - Enable GPU support for faster inference
   - Use smaller YOLO models for real-time processing
   - Implement caching for repeated operations

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