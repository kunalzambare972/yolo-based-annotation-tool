// Auto-annotation functionality for the Object Detection Annotation System
// This uses the YOLO model to automatically detect objects in images

async function runAutoDetection() {
    const imageName = images[currentIndex];
    
    // Show loading spinner
    showSpinner(true);
    showNotification('Running YOLO detection...', 'success');
    
    try {
        const response = await fetch(`/auto_annotate/${projectName}/${encodeURIComponent(imageName)}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to run auto-detection');
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showNotification('Auto-detection complete!', 'success');
            // Reload annotations to show the new detections
            await loadAnnotations();
        } else {
            showNotification('Auto-detection failed: ' + result.message, 'error');
        }
    } catch (error) {
        console.error('Error running auto-detection:', error);
        showNotification('Error running auto-detection', 'error');
    } finally {
        showSpinner(false);
    }
}

// Add event listener to the auto-annotate button when the document is loaded
document.addEventListener('DOMContentLoaded', function() {
    const autoAnnotateBtn = document.getElementById('auto-annotate');
    if (autoAnnotateBtn) {
        autoAnnotateBtn.addEventListener('click', runAutoDetection);
    }
});