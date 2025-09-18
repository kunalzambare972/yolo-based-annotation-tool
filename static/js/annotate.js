// Configuration
let projectName = "";
let images = [];
let currentIndex = 0;
let canvas = null;
let currentImage = null;
let isDrawing = false;
let isShiftPressed = false;
let currentRect = null;
let startX, startY;
let totalAnnotations = 0;
let imageStatus = {}; // Track annotation status per image
let currentViewMode = 'original'; // Default view mode
let originalAnnotations = []; // Track original annotations for comparison
let deletedAnnotations = []; // Track deleted original annotations

// Initialize Fabric.js canvas
function initCanvas() {
    canvas = new fabric.Canvas('canvas', {
        selection: true,
        selectionColor: 'rgba(46, 204, 113, 0.3)',
        selectionBorderColor: '#2ecc71',
        selectionLineWidth: 2,
        backgroundColor: '#2c3e50'
    });

    // Update project info display
    document.getElementById('project-name-display').textContent = projectName;
    document.getElementById('image-count').textContent = images.length;
    
    // Handle shift key for drawing mode
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Shift') {
            isShiftPressed = true;
            document.getElementById('shift-warning').style.display = 'block';
        }
        
        // Save on S
        if (e.key === 's' || e.key === 'S') {
            e.preventDefault();
            saveAnnotations();
        }
        
        // Arrow key navigation
        if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
            e.preventDefault();
            prevImage();
        }
        
        if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
            e.preventDefault();
            nextImage();
        }
    });

    document.addEventListener('keyup', function(e) {
        if (e.key === 'Shift') {
            isShiftPressed = false;
            document.getElementById('shift-warning').style.display = 'none';
        }
    });

    // Drawing rectangle ONLY with left mouse button + shift
    canvas.on('mouse:down', function(options) {
        // Only allow drawing if Shift is pressed
        if (!(options.e.button === 0 && isShiftPressed)) {
            return;
        }
        isDrawing = true;
        startX = options.pointer.x;
        startY = options.pointer.y;

        // Create a new rectangle with green stroke for new annotations
        currentRect = new fabric.Rect({
            left: startX,
            top: startY,
            width: 0,
            height: 0,
            fill: 'transparent',
            stroke: '#2ecc71', // Green for new annotations
            strokeWidth: 2,
            strokeUniform: true,
            selectable: true,
            hasControls: true,
            hasBorders: true,
            isOriginal: false // Mark as non-original
        });

        canvas.add(currentRect);
        canvas.setActiveObject(currentRect);
    });

    canvas.on('mouse:move', function(options) {
        if (isDrawing && currentRect) {
            const pointer = canvas.getPointer(options.e);
            const width = pointer.x - startX;
            const height = pointer.y - startY;

            currentRect.set({
                left: width < 0 ? pointer.x : startX,
                top: height < 0 ? pointer.y : startY,
                width: Math.abs(width),
                height: Math.abs(height)
            });

            currentRect.setCoords();
            canvas.renderAll();
        }
    });

    canvas.on('mouse:up', function() {
        isDrawing = false;
        if (currentRect && (currentRect.width < 5 || currentRect.height < 5)) {
            // Remove if too small
            canvas.remove(currentRect);
        }
        currentRect = null;
    });

    // Selection highlight
    canvas.on('selection:created', function() {
        const obj = canvas.getActiveObject();
        if (obj && obj.type === 'rect') {
            obj._prevStroke = obj.stroke; // Store original stroke
            obj.set('stroke', '#e100ffff'); 
            canvas.renderAll();
        }
    });

    canvas.on('selection:cleared', function() {
        canvas.getObjects().forEach(obj => {
            if (obj.type === 'rect' && obj._prevStroke) {
                obj.set('stroke', obj._prevStroke);
                delete obj._prevStroke;
            }
        });
        canvas.renderAll();
    });

    // Delete key - handle differently for original vs user-added boxes
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Delete' && canvas.getActiveObject()) {
            const selectedObject = canvas.getActiveObject();
            // Only allow deleting original annotations in 'modified' view
            if (selectedObject.isOriginal === true) {
                if (currentViewMode === 'modified') {
                    // If original annotation, mark as deleted but keep visible with dashed line
                    selectedObject.set({
                        stroke: '#e74c3c', // Red for deleted
                        strokeDashArray: [5, 5], // Dashed line
                        deleted: true
                    });
                    canvas.renderAll();
                    // Add to deletedAnnotations for backend
                    deletedAnnotations.push({
                        class: selectedObject.classId,
                        x_center: (selectedObject.left + selectedObject.width / 2) / canvas.width,
                        y_center: (selectedObject.top + selectedObject.height / 2) / canvas.height,
                        width: selectedObject.width / canvas.width,
                        height: selectedObject.height / canvas.height,
                        is_original: true
                    });
                    saveAnnotations();
                } else {
                    // Prevent deletion in other modes
                    showNotification('You can only delete YOLO annotations in the Modified view.', 'error');
                }
            } else {
                // If user-added annotation, just remove
                canvas.remove(selectedObject);
                saveAnnotations();
            }
        }
        
        if (e.key === 'Escape') {
            canvas.discardActiveObject();
            canvas.renderAll();
        }
    });
    
    // Navigation buttons
    document.getElementById('prev-btn').onclick = prevImage;
    document.getElementById('next-btn').onclick = nextImage;
    document.getElementById('go-btn').onclick = jumpToImage;

    // Annotation controls
    document.getElementById('discard-all').onclick = discardAll;
    document.getElementById('save-btn').onclick = saveAnnotations;

    // Export button
    document.getElementById('export-btn').onclick = function() {
        window.location.href = `/export_dataset/${projectName}`;
    };

    // View mode toggles
    document.querySelectorAll('.view-mode-btn').forEach(btn => {
        btn.onclick = function() {
            setViewMode(btn.dataset.mode);
        };
    });
    
    // Make canvas responsive
    window.addEventListener('resize', function() {
        centerCanvas();
    });
}

function showNotification(message, type) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type} show`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

function showSpinner(show) {
    document.getElementById('spinner').style.display = show ? 'block' : 'none';
}

function updateImageList() {
    const imageList = document.getElementById('image-list');
    imageList.innerHTML = '';
    
    if (images.length === 0) {
        document.getElementById('empty-state').style.display = 'block';
        return;
    }
    
    images.forEach((img, index) => {
        const item = document.createElement('div');
        item.className = `image-item ${index === currentIndex ? 'active' : ''}`;
        
        const statusIndicator = document.createElement('div');
        statusIndicator.className = 'status-indicator';
        if (imageStatus[img] && imageStatus[img].annotated) {
            statusIndicator.classList.add('annotated');
        }
        
        item.innerHTML = `
            <div class="annotation-status">
                ${statusIndicator.outerHTML}
                <span>${img}</span>
            </div>
        `;
        
        item.addEventListener('click', () => {
            saveAnnotations().then(() => loadImage(index));
        });
        imageList.appendChild(item);
    });
}

async function loadImage(index) {
    if (index < 0 || index >= images.length) return;
    
    showSpinner(true);
    currentIndex = index;

    document.getElementById('image-counter').textContent = `${currentIndex + 1} / ${images.length}`;
    document.getElementById('jump-input').value = currentIndex + 1;
    
    // Update active image in list
    document.querySelectorAll('.image-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });

    canvas.clear();
    canvas.renderAll();

    const imageName = images[index];

    return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = function() {
            showSpinner(false);
            currentImage = new fabric.Image(img, {
                originX: 'left',
                originY: 'top',
                selectable: false,
                evented: false,
                hasControls: false,
                hasBorders: false
            });

            canvas.add(currentImage);
            canvas.setWidth(img.width);
            canvas.setHeight(img.height);

            // Center the canvas in its container
            centerCanvas();

            // Always send image to back
            canvas.sendToBack(currentImage);

            // Reset deleted annotations when loading a new image
            deletedAnnotations = [];

            // Now load annotations after the image is rendered
            loadAnnotations();
            resolve();
        };
        
        img.onerror = function() {
            showSpinner(false);
            console.error('Failed to load image:', images[index]);
            showNotification(`Failed to load image: ${images[index]}`, 'error');
            resolve();
        };
        
        img.src = `/get_image/${projectName}/${encodeURIComponent(images[index])}`;
    });
}

// Center the canvas in its container
function centerCanvas() {
    const canvasWrapper = document.querySelector('.canvas-container');
    const lowerCanvas = canvasWrapper.querySelector('.lower-canvas');
    const upperCanvas = canvasWrapper.querySelector('.upper-canvas');

    const imgWidth = canvas.getWidth();
    const imgHeight = canvas.getHeight();

    // Use the container's size for scaling
    const containerWidth = canvasWrapper.clientWidth;
    const containerHeight = canvasWrapper.clientHeight;
    const scale = Math.min(containerWidth / imgWidth, containerHeight / imgHeight, 1);
    const scaledWidth = imgWidth * scale;
    const scaledHeight = imgHeight * scale;

    // Center the canvases absolutely in the container
    [lowerCanvas, upperCanvas].forEach(cnv => {
        if (cnv) {
            cnv.style.width = `${scaledWidth}px`;
            cnv.style.height = `${scaledHeight}px`;
            cnv.style.position = 'absolute';
            cnv.style.left = '50%';
            cnv.style.top = '50%';
            cnv.style.transform = 'translate(-50%, -50%)';
            cnv.style.maxWidth = '100%';
            cnv.style.maxHeight = '100%';
            cnv.style.margin = '0';
            cnv.style.display = 'block';
        }
    });
    // Also apply to actual canvas element (for direct queries)
    const canvasEl = document.getElementById('canvas');
    if (canvasEl) {
        canvasEl.style.width = `${scaledWidth}px`;
        canvasEl.style.height = `${scaledHeight}px`;
        canvasEl.style.position = 'absolute';
        canvasEl.style.left = '50%';
        canvasEl.style.top = '50%';
        canvasEl.style.transform = 'translate(-50%, -50%)';
        canvasEl.style.maxWidth = '100%';
        canvasEl.style.maxHeight = '100%';
        canvasEl.style.margin = '0';
        canvasEl.style.display = 'block';
    }
    // Set wrapper to relative and fill parent
    canvasWrapper.style.position = 'relative';
    canvasWrapper.style.width = '100%';
    canvasWrapper.style.height = '100%';
}



function loadAnnotations() {
    const imageName = images[currentIndex];
    fetch(`/get_annotations/${projectName}/${encodeURIComponent(imageName)}/${currentViewMode}`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            // Clear existing annotations
            canvas.getObjects().forEach(obj => {
                if (obj.type === 'rect') canvas.remove(obj);
            });

            originalAnnotations = [];

            // Regular annotations
            if (data.annotations && data.annotations.length > 0) {
                data.annotations.forEach(ann => {
                    const width = ann.width * canvas.width;
                    const height = ann.height * canvas.height;
                    const left = ann.x_center * canvas.width - width / 2;
                    const top = ann.y_center * canvas.height - height / 2;

                    const rect = new fabric.Rect({
                        left,
                        top,
                        width,
                        height,
                        fill: 'transparent',
                        strokeWidth: 2,
                        strokeUniform: true,
                        selectable: true,
                        hasControls: true,
                        hasBorders: true,
                        classId: ann.class,
                        isOriginal: ann.is_original
                    });

                    // Color logic
                    if (ann.is_original) {
                        rect.set({ stroke: '#3498db' }); // Blue
                    } else {
                        rect.set({ stroke: '#2ecc71' }); // Green
                    }

                    originalAnnotations.push({
                        class: ann.class,
                        x_center: ann.x_center,
                        y_center: ann.y_center,
                        width: ann.width,
                        height: ann.height
                    });

                    canvas.add(rect);
                    canvas.bringToFront(rect);
                });
            }

            // Deleted/modified annotations
            if (data.deleted_annotations && data.deleted_annotations.length > 0) {
                data.deleted_annotations.forEach(ann => {
                    const width = ann.width * canvas.width;
                    const height = ann.height * canvas.height;
                    const left = ann.x_center * canvas.width - width / 2;
                    const top = ann.y_center * canvas.height - height / 2;

                    // White dotted trace (optional, can be omitted if not needed)
                    // const originalRect = new fabric.Rect({
                    //     left,
                    //     top,
                    //     width,
                    //     height,
                    //     fill: 'transparent',
                    //     stroke: '#ffffff',
                    //     strokeWidth: 1,
                    //     strokeDashArray: [5, 5],
                    //     strokeUniform: true,
                    //     selectable: false,
                    //     hasControls: false,
                    //     hasBorders: false,
                    //     hoverCursor: 'default'
                    // });

                    // Red box for deleted/modified (always draw in modified view)
                    const deletedRect = new fabric.Rect({
                        left,
                        top,
                        width,
                        height,
                        fill: 'transparent',
                        stroke: '#e74c3c',
                        strokeWidth: 2,
                        strokeDashArray: [5, 5],
                        strokeUniform: true,
                        selectable: true,
                        hasControls: true,
                        hasBorders: true,
                        classId: ann.class,
                        isOriginal: true,
                        deleted: true
                    });

                    // Optionally add the white trace below the red box
                    // canvas.add(originalRect);
                    canvas.add(deletedRect);
                    // canvas.sendToBack(originalRect);

                    deletedAnnotations.push({
                        class: ann.class,
                        x_center: ann.x_center,
                        y_center: ann.y_center,
                        width: ann.width,
                        height: ann.height,
                        is_original: true
                    });
                });
            }

            updateImageList();
            canvas.renderAll();
            updateAnnotationCount();
        })
        .catch(error => {
            console.error('Error loading annotations:', error);
            showNotification('Failed to load annotations!', 'error');
        });
}

function updateAnnotationCount() {
    let count = 0;
    canvas.getObjects().forEach(obj => {
        if (obj.type === 'rect' && obj !== currentImage && !obj.deleted) {
            count++;
        }
    });
    document.getElementById('annotation-count').textContent = count;
}

async function saveAnnotations() {
    const annotations = [];
    
    // Collect all non-deleted annotations
    canvas.getObjects().forEach(obj => {
        if (obj.type === 'rect' && obj !== currentImage && !obj.deleted) {
            // Convert to normalized coordinates
            const width = obj.width / canvas.width;
            const height = obj.height / canvas.height;
            const x_center = (obj.left + obj.width / 2) / canvas.width;
            const y_center = (obj.top + obj.height / 2) / canvas.height;

            annotations.push({ 
                class: obj.classId !== undefined ? obj.classId : 0, 
                x_center, 
                y_center, 
                width, 
                height,
                is_original: obj.isOriginal
            });
        }
    });

    try {
        const imageName = images[currentIndex];
        const response = await fetch(`/save_annotations/${projectName}/${encodeURIComponent(imageName)}/${currentViewMode}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                annotations: annotations,
                deleted_annotations: deletedAnnotations
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to save annotations');
        }
        
        // Update annotation status
        imageStatus[imageName] = {
            annotated: annotations.length > 0 || deletedAnnotations.length > 0,
            count: annotations.length
        };
        updateImageList();
        
        showNotification('Annotations saved successfully!', 'success');
        updateAnnotationCount();
        return true;
    } catch (error) {
        console.error('Error saving annotations:', error);
        showNotification('Failed to save annotations!', 'error');
        return false;
    }
}

async function nextImage() {
    if (currentIndex < images.length - 1) {
        await saveAnnotations();
        await loadImage(currentIndex + 1);
    }
}

async function prevImage() {
    if (currentIndex > 0) {
        await saveAnnotations();
        await loadImage(currentIndex - 1);
    }
}

async function jumpToImage() {
    const index = parseInt(document.getElementById('jump-input').value) - 1;
    if (index >= 0 && index < images.length) {
        await saveAnnotations();
        await loadImage(index);
    }
}

function setViewMode(mode) {
    // Update UI
    document.querySelectorAll('.view-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    
    // Set current mode
    currentViewMode = mode;
    
    // Reload annotations with new mode
    loadAnnotations();
    
    showNotification(`View mode changed to: ${mode}`, 'success');
}

// Function to discard all annotations for the current image
async function discardAll() {
    // Mark all original boxes as deleted
    const originalBoxes = canvas.getObjects().filter(obj => 
        obj.type === 'rect' && obj.isOriginal === true && !obj.deleted
    );
    
    // Add all original annotations to deleted list
    originalBoxes.forEach(box => {
        box.set({
            stroke: '#e74c3c',
            strokeDashArray: [5, 5],
            deleted: true
        });
        
        deletedAnnotations.push({
            class: box.classId,
            x_center: (box.left + box.width / 2) / canvas.width,
            y_center: (box.top + box.height / 2) / canvas.height,
            width: box.width / canvas.width,
            height: box.height / canvas.height,
            is_original: true
        });
    });
    
    // Remove all non-original boxes
    const userBoxes = canvas.getObjects().filter(obj => 
        obj.type === 'rect' && obj.isOriginal === false
    );
    
    userBoxes.forEach(box => {
        canvas.remove(box);
    });
    
    canvas.renderAll();
    await saveAnnotations();
    showNotification('All annotations discarded', 'success');
}

// Initialize the projectName and images variables from the template variables
function initFromTemplate(projectNameValue, imagesArray) {
    projectName = projectNameValue;
    images = imagesArray;
    
    // Check if already loaded (for when script loads before DOM)
    if (document.readyState === "complete" || document.readyState === "interactive") {
        if (images.length === 0) {
            document.getElementById('empty-state').style.display = 'block';
        } else {
            initCanvas();
            loadImage(0);
            updateImageList();
        }
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            if (images.length === 0) {
                document.getElementById('empty-state').style.display = 'block';
            } else {
                initCanvas();
                loadImage(0);
                updateImageList();
            }
        });
    }
}