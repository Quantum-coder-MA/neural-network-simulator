import os
# ... other imports

data_dir="/home/nigga/engine/data/cats"
# ... other setup

print("--- Starting File Cleaning ---")
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if os.path.isdir(class_path):
        for image in os.listdir(class_path):
            
            # --- CRITICAL FIX ---
            if ":Zone.Identifier" in image:
                image_path = os.path.join(class_path, image)
                print(f"Removing Alternate Data Stream file: {image_path}")
                try:
                    os.remove(image_path)
                except Exception as e:
                    print(f"Could not remove stream file {image_path}: {e}")
                continue # Skip all other checks for this item
            
            # Continue with your original checks for valid images
            image_path = os.path.join(class_path, image)
            
            # ... rest of your cv2 and PIL checks