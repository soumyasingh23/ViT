import os
import cv2

def create_video(folder1, folder2, output_video):
    # Get list of images from folder 1
    folder1_images = sorted([os.path.join(folder1, img) for img in os.listdir(folder1)])
    # Get list of images from folder 2
    folder2_images = sorted([os.path.join(folder2, img) for img in os.listdir(folder2)])

    # Check if both folders have the same number of images
    if len(folder1_images) != len(folder2_images):
        print("Error: Both folders must contain the same number of images.")
        return

    # Initialize video writer
    frame_width = 128
    frame_height = 64
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 1, (frame_width, frame_height))

    for img1_path, img2_path in zip(folder1_images, folder2_images):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Resize images to fit the video frame
        img1 = cv2.resize(img1, (frame_width // 2, frame_height))
        img2 = cv2.resize(img2, (frame_width // 2, frame_height))

        # Combine images horizontally
        combined_img = cv2.hconcat([img1, img2])

        # Write frame to video
        out.write(combined_img)

    # Release video writer
    out.release()
    print("Video created successfully.")

# Example usage:
true_folder = 'data/true'
pred_folder = 'data/pred'
output_video = 'output_video.mp4'
create_video(true_folder, pred_folder, output_video)
