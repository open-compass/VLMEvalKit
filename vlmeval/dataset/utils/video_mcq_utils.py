import base64


# video_mcq use base64 for mp4 video encoding and decoding.
# using this code to convert mp4 to base64
def mp4_to_base64(mp4_path):
    try:
        with open(mp4_path, 'rb') as video_file:
            video_data = video_file.read()
            base64_encoded_video = base64.b64encode(video_data).decode('utf-8')
        return base64_encoded_video
    except FileNotFoundError:
        return 'The file was not found.'
    except Exception as e:
        return f'An error occurred: {e}'
