class VisCPM_API:
    
    def __init__(self, url='http://34.87.78.69:3389/viscpm'):
        self.url = url
        import base64
        
    def generate(self, image_path, prompt):
        try:
            im = base64.b64encode(open(image_path, 'rb').read()).decode()
            resp = requests.post(self.url, json={'image': im, 'question': prompt})
            return resp.json()['response']
        except:
            return "E. Failed to obtain the response. "